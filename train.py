#!/usr/bin/env python3
import argparse
from collections import defaultdict
import logging
from pathlib import Path
import random
import time
from typing import Dict, Iterable, List

import attr
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import utils


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(module)s: %(message)s'))
logger.addHandler(ch)


@attr.s(slots=True)
class HyperParams:
    n_channels = attr.ib(default=3)
    n_classes = attr.ib(default=10)
    thresholds = attr.ib(default=[0.08, 0.1, 0.12, 0.3, 0.5])

    patch_inner = attr.ib(default=16)
    patch_border = attr.ib(default=8)

    n_epochs = attr.ib(default=10)
    learning_rate = attr.ib(default=0.1)
    batch_size = attr.ib(default=32)

    @property
    def patch_size(self):
        return self.patch_border * 2 + self.patch_inner

    def update(self, hps_string: str):
        if hps_string:
            for pair in hps_string.split(','):
                k, v = pair.split('=')
                v = float(v) if '.' in v else int(v)
                setattr(self, k, v)


@attr.s
class Image:
    id = attr.ib()
    data = attr.ib()
    mask = attr.ib()


class Model:
    def __init__(self, hps: HyperParams):
        self.hps = hps
        self.x = tf.placeholder(
            tf.float32, [None, hps.patch_size, hps.patch_size, hps.n_channels])
        self.y = tf.placeholder(
            tf.float32, [None, hps.patch_inner, hps.patch_inner, hps.n_classes])

        input_dim = hps.patch_size ** 2 * hps.n_channels
        hidden_dim = output_dim = hps.patch_inner ** 2 * hps.n_classes
        x = tf.reshape(self.x, [-1, input_dim])
        w0 = tf.get_variable('w0', shape=[input_dim, hidden_dim])
        b0 = tf.get_variable('b0', shape=[hidden_dim],
                             initializer=tf.zeros_initializer)
        x_hidden = tf.nn.relu(tf.nn.xw_plus_b(x, w0, b0))

        w1 = tf.get_variable('w1', shape=[hidden_dim, output_dim])
        b1 = tf.get_variable('b1', shape=[output_dim],
                             initializer=tf.zeros_initializer)
        x_logits = tf.reshape(
            tf.nn.xw_plus_b(x_hidden, w1, b1),
            [-1, hps.patch_inner, hps.patch_inner, hps.n_classes])
        self.pred = tf.nn.sigmoid(x_logits)
        self.add_image_summaries()
        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(x_logits, self.y))
        tf.summary.scalar('loss', self.loss)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=hps.learning_rate)
        self.train_op = optimizer.minimize(
            self.loss * hps.batch_size, self.global_step)
        self.summary_op = tf.summary.merge_all()

    def add_image_summaries(self):
        images = [self.x]
        for cls in range(self.hps.n_classes):
            images.append(
                tf.concat(1, [tf.pack(3 * [im[:, :, :, cls]], axis=3)
                              for im in [self.y, self.pred]]))
        tf.summary.image('image', tf.concat(2, images), max_outputs=16)

    def train(self, logdir: str, im_ids: List[str]):
        im_ids = sorted(im_ids)
        train_images = [self.load_image(im_id) for im_id in im_ids]
        sv = tf.train.Supervisor(
            logdir=logdir,
            summary_op=None,
            global_step=self.global_step,
            save_summaries_secs=10,
            save_model_secs=60,
        )
        with sv.managed_session() as sess:
            for n_epoch in range(self.hps.n_epochs):
                logger.info('Epoch {}'.format(n_epoch))
                self.train_on_images(train_images, sv, sess)

    def load_image(self, im_id: str) -> Image:
        logger.info('Loading {}'.format(im_id))
        im_data_filename = './im_data/{}.npy'.format(im_id)
        mask_filename = './mask/{}.npy'.format(im_id)
        if all(Path(p).exists() for p in [im_data_filename, mask_filename]):
            im_data = np.load(im_data_filename)
            mask = np.load(mask_filename)
        else:
            im_data = utils.load_image(im_id)
            im_data = utils.scale_percentile(im_data)
            im_size = im_data.shape[:2]
            poly_by_type = utils.load_polygons(im_id, im_size)
            mask = np.array(
                [utils.mask_for_polygons(im_size, poly_by_type[poly_type + 1])
                 for poly_type in range(self.hps.n_classes)],
                dtype=np.uint8).transpose([1, 2, 0])
            with open(im_data_filename, 'wb') as f:
                np.save(f, im_data)
            with open(mask_filename, 'wb') as f:
                np.save(f, mask)
        return Image(im_id, im_data, mask)

    def train_on_images(self, train_images: List[Image],
                        sv: tf.train.Supervisor, sess: tf.Session):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        avg_area = np.mean(
            [im.data.shape[0] * im.data.shape[1] for im in train_images])
        n_batches = int(avg_area / (s + b) / self.hps.batch_size)

        def feeds():
            for _ in range(n_batches):
                inputs, outputs = [], []
                for _ in range(self.hps.batch_size):
                    im = random.choice(train_images)
                    w, h = im.data.shape[:2]
                    x, y = (random.randint(b, w - (b + s)),
                            random.randint(b, h - (b + s)))
                    inputs.append(im.data[x - b: x + s + b,
                                          y - b: y + s + b, :])
                    outputs.append(im.mask[x: x + s, y: y + s, :])
                # import IPython; IPython.embed()
                yield {self.x: np.array(inputs), self.y: np.array(outputs)}

        self._train_on_feeds(feeds(), sv, sess)

    def _train_on_feeds(self, feed_dicts: Iterable[Dict],
                        sv: tf.train.Supervisor, sess: tf.Session):
        losses = []
        tp_fp_fn = {threshold: [defaultdict(list) for _ in range(3)]
                    for threshold in self.hps.thresholds}
        t0 = t00 = time.time()
        for i, feed_dict in enumerate(feed_dicts):
            fetches = {'loss': self.loss, 'train': self.train_op}
            if i % 10 == 0:
                fetches['summary'] = self.summary_op
                fetches['pred'] = self.pred
            fetched = sess.run(fetches, feed_dict)
            losses.append(fetched['loss'])
            if 'pred' in fetched:
                self._update_jaccard(
                    tp_fp_fn, feed_dict[self.y], fetched['pred'])
                # TODO - log pred summary
            if 'summary' in fetched:
                sv.summary_computed(sess, fetched['summary'])
            t1 = time.time()
            dt = t1 - t0
            if dt > 30 or (t1 - t00 < 30 and dt > 5):
                logger.info(
                    'Loss: {loss:.3f}, speed: {speed:,} patches/s, '
                    'Jaccard: {jaccard}'.format(
                        loss=np.mean(losses),
                        speed=int(len(losses) * self.hps.batch_size / dt),
                        jaccard=', '.join(
                            'at {:.2f}: {:.3f}'.format(
                                threshold, self._jaccard(tp, fp, fn))
                            for threshold, (tp, fp, fn)
                            in sorted(tp_fp_fn.items()))
                    ))
                losses = []
                for for_threshold in tp_fp_fn.values():
                    for dct in for_threshold:
                        dct.clear()
                t0 = t1

    def _update_jaccard(self, tp_fp_fn, mask, pred):
        for threshold, (tp, fp, fn) in tp_fp_fn.items():
            for cls in range(self.hps.n_classes):
                pred_ = pred[:, :, cls]
                mask_ = mask[:, :, cls]
                tp[cls].append(((pred_ >= threshold) * (mask_ == 1)).sum())
                fp[cls].append(((pred_ >= threshold) * (mask_ == 0)).sum())
                fn[cls].append(((pred_ <  threshold) * (mask_ == 1)).sum())

    def _jaccard(self, tp, fp, fn):
        return np.mean([
            sum(tp[cls]) / (sum(tp[cls]) + sum(fn[cls]) + sum(fp[cls]))
            for cls in range(self.hps.n_classes)])


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', type=str, help='Path to log directory')
    arg('--hps', type=str, help='Change hyperparameters in k1=v1,k2=v2 format')
    args = parser.parse_args()
    hps = HyperParams()
    hps.update(args.hps)

    model = Model(hps=hps)
    all_img_ids = list(utils.get_wkt_data())
    train_ids, valid_ids = train_test_split(all_img_ids, random_state=0)
    random.seed(0)
    model.train(logdir=args.logdir, im_ids=train_ids)


if __name__ == '__main__':
    main()