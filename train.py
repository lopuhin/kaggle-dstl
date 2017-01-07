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
    thresholds = attr.ib(default=[0.3, 0.4, 0.5])

    patch_inner = attr.ib(default=32)
    patch_border = attr.ib(default=16)

    n_epochs = attr.ib(default=10)
    learning_rate = attr.ib(default=0.001)
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
    mask = attr.ib(default=None)


class Model:
    def __init__(self, hps: HyperParams):
        self.hps = hps
        self.x = tf.placeholder(
            tf.float32, [None, hps.patch_size, hps.patch_size, hps.n_channels])
        self.y = tf.placeholder(
            tf.float32, [None, hps.patch_inner, hps.patch_inner, hps.n_classes])

        w0 = tf.get_variable('w0', shape=[5, 5, 3, 32])
        b0 = tf.get_variable('b0', shape=[32],
                             initializer=tf.zeros_initializer)
        conv2d = lambda _x, _w: tf.nn.conv2d(
            _x, _w, strides=[1, 1, 1, 1], padding='SAME')
        x0 = tf.nn.relu(conv2d(self.x, w0) + b0)

        w1 = tf.get_variable('w1', shape=[5, 5, 32, hps.n_classes])
        b1 = tf.get_variable('b1', shape=[hps.n_classes],
                             initializer=tf.zeros_initializer)
        x1 = conv2d(x0, w1) + b1
        b = hps.patch_border
        x_logits = x1[:, b:-b, b:-b, :]
        self.pred = tf.nn.sigmoid(x_logits)
        self.add_image_summaries()
        losses = tf.nn.sigmoid_cross_entropy_with_logits(x_logits, self.y)
        self.cls_losses = [tf.reduce_mean(losses[:, :, :, cls])
                           for cls in range(hps.n_classes)]
        self.loss = tf.reduce_mean(losses)
        self.add_loss_summaries()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=hps.learning_rate)
        self.train_op = optimizer.minimize(
            self.loss * hps.batch_size, self.global_step)
        self.summary_op = tf.summary.merge_all()

    def add_loss_summaries(self):
        tf.summary.scalar('loss/average', self.loss)
        for cls, loss in enumerate(self.cls_losses):
            tf.summary.scalar('loss/cls-{}'.format(cls), loss)

    def add_image_summaries(self):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        border = np.zeros([b * 2 + s, b * 2 + s, 3], dtype=np.float32)
        border[ b, b:-b, 0] = border[-b, b:-b, 0] = 1
        border[b:-b,  b, 0] = border[b:-b, -b, 0] = 1
        border[-b, -b, 0] = 1
        border_t = tf.pack(self.hps.batch_size * [tf.constant(border)])
        images = [tf.maximum(self.x, border_t)]
        mark = np.zeros([s, s], dtype=np.float32)
        mark[0, 0] = 1
        mark_t = tf.pack(self.hps.batch_size * [tf.constant(mark)])
        for cls in range(self.hps.n_classes):
            images.append(
                tf.concat(1, [
                    tf.pack(3 * [tf.maximum(im[:, :, :, cls], mark_t)], axis=3)
                    for im in [self.y, self.pred]]))
        tf.summary.image('image', tf.concat(2, images), max_outputs=8)

    def train(self, logdir: str, train_ids: List[str], valid_ids: List[str]):
        train_images = [self.load_image(im_id) for im_id in sorted(train_ids)]
        sv = tf.train.Supervisor(
            logdir=logdir,
            summary_op=None,
            global_step=self.global_step,
            save_summaries_secs=10,
            save_model_secs=60,
        )
        valid_images = None
        with sv.managed_session() as sess:
            for n_epoch in range(self.hps.n_epochs):
                logger.info('Epoch {}, training'.format(n_epoch + 1))
                self.train_on_images(train_images, sv, sess)
                if valid_images is None:
                    valid_images = [self.load_image(im_id)
                                    for im_id in sorted(valid_ids)]
                logger.info('Epoch {}, validation'.format(n_epoch + 1))
                self.validate_on_images(valid_images, sv, sess)

    def preprocess_image(self, im_data: np.ndarray) -> np.ndarray:
        return utils.scale_percentile(im_data)

    def load_image(self, im_id: str) -> Image:
        logger.info('Loading {}'.format(im_id))
        im_data_filename = './im_data/{}.npy'.format(im_id)
        mask_filename = './mask/{}.npy'.format(im_id)
        if all(Path(p).exists() for p in [im_data_filename, mask_filename]):
            im_data = np.load(im_data_filename)
            mask = np.load(mask_filename)
        else:
            im_data = self.preprocess_image(utils.load_image(im_id))
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
                yield {self.x: np.array(inputs), self.y: np.array(outputs)}

        self._train_on_feeds(feeds(), sv=sv, sess=sess)

    def _train_on_feeds(self, feed_dicts: Iterable[Dict],
                        sv: tf.train.Supervisor, sess: tf.Session):
        losses = []
        tp_fp_fn = self._jaccard_store()

        def log():
            logger.info(
                'Loss: {loss:.3f}, speed: {speed:,} patches/s, '
                'Jaccard: {jaccard}'.format(
                    loss=np.mean(losses),
                    speed=int(len(losses) * self.hps.batch_size / dt),
                    jaccard=self._format_jaccard(tp_fp_fn),
                ))

        t0 = t00 = time.time()
        for i, feed_dict in enumerate(feed_dicts):
            fetches = {'loss': self.loss, 'train': self.train_op}
            if i % 20 == 0:
                fetches['summary'] = self.summary_op
                fetches['pred'] = self.pred
            fetched = sess.run(fetches, feed_dict)
            losses.append(fetched['loss'])
            if 'pred' in fetched:
                self._update_jaccard(
                    tp_fp_fn, feed_dict[self.y], fetched['pred'])
                self._log_jaccard(tp_fp_fn, sv, sess)
            if 'summary' in fetched:
                sv.summary_computed(sess, fetched['summary'])
            t1 = time.time()
            dt = t1 - t0
            if dt > 30 or (t1 - t00 < 30 and dt > 5):
                log()
                losses = []
                tp_fp_fn = self._jaccard_store()
                t0 = t1
        log()

    def _jaccard_store(self):
        return {threshold: [defaultdict(list) for _ in range(3)]
                for threshold in self.hps.thresholds}

    def _update_jaccard(self, tp_fp_fn, mask, pred):
        for threshold, (tp, fp, fn) in tp_fp_fn.items():
            for cls in range(self.hps.n_classes):
                pos_pred = pred[:, :, cls] >= threshold
                pos_mask = mask[:, :, cls] == 1
                tp[cls].append(( pos_pred &  pos_mask).sum())
                fp[cls].append(( pos_pred & ~pos_mask).sum())
                fn[cls].append((~pos_pred &  pos_mask).sum())

    def _log_jaccard(self, tp_fp_fn, sv, sess, prefix=''):
        for threshold, (tp, fp, fn) in tp_fp_fn.items():
            jaccards = []
            for cls in range(self.hps.n_classes):
                jaccard = self._cls_jaccard(tp, fp, fn, cls)
                self._log_summary(
                    '{}jaccard-{}/cls-{}'.format(prefix, threshold, cls),
                    jaccard, sv, sess)
                jaccards.append(jaccard)
            self._log_summary(
                '{}jaccard-{}/average'.format(prefix, threshold),
                np.mean(jaccards), sv, sess)

    def _jaccard(self, tp, fp, fn):
        return np.mean([self._cls_jaccard(tp, fp, fn, cls)
                        for cls in range(self.hps.n_classes)])

    def _cls_jaccard(self, tp, fp, fn, cls):
        if sum(tp[cls]) == 0:
            return 0
        return sum(tp[cls]) / (sum(tp[cls]) + sum(fn[cls]) + sum(fp[cls]))

    def _format_jaccard(self, tp_fp_fn):
        return ', '.join(
            'at {:.2f}: {:.3f}'.format(
                threshold, self._jaccard(tp, fp, fn))
            for threshold, (tp, fp, fn)
            in sorted(tp_fp_fn.items()))

    def validate_on_images(self, valid_images: List[Image],
                           sv: tf.train.Supervisor, sess: tf.Session):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        losses = []
        tp_fp_fn = self._jaccard_store()
        for im in valid_images:
            logger.info(im.id)
            w, h = im.data.shape[:2]
            xs = range(b, w - (b + s), s)
            ys = range(b, h - (b + s), s)
            all_xy = [(x, y) for x in xs for y in ys]
            pred_mask = np.zeros([w, h, self.hps.n_classes])
            for xy_batch in utils.chunks(all_xy, 16 * self.hps.batch_size):
                inputs = np.array([im.data[x - b: x + s + b,
                                           y - b: y + s + b, :]
                                   for x, y in xy_batch])
                outputs = np.array([im.mask[x: x + s, y: y + s, :]
                                    for x, y in xy_batch])
                feed_dict = {self.x: inputs, self.y: outputs}
                cls_losses, pred = sess.run([
                    self.cls_losses, self.pred], feed_dict)
                losses.append(cls_losses)
                for (x, y), mask in zip(xy_batch, pred):
                    pred_mask[x: x + s, y: y + s, :] = mask
            self._update_jaccard(tp_fp_fn, im.mask, pred_mask)
        losses = np.array(losses)
        loss = np.mean(losses)
        logger.info('Validation loss: {:.3f}, jaccard: {}'.format(
            loss, self._format_jaccard(tp_fp_fn)))
        self._log_summary('valid-loss/average', loss, sv, sess)
        for cls in range(self.hps.n_classes):
            self._log_summary('valid-loss/cls-{}'.format(cls),
                              np.mean(losses[:, cls]), sv, sess)
        self._log_jaccard(tp_fp_fn, sv, sess, prefix='valid-')

    def image_prediction(self, im: Image, sess: tf.Session):
        # FIXME - some copy-paste
        w, h = im.data.shape[:2]
        b = self.hps.patch_border
        s = self.hps.patch_inner
        xs = range(b, w - (b + s), s)
        ys = range(b, h - (b + s), s)
        all_xy = [(x, y) for x in xs for y in ys]
        pred_mask = np.zeros([w, h, self.hps.n_classes])
        for xy_batch in utils.chunks(all_xy, 16 * self.hps.batch_size):
            inputs = np.array([im.data[x - b: x + s + b,
                                       y - b: y + s + b, :]
                               for x, y in xy_batch])
            feed_dict = {self.x: inputs}
            pred = sess.run(self.pred, feed_dict)
            for (x, y), mask in zip(xy_batch, pred):
                pred_mask[x: x + s, y: y + s, :] = mask
        return pred_mask

    def _log_summary(self, name: str, value,
                     sv: tf.train.Supervisor, sess: tf.Session):
        summary = tf.Summary()
        summary.value.add(tag=name, simple_value=float(value))
        sv.summary_computed(sess, summary,
                            global_step=self.global_step.eval(session=sess))


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
    model.train(logdir=args.logdir, train_ids=train_ids, valid_ids=valid_ids)


if __name__ == '__main__':
    main()