#!/usr/bin/env python3
import argparse
import logging
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
        x_logits = tf.nn.xw_plus_b(x_hidden, w1, b1)
        self.x_out = tf.nn.sigmoid(x_logits)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                x_logits, tf.reshape(self.y, [-1, output_dim])))
        tf.summary.scalar('loss', self.loss)
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=hps.learning_rate)
        self.train_op = optimizer.minimize(
            self.loss * hps.batch_size, self.global_step)
        self.summary_op = tf.summary.merge_all()

    def train(self, logdir: str, im_ids: List[str]):
        im_ids = list(im_ids)
        sv = tf.train.Supervisor(
            logdir=logdir,
            summary_op=None,
            global_step=self.global_step,
            save_summaries_secs=10,
            save_model_secs=60,
        )
        with sv.managed_session() as sess:
            for n_epoch in range(self.hps.n_epochs):
                random.shuffle(im_ids)
                for im_id in im_ids:
                    self.train_on_image(im_id, sv, sess)

    def train_on_image(self, im_id: str,
                       sv: tf.train.Supervisor, sess: tf.Session):
        logger.info('Loading {}'.format(im_id))
        hps = self.hps
        im_data = utils.load_image(im_id)
        im_data = utils.scale_percentile(im_data)
        im_size = w, h = im_data.shape[:2]
        logger.info('Loading polygons')
        poly_by_type = utils.load_polygons(im_id, im_size)
        logger.info('Computing masks')
        masks = np.array(
            [utils.mask_for_polygons(im_size, poly_by_type[poly_type + 1])
             for poly_type in range(hps.n_classes)],
            dtype=np.float32).transpose([1, 2, 0])
        b = hps.patch_border
        s = hps.patch_inner
        all_patch_coords = [(
            # TODO - on edges too
            random.randint(b, w - (b + s)),
            random.randint(b, h - (b + s)))
            for _ in range(w * h // s)]
        logger.info('Starting training')

        def feeds():
            for patch_coords in utils.chunks(all_patch_coords, hps.batch_size):
                inputs = np.array(
                    [im_data[x - b: x + s + b, y - b: y + s + b, :]
                     for x, y in patch_coords])
                outputs = np.array(
                    [masks[x: x + s, y: y + s, :] for x, y in patch_coords])
                yield {self.x: inputs, self.y: outputs}

        self._train_on_feeds(feeds(), sv, sess)

    def _train_on_feeds(self, feed_dicts: Iterable[Dict],
                        sv: tf.train.Supervisor, sess: tf.Session):
        losses = []
        t0 = t00 = time.time()
        for i, feed_dict in enumerate(feed_dicts):
            fetches = {'loss': self.loss, 'train': self.train_op}
            if i % 10 == 0:
                fetches['summary'] = self.summary_op
            fetched = sess.run(fetches, feed_dict)
            losses.append(fetched['loss'])
            if 'summary' in fetched:
                sv.summary_computed(sess, fetched['summary'])
            t1 = time.time()
            dt = t1 - t0
            if dt > 60 or (t1 - t00 < 60 and dt > 5):
                logger.info('Loss: {:.3f}, speed: {:,} patches/s'.format(
                    np.mean(losses),
                    int(len(losses) * self.hps.batch_size / dt)))
                losses = []
                t0 = t1




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
    train_ids, valid_ids = train_test_split(all_img_ids)
    model.train(logdir=args.logdir, im_ids=train_ids)


if __name__ == '__main__':
    main()