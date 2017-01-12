#!/usr/bin/env python3
import argparse
from collections import defaultdict
import logging
from pathlib import Path
from multiprocessing.pool import ThreadPool
import random
import time
from typing import Callable, Dict, List

import attr
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
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
    n_channels = attr.ib(default=20)
    classes = attr.ib(default=range(10))
    total_classes = 10
    thresholds = attr.ib(default=[0.3, 0.4, 0.5])

    patch_inner = attr.ib(default=32)
    patch_border = attr.ib(default=16)

    dropout_keep_prob = attr.ib(default=0.0)
    size1 = attr.ib(default=5)
    filters1 = attr.ib(default=64)
    size2 = attr.ib(default=5)
    filters2 = attr.ib(default=64)
    size3 = attr.ib(default=5)
    filters3 = attr.ib(default=64)
    size4 = attr.ib(default=7)

    n_epochs = attr.ib(default=10)
    learning_rate = attr.ib(default=0.0001)
    batch_size = attr.ib(default=128)

    @property
    def n_classes(self):
        return len(self.classes)

    @property
    def has_all_classes(self):
        return self.n_classes == self.total_classes

    def update(self, hps_string: str):
        if hps_string:
            for pair in hps_string.split(';'):
                k, v = pair.split('=')
                if k == 'classes':
                    v = [int(cls) for cls in v.split(',')]
                elif '.' in v:
                    v = float(v)
                else:
                    v = int(v)
                setattr(self, k, v)


@attr.s
class Image:
    id = attr.ib()
    data = attr.ib()
    mask = attr.ib(default=None)


class Model:
    def __init__(self, hps: HyperParams):
        self.hps = hps
        patch_size = 2 * hps.patch_border + hps.patch_inner
        self.x = tf.placeholder(
            tf.float32, [None, patch_size, patch_size, hps.n_channels])
        self.y = tf.placeholder(
            tf.float32, [None, hps.patch_inner, hps.patch_inner, hps.n_classes])
        self.dropout_keep_prob = tf.placeholder(tf.float32, [])

        conv2d = lambda _x, _w: tf.nn.conv2d(
            _x, _w, strides=[1, 1, 1, 1], padding='SAME')

        w1 = tf.get_variable(
            'w1', shape=[hps.size1, hps.size1, hps.n_channels, hps.filters1])
        b1 = tf.get_variable(
            'b1', shape=[hps.filters1], initializer=tf.zeros_initializer)
        x = tf.nn.relu(conv2d(self.x, w1) + b1)

        w2 = tf.get_variable(
            'w2', shape=[hps.size2, hps.size2, hps.filters1, hps.filters2])
        b2 = tf.get_variable(
            'b2', shape=[hps.filters2], initializer=tf.zeros_initializer)
        x = tf.nn.relu(conv2d(x, w2) + b2)

        if hps.dropout_keep_prob:
            x = tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)

        w3 = tf.get_variable(
            'w3', shape=[hps.size3, hps.size3, hps.filters2, hps.filters3])
        b3 = tf.get_variable(
            'b3', shape=[hps.filters3], initializer=tf.zeros_initializer)
        x = tf.nn.relu(conv2d(x, w3) + b3)

        if hps.dropout_keep_prob:
            x = tf.nn.dropout(x, keep_prob=self.dropout_keep_prob)

        w4 = tf.get_variable(
            'w4', shape=[hps.size4, hps.size4, hps.filters3, hps.n_classes])
        b4 = tf.get_variable(
            'b4', shape=[hps.n_classes], initializer=tf.zeros_initializer)
        x = conv2d(x, w4) + b4

        b = hps.patch_border
        x_logits = x[:, b:-b, b:-b, :]
        self.pred = tf.nn.sigmoid(x_logits)
        self.add_image_summaries()
        losses = tf.nn.sigmoid_cross_entropy_with_logits(x_logits, self.y)
        self.cls_losses = [tf.reduce_mean(losses[:, :, :, cls_idx])
                           for cls_idx in range(hps.n_classes)]
        self.loss = tf.reduce_mean(losses)
        self.add_loss_summaries()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=hps.learning_rate)
        self.train_op = optimizer.minimize(
            self.loss * hps.batch_size, self.global_step)
        self.summary_op = tf.summary.merge_all()

    def add_loss_summaries(self):
        if self.hps.has_all_classes:
            tf.summary.scalar('loss/average', self.loss)
        for cls_idx, loss in enumerate(self.cls_losses):
            tf.summary.scalar(
                'loss/cls-{}'.format(self.hps.classes[cls_idx]), loss)

    def add_image_summaries(self):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        border = np.zeros([b * 2 + s, b * 2 + s, 3], dtype=np.float32)
        border[ b, b:-b, 0] = border[-b, b:-b, 0] = 1
        border[b:-b,  b, 0] = border[b:-b, -b, 0] = 1
        border[-b, -b, 0] = 1
        border_t = tf.pack(self.hps.batch_size * [tf.constant(border)])
        # TODO: would be cool to add other channels as well
        # TODO: fix image color range
        images = [tf.maximum(self.x[:, :, :, :3], border_t)]
        mark = np.zeros([s, s], dtype=np.float32)
        mark[0, 0] = 1
        mark_t = tf.pack(self.hps.batch_size * [tf.constant(mark)])
        for cls_idx in range(self.hps.n_classes):
            images.append(
                tf.concat(1, [
                    tf.pack(3 * [tf.maximum(im[:, :, :, cls_idx], mark_t)], axis=3)
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
                subsample = 10
                for _ in range(subsample):
                    self.train_on_images(train_images, sv, sess,
                                         subsample=subsample)
                    if valid_images is None:
                        valid_images = [self.load_image(im_id)
                                        for im_id in sorted(valid_ids)]
                    if valid_images:
                        self.validate_on_images(valid_images, sv, sess,
                                                subsample=subsample)

    def preprocess_image(self, im_data: np.ndarray) -> np.ndarray:
        # mean = np.mean(im_data, axis=(0, 1))
        # std = np.std(im_data, axis=(0, 1))
        std = np.array([
            62.00827863,  46.65453694,  24.7612776,   54.50255552,
            13.48645938,  24.76103598,  46.52145521,  62.36207267,
            61.54443128,  59.2848377,   85.72930307,  68.62678882,
            448.43441827, 634.79572682, 567.21509273, 523.10079804,
            530.42441592, 461.8304455,  486.95994727, 478.63768386],
            dtype=np.float32)
        mean = np.array([
            413.62140162,  459.99189475,  325.6722122,   502.57730746,
            294.6884949,   325.82117752,  460.0356966,   482.39001004,
            413.79388678,  527.57681818,  678.22878001,  529.64198655,
            4243.25847972, 4473.47956815, 4178.84648439, 3708.16482918,
            2887.49330138, 2589.61786722, 2525.53347208, 2417.23798598],
            dtype=np.float32)
        return ((im_data - mean) / std).astype(np.float16)

    def load_image(self, im_id: str) -> Image:
        logger.info('Loading {}'.format(im_id))
        im_data_filename = './im_data/{}.npy'.format(im_id)
        mask_filename = './mask/{}.npy'.format(im_id)
        if Path(im_data_filename).exists():
            im_data = np.load(im_data_filename)
        else:
            im_data = self.preprocess_image(utils.load_image(im_id))
            with open(im_data_filename, 'wb') as f:
                np.save(f, im_data)
        if Path(mask_filename).exists():
            mask = np.load(mask_filename)
        else:
            im_size = im_data.shape[:2]
            poly_by_type = utils.load_polygons(im_id, im_size)
            mask = np.array(
                [utils.mask_for_polygons(im_size, poly_by_type[poly_type + 1])
                 for poly_type in range(self.hps.total_classes)],
                dtype=np.uint8).transpose([1, 2, 0])
            with open(mask_filename, 'wb') as f:
                np.save(f, mask)
        if not self.hps.has_all_classes:
            mask = np.stack([mask[:, :, cls] for cls in self.hps.classes], 2)
        return Image(im_id, im_data, mask)

    def train_on_images(self, train_images: List[Image],
                        sv: tf.train.Supervisor, sess: tf.Session,
                        subsample: int=1):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        # Extra margin for rotation
        m = int(np.ceil((np.sqrt(2) - 1) * (b + s / 2)))
        mb = m + b  # full margin
        avg_area = np.mean(
            [im.data.shape[0] * im.data.shape[1] for im in train_images])
        n_batches = int(avg_area / (s + b) / self.hps.batch_size / subsample)

        def gen_batch(_):
            inputs, outputs = [], []
            for _ in range(self.hps.batch_size):
                im = random.choice(train_images)
                w, h = im.data.shape[:2]
                x, y = (random.randint(mb, w - (mb + s)),
                        random.randint(mb, h - (mb + s)))
                patch = im.data[x - mb: x + s + mb, y - mb: y + s + mb, :]
                mask = im.mask[x - m: x + s + m, y - m: y + s + m, :]
                # TODO - mirror flips
                angle = random.random() * 360
                patch = utils.rotated(patch.astype(np.float32), angle)
                mask = utils.rotated(mask.astype(np.float32), angle)
                inputs.append(patch[m: -m, m: -m, :])
                outputs.append(mask[m: -m, m: -m, :])
                # TODO - check that they are still aligned
            return {self.x: np.array(inputs),
                    self.y: np.array(outputs),
                    self.dropout_keep_prob: self.hps.dropout_keep_prob}

        self._train_on_feeds(gen_batch, n_batches, sv=sv, sess=sess)

    def _train_on_feeds(self, gen_batch: Callable[[int], Dict], n_batches: int,
                        sv: tf.train.Supervisor, sess: tf.Session):
        losses = []
        tp_fp_fn = self._jaccard_store()

        def log():
            logger.info(
                'Train loss: {loss:.3f}, Jaccard: {jaccard}, '
                'speed: {speed:,} patches/s'.format(
                    loss=np.mean(losses),
                    speed=int(len(losses) * self.hps.batch_size / dt),
                    jaccard=self._format_jaccard(tp_fp_fn),
                ))

        t0 = time.time()
        with ThreadPool(processes=4) as pool:
            for i, feed_dict in enumerate(pool.imap_unordered(
                    gen_batch, range(n_batches), chunksize=16)):
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
                if dt > 10:
                    log()
                    losses = []
                    tp_fp_fn = self._jaccard_store()
                    t0 = t1
            if losses:
                log()

    def _jaccard_store(self):
        return {threshold: [defaultdict(list) for _ in range(3)]
                for threshold in self.hps.thresholds}

    def _update_jaccard(self, tp_fp_fn, mask, pred):
        assert len(mask.shape) == len(pred.shape)
        assert len(mask.shape) in {3, 4}
        for threshold, (tp, fp, fn) in tp_fp_fn.items():
            for cls_idx, cls in enumerate(self.hps.classes):
                if len(mask.shape) == 4:
                    cls_pred = pred[:, :, :, cls_idx]
                    cls_mask = mask[:, :, :, cls_idx]
                else:
                    cls_pred = pred[:, :, cls_idx]
                    cls_mask = mask[:, :, cls_idx]
                pos_pred = cls_pred >= threshold
                pos_mask = cls_mask == 1
                tp[cls].append(( pos_pred &  pos_mask).sum())
                fp[cls].append(( pos_pred & ~pos_mask).sum())
                fn[cls].append((~pos_pred &  pos_mask).sum())

    def _log_jaccard(self, tp_fp_fn, sv, sess, prefix=''):
        for threshold, (tp, fp, fn) in tp_fp_fn.items():
            jaccards = []
            for cls in self.hps.classes:
                jaccard = self._cls_jaccard(tp, fp, fn, cls)
                self._log_summary(
                    '{}jaccard-{}/cls-{}'.format(prefix, threshold, cls),
                    jaccard, sv, sess)
                jaccards.append(jaccard)
            if self.hps.has_all_classes:
                self._log_summary(
                    '{}jaccard-{}/average'.format(prefix, threshold),
                    np.mean(jaccards), sv, sess)

    def _jaccard(self, tp, fp, fn):
        return np.mean([self._cls_jaccard(tp, fp, fn, cls)
                        for cls in self.hps.classes])

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
                           sv: tf.train.Supervisor, sess: tf.Session,
                           subsample: int=1):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        losses = []
        tp_fp_fn = self._jaccard_store()
        for im in valid_images:
            w, h = im.data.shape[:2]
            xs = range(b, w - (b + s), s)
            ys = range(b, h - (b + s), s)
            all_xy = [(x, y) for x in xs for y in ys]
            if subsample != 1:
                random.shuffle(all_xy)
                all_xy = all_xy[:len(all_xy) // subsample]
            pred_mask = np.zeros([w, h, self.hps.n_classes])
            for xy_batch in utils.chunks(all_xy, self.hps.batch_size):
                inputs = np.array([im.data[x - b: x + s + b,
                                           y - b: y + s + b, :]
                                   for x, y in xy_batch])
                outputs = np.array([im.mask[x: x + s, y: y + s, :]
                                    for x, y in xy_batch])
                feed_dict = {self.x: inputs, self.y: outputs,
                             self.dropout_keep_prob: 1.0}
                cls_losses, pred = sess.run([
                    self.cls_losses, self.pred], feed_dict)
                losses.append(cls_losses)
                for (x, y), mask in zip(xy_batch, pred):
                    pred_mask[x: x + s, y: y + s, :] = mask
                self._update_jaccard(tp_fp_fn, outputs, pred)
        losses = np.array(losses)
        loss = np.mean(losses)
        logger.info('Valid loss: {:.3f}, Jaccard: {}'.format(
            loss, self._format_jaccard(tp_fp_fn)))
        if self.hps.has_all_classes:
            self._log_summary('valid-loss/average', loss, sv, sess)
        for cls, cls_name in enumerate(self.hps.classes):
            self._log_summary('valid-loss/cls-{}'.format(cls_name),
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
        for xy_batch in utils.chunks(all_xy, self.hps.batch_size):
            inputs = np.array([im.data[x - b: x + s + b,
                                       y - b: y + s + b, :]
                               for x, y in xy_batch])
            feed_dict = {self.x: inputs, self.dropout_keep_prob: 1.0}
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
    arg('--hps', type=str, help='Change hyperparameters in k1=v1;k2=v2 format')
    arg('--all', action='store_true',
        help='Train on all images without validation')
    arg('--only', type=str,
        help='Train on this image ids only (comma-separated) without validation')
    args = parser.parse_args()
    hps = HyperParams()
    hps.update(args.hps)

    model = Model(hps=hps)
    all_img_ids = list(utils.get_wkt_data())
    valid_ids = []
    if args.only:
        train_ids = args.only.split(',')
    elif args.all:
        train_ids = all_img_ids
    else:
        # Fix for images of the same place in different seasons
        labels = [im_id.replace('6110', '6140')
                       .replace('6020', '6130')
                       .replace('6030', '6150')
                  for im_id in all_img_ids]
        train_ids, valid_ids = [[all_img_ids[idx] for idx in g] for g in next(
            GroupShuffleSplit(random_state=1).split(all_img_ids, groups=labels))]
        logger.info('Train: {}'.format(' '.join(sorted(train_ids))))
        logger.info('Valid: {}'.format(' '.join(sorted(valid_ids))))
    random.seed(0)
    model.train(logdir=args.logdir, train_ids=train_ids, valid_ids=valid_ids)


if __name__ == '__main__':
    main()