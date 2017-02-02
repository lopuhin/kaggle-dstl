#!/usr/bin/env python3
import argparse
import json
from functools import partial
from pathlib import Path
from multiprocessing.pool import ThreadPool
from pprint import pprint
import random
import time
from typing import List

import attr
import cv2
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import tensorboard_logger
import torch
from torch.autograd import Variable
import torch.cuda
import torch.optim as optim
import torch.nn as nn
import tqdm

import utils
from models import HyperParams
import models


logger = utils.get_logger(__name__)


@attr.s
class Image:
    id = attr.ib()
    data = attr.ib()
    mask = attr.ib(default=None)

    @property
    def size(self):
        assert self.data.shape[0] <= 20
        return self.data.shape[1:]


class Model:
    def __init__(self, hps: HyperParams):
        self.hps = hps
        self.net = getattr(models, hps.net)(hps)
        self.bce_loss = nn.BCELoss()
        self.optimizer = optim.Adam(self.net.parameters(), lr=hps.lr)
        self.tb_logger = None  # type: tensorboard_logger.Logger
        self.logdir = None  # type: Path
        self.on_gpu = torch.cuda.is_available()
        if self.on_gpu:
            self.net.cuda()

    def _var(self, x):
        return Variable(x.cuda() if self.on_gpu else x)

    def train_step(self, x, y):
        x, y = self._var(x), self._var(y)
        self.optimizer.zero_grad()
        y_pred = self.net(x)
        batch_size = x.size()[0]
        losses = self.losses(y, y_pred)
        loss = losses[0]
        for l in losses[1:]:
            loss += l
        (loss * batch_size).backward()
        self.optimizer.step()
        self.net.global_step += 1
        return losses

    def losses(self, ys, y_preds):
        losses = []
        for cls_idx in range(self.hps.n_classes):
            y, y_pred = ys[:, cls_idx], y_preds[:, cls_idx]
            loss = self.bce_loss(y_pred, y)
            if self.hps.dice_loss:
                intersection = (y_pred * y).sum()
                uwi = y_pred.sum() + y.sum()  # without intersection union
                if uwi[0] != 0:
                    loss = (loss / self.hps.dice_loss + (1 - intersection / uwi))
            losses.append(loss)
        return losses

    def train(self, logdir: Path, train_ids: List[str], valid_ids: List[str],
              validation: str, no_mp: bool=False):
        self.tb_logger = tensorboard_logger.Logger(str(logdir))
        self.logdir = logdir
        train_images = [self.load_image(im_id) for im_id in sorted(train_ids)]
        valid_images = None
        n_epoch = self.restore_last_snapshot(logdir)
        square_validation = validation == 'square'
        for n_epoch in range(n_epoch, self.hps.n_epochs):
            logger.info('Epoch {}, training'.format(n_epoch + 1))
            subsample = 4  # make validation more often
            for _ in range(subsample):
                self.train_on_images(
                    train_images,
                    subsample=subsample,
                    square_validation=square_validation,
                    no_mp=no_mp)
                if valid_images is None:
                    if square_validation:
                        s = self.hps.validation_square
                        valid_images = [
                            Image(None, im.data[:, :s, :s], im.mask[:, :s, :s])
                            for im in train_images]
                    else:
                        valid_images = [self.load_image(im_id)
                                        for im_id in sorted(valid_ids)]
                if valid_images:
                    self.validate_on_images(
                        valid_images,
                        subsample=1 if square_validation else subsample)
            self.save_snapshot(n_epoch)
        self.tb_logger = None
        self.logdir = None

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
        scaled = ((im_data - mean) / std).astype(np.float16)
        return scaled.transpose([2, 0, 1])  # torch order

    def load_image(self, im_id: str) -> Image:
        logger.info('Loading {}'.format(im_id))
        im_cache = Path('im_cache')
        im_cache.mkdir(exist_ok=True)
        im_data_path = im_cache.joinpath('{}.data'.format(im_id))
        mask_path = im_cache.joinpath('{}.mask'.format(im_id))
        if im_data_path.exists():
            im_data = np.load(str(im_data_path))
        else:
            im_data = self.preprocess_image(utils.load_image(im_id))
            with im_data_path.open('wb') as f:
                np.save(f, im_data)
        if mask_path.exists():
            mask = np.load(str(mask_path))
        else:
            im_size = im_data.shape[1:]
            poly_by_type = utils.load_polygons(im_id, im_size)
            mask = np.array(
                [utils.mask_for_polygons(im_size, poly_by_type[cls + 1])
                 for cls in range(self.hps.total_classes)],
                dtype=np.uint8)
            with mask_path.open('wb') as f:
                np.save(f, mask)
        if self.hps.n_channels != im_data.shape[0]:
            im_data = im_data[:self.hps.n_channels]
        return Image(im_id, im_data, mask[self.hps.classes])

    def train_on_images(self, train_images: List[Image],
                        subsample: int=1,
                        square_validation: bool=False,
                        no_mp: bool=False):
        self.net.train()
        b = self.hps.patch_border
        s = self.hps.patch_inner
        # Extra margin for rotation
        m = int(np.ceil((np.sqrt(2) - 1) * (b + s / 2)))
        mb = m + b  # full margin
        avg_area = np.mean(
            [im.size[0] * im.size[1] for im in train_images])
        n_batches = int(avg_area / (s + b) / self.hps.batch_size / subsample)

        def gen_batch(_):
            inputs, outputs = [], []
            for _ in range(self.hps.batch_size):
                im, (x, y) = self.sample_im_xy(train_images, square_validation)
                if random.random() < self.hps.oversample:
                    for _ in range(1000):
                        if im.mask[x: x + s, y: y + s].sum():
                            break
                        im, (x, y) = self.sample_im_xy(
                            train_images, square_validation)
                patch = im.data[:, x - mb: x + s + mb, y - mb: y + s + mb]
                mask = im.mask[:, x - m: x + s + m, y - m: y + s + m]
                if self.hps.augment_flips:
                    if random.random() < 0.5:
                        patch = np.flip(patch, 1)
                        mask = np.flip(mask, 1)
                    if random.random() < 0.5:
                        patch = np.flip(patch, 2)
                        mask = np.flip(mask, 2)
                if self.hps.augment_rotations:
                    angle = random.random() * 180
                    patch = utils.rotated(patch, angle)
                    mask = utils.rotated(mask, angle)
                inputs.append(patch[:, m: -m, m: -m].astype(np.float32))
                outputs.append(mask[:, m: -m, m: -m].astype(np.float32))

            return (torch.from_numpy(np.array(inputs)),
                    torch.from_numpy(np.array(outputs)))

        self._train_on_feeds(gen_batch, n_batches, no_mp=no_mp)

    def sample_im_xy(self, train_images, square_validation=False):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        # Extra margin for rotation
        m = int(np.ceil((np.sqrt(2) - 1) * (b + s / 2)))
        mb = m + b  # full margin
        im = random.choice(train_images)
        w, h = im.size
        min_xy = mb
        if square_validation:
            min_xy += self.hps.validation_square
        return im, (random.randint(min_xy, w - (mb + s)),
                    random.randint(min_xy, h - (mb + s)))

    def _train_on_feeds(self, gen_batch, n_batches: int, no_mp: bool):
        losses = [[] for _ in range(self.hps.n_classes)]
        jaccard_stats = self._jaccard_stats()

        def log():
            logger.info(
                'Train loss: {loss:.3f}, Jaccard: {jaccard}, '
                'speed: {speed:,} patches/s'.format(
                    loss=np.array(losses)[:, -log_step:].mean(),
                    speed=int(len(losses[0]) * self.hps.batch_size / (t1 - t00)),
                    jaccard=self._format_jaccard(jaccard_stats),
                ))

        t0 = t00 = time.time()
        log_step = 100
        im_log_step = n_batches // log_step * log_step
        map_ = map if no_mp else partial(utils.imap_fixed_output_buffer, processes=4)
        for i, (x, y) in enumerate(map_(gen_batch, range(n_batches))):
            if losses[0] and i % log_step == 0:
                for cls, ls in zip(self.hps.classes, losses):
                    self._log_value(
                        'loss/cls-{}'.format(cls), np.mean(ls[-log_step:]))
                pred_y = self.net(self._var(x)).data.cpu()
                self._update_jaccard(jaccard_stats, y.numpy(), pred_y.numpy())
                self._log_jaccard(jaccard_stats)
                if i == im_log_step:
                    self._log_im(x.numpy(), y.numpy(), pred_y.numpy())
            step_losses = self.train_step(x, y)
            for ls, l in zip(losses, step_losses):
                ls.append(l.data[0])
            t1 = time.time()
            dt = t1 - t0
            if dt > 10:
                log()
                jaccard_stats = self._jaccard_stats()
                t0 = t1
        if losses:
            log()

    def _jaccard_stats(self):
        return {cls: {threshold: [[] for _ in range(3)]
                      for threshold in self.hps.thresholds}
                for cls in self.hps.classes}

    def _update_jaccard(self, stats, mask, pred):
        assert mask.shape == pred.shape
        assert len(mask.shape) in {3, 4}
        for cls, tp_fp_fn in stats.items():
            cls_idx = self.hps.classes.index(cls)
            if len(mask.shape) == 3:
                assert mask.shape[0] == self.hps.n_classes
                p, y = pred[cls_idx], mask[cls_idx]
            else:
                assert mask.shape[1] == self.hps.n_classes
                p, y = pred[:, cls_idx], mask[:, cls_idx]
            for threshold, (tp, fp, fn) in tp_fp_fn.items():
                _tp, _fp, _fn = utils.mask_tp_fp_fn(p, y, threshold)
                tp.append(_tp)
                fp.append(_fp)
                fn.append(_fn)

    def _log_jaccard(self, stats, prefix=''):
        for cls, tp_fp_fn in stats.items():
            for threshold, (tp, fp, fn) in tp_fp_fn.items():
                self._log_value(
                    '{}jaccard-{}/cls-{}'.format(prefix, threshold, cls),
                    self._jaccard(tp, fp, fn))

    def _jaccard(self, tp, fp, fn):
        if sum(tp) == 0:
            return 0
        return sum(tp) / (sum(tp) + sum(fn) + sum(fp))

    def _format_jaccard(self, stats):
        jaccard_by_threshold = {}
        for cls, tp_fp_fn in stats.items():
            for threshold, (tp, fp, fn) in tp_fp_fn.items():
                jaccard_by_threshold.setdefault(threshold, []).append(
                    self._jaccard(tp, fp, fn))
        return ', '.join(
            'at {:.2f}: {:.3f}'.format(threshold, np.mean(cls_jaccards))
            for threshold, cls_jaccards in sorted(jaccard_by_threshold.items()))

    def _log_im(self, xs: np.ndarray, ys: np.ndarray, pred_ys: np.ndarray):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        border = np.zeros([b * 2 + s, b * 2 + s, 3], dtype=np.float32)
        border[b, b:-b, :] = border[-b, b:-b, :] = 1
        border[b:-b, b, :] = border[b:-b, -b, :] = 1
        border[-b, -b, :] = 1
        for i, (x, y, p) in enumerate(zip(xs, ys, pred_ys)):
            fname = lambda s: str(self.logdir / ('{:0>3}_{}.png'.format(i, s)))
            rgb = utils.scale_percentile(x[:3].transpose(1, 2, 0))
            cv2.imwrite(fname('x'), np.maximum(border, rgb) * 255)
            for cls, c_y, c_p in zip(self.hps.classes, y, p):
                cv2.imwrite(fname('y_{}'.format(cls)), c_y * 255)
                cv2.imwrite(fname('z_{}'.format(cls)), c_p * 255)

    def _log_value(self, name, value):
        self.tb_logger.log_value(name, value, step=self.net.global_step[0])

    def validate_on_images(self, valid_images: List[Image],
                           subsample: int=1):
        self.net.eval()
        b = self.hps.patch_border
        s = self.hps.patch_inner
        losses = [[] for _ in range(self.hps.n_classes)]
        jaccard_stats = self._jaccard_stats()
        for im in valid_images:
            w, h = im.size
            xs = range(b, w - (b + s), s)
            ys = range(b, h - (b + s), s)
            all_xy = [(x, y) for x in xs for y in ys]
            if subsample != 1:
                random.shuffle(all_xy)
                all_xy = all_xy[:len(all_xy) // subsample]
            for xy_batch in utils.chunks(all_xy, self.hps.batch_size):
                inputs = np.array(
                    [im.data[:, x - b: x + s + b, y - b: y + s + b]
                     for x, y in xy_batch]).astype(np.float32)
                outputs = np.array(
                    [im.mask[:, x: x + s, y: y + s] for x, y in xy_batch])
                outputs = outputs.astype(np.float32)
                y_pred = self.net(self._var(torch.from_numpy(inputs)))
                step_losses = self.losses(
                    self._var(torch.from_numpy(outputs)), y_pred)
                for ls, l in zip(losses, step_losses):
                    ls.append(l.data[0])
                y_pred_numpy = y_pred.data.cpu().numpy()
                self._update_jaccard(jaccard_stats, outputs, y_pred_numpy)
        losses = np.array(losses)
        logger.info('Valid loss: {:.3f}, Jaccard: {}'.format(
            losses.mean(), self._format_jaccard(jaccard_stats)))
        for cls, cls_losses in zip(self.hps.classes, losses):
            self._log_value('valid-loss/cls-{}'.format(cls), cls_losses.mean())
        self._log_jaccard(jaccard_stats, prefix='valid-')

    def restore_last_snapshot(self, logdir: Path) -> int:
        for n_epoch in reversed(range(self.hps.n_epochs)):
            model_path = self._model_path(logdir, n_epoch)
            if model_path.exists():
                logger.info('Loading snapshot {}'.format(model_path))
                self.restore_snapshot(model_path)
                return n_epoch + 1
        return 0

    def restore_snapshot(self, model_path: Path):
        state = torch.load(str(model_path))
        self.net.load_state_dict(state)

    def save_snapshot(self, n_epoch: int):
        model_path = self._model_path(self.logdir, n_epoch)
        logger.info('Saving snapshot {}'.format(model_path))
        torch.save(self.net.state_dict(), str(model_path))

    def _model_path(self, logdir: Path, n_epoch: int) -> Path:
        return logdir.joinpath('model-{}'.format(n_epoch))

    def predict_image_mask(self, im: Image) -> np.ndarray:
        self.net.eval()
        # FIXME - some copy-paste
        w, h = im.size
        b = self.hps.patch_border
        s = self.hps.patch_inner
        xs = range(b, w - (b + s), s)
        ys = range(b, h - (b + s), s)
        all_xy = [(x, y) for x in xs for y in ys]
        pred_mask = np.zeros([self.hps.n_classes, w, h], dtype=np.float32)
        for xy_batch in tqdm.tqdm(
                list(utils.chunks(all_xy, self.hps.batch_size))):
            inputs = np.array(
                [im.data[:, x - b: x + s + b, y - b: y + s + b]
                 for x, y in xy_batch]).astype(np.float32)
            y_pred = self.net(self._var(torch.from_numpy(inputs)))
            for (x, y), mask in zip(xy_batch, y_pred.data.cpu().numpy()):
                pred_mask[:, x: x + s, y: y + s] = mask
        return pred_mask


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', type=str, help='Path to log directory')
    arg('--hps', type=str, help='Change hyperparameters in k1=v1,k2=v2 format')
    arg('--all', action='store_true',
        help='Train on all images without validation')
    arg('--validation', choices=['random', 'stratified', 'square'],
        default='stratified', help='validation strategy')
    arg('--only', type=str,
        help='Train on this image ids only (comma-separated) without validation')
    arg('--clean', action='store_true', help='Clean logdir')
    arg('--no-mp', action='store_true', help='Disable multiprocessing')
    args = parser.parse_args()
    hps = HyperParams()
    hps.update(args.hps)
    pprint(attr.asdict(hps))
    logdir = Path(args.logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    if args.clean:
        for p in logdir.iterdir():
            p.unlink()
    logdir.joinpath('hps.json').write_text(
        json.dumps(attr.asdict(hps), indent=True, sort_keys=True))

    model = Model(hps=hps)
    all_im_ids = list(utils.get_wkt_data())
    valid_ids = []
    bad_pairs = [('6110', '6140'),
                 ('6020', '6130'),
                 ('6030', '6150')]
    if args.only:
        train_ids = args.only.split(',')
    elif args.all:
        train_ids = all_im_ids
    elif args.validation == 'stratified':
        mask_stats = utils.load_mask_stats()
        im_area = [(im_id, np.mean([mask_stats[im_id][str(cls)]['area']
                                    for cls in hps.classes]))
                   for im_id in all_im_ids]
        im_area.sort(key=lambda x: (x[1], x[0]), reverse=True)
        train_ids, valid_ids = [], []
        for idx, (im_id, _) in enumerate(im_area):
            (valid_ids if (idx % 4 == 1) else train_ids).append(im_id)
        area_by_id = dict(im_area)
        logger.info('Train area mean: {}'.format(
            np.mean([area_by_id[im_id] for im_id in valid_ids])))
        logger.info('Valid area mean: {}'.format(
            np.mean([area_by_id[im_id] for im_id in train_ids])))
        # TODO - recover
        for a, b in bad_pairs:
            if a in valid_ids and b in valid_ids:
                assert False
            if a in train_ids and b in train_ids:
                assert False
    elif args.validation == 'square':
        train_ids = valid_ids = all_im_ids
    elif args.validation == 'random':
        # Fix for images of the same place in different seasons
        labels = []
        for im_id in all_im_ids:
            for pair in bad_pairs:
                im_id = im_id.replace(*pair)
            labels.append(im_id)
        train_ids, valid_ids = [[all_im_ids[idx] for idx in g] for g in next(
            GroupShuffleSplit(random_state=1).split(all_im_ids, groups=labels))]
        logger.info('Train: {}'.format(' '.join(sorted(train_ids))))
        logger.info('Valid: {}'.format(' '.join(sorted(valid_ids))))
    else:
        raise ValueError('Unexpected validation kind: {}'.format(args.validation))
    model.train(logdir=logdir,
                train_ids=train_ids,
                valid_ids=valid_ids,
                validation=args.validation,
                no_mp=args.no_mp,
                )


if __name__ == '__main__':
    main()