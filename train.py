#!/usr/bin/env python3
import argparse
import json
from functools import partial
from pathlib import Path
from pprint import pprint
import random
import time
from typing import List, Iterable

import attr
import cv2
import numpy as np
from sklearn.model_selection import ShuffleSplit
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
    _dist_mask = attr.ib(default=None)

    @property
    def size(self):
        assert self.data.shape[0] <= 20
        return self.data.shape[1:]

    @property
    def dist_mask(self):
        if self._dist_mask is None:
            assert self.mask.shape[0] <= 10
            self._dist_mask = (
                np.stack([utils.dist_mask(m, max_dist=5) for m in self.mask])
                .astype(np.float16))
        return self._dist_mask


class Model:
    def __init__(self, hps: HyperParams):
        self.hps = hps
        self.net = getattr(models, hps.net)(hps)
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
        self.optimizer = None  # type: optim.Optimizer
        self.tb_logger = None  # type: tensorboard_logger.Logger
        self.logdir = None  # type: Path
        self.on_gpu = torch.cuda.is_available()
        if self.on_gpu:
            self.net.cuda()

    def _init_optimizer(self, lr):
        return optim.Adam(self.net.parameters(),
                          lr=lr, weight_decay=self.hps.weight_decay)

    def _var(self, x: torch.FloatTensor) -> Variable:
        return Variable(x.cuda() if self.on_gpu else x)

    def train_step(self, x, y, dist_y):
        self.optimizer.zero_grad()
        y_pred = self.net(self._var(x))
        batch_size = x.size()[0]
        losses = self.losses(y, dist_y, y_pred)
        cls_losses = [float(l.data[0]) for l in losses]
        loss = losses[0]
        for l in losses[1:]:
            loss += l
        (loss * batch_size).backward()
        self.optimizer.step()
        self.net.global_step += 1
        return cls_losses

    def losses(self,
               ys: torch.FloatTensor,
               ys_dist: torch.FloatTensor,
               y_preds: Variable):
        losses = []
        ys = self._var(ys)
        if self.hps.needs_dist:
            ys_dist = self._var(ys_dist)
        for cls_idx, _ in enumerate(self.hps.classes):
            y, y_pred = ys[:, cls_idx], y_preds[:, cls_idx]
            y_dist = ys_dist[:, cls_idx] if self.hps.needs_dist else None
            loss = self._cls_loss(y, y_dist, y_pred)
            losses.append(loss)
        return losses

    def _cls_loss(self, y, y_dist, y_pred):
        hps = self.hps
        loss = 0.
        if hps.log_loss:
            loss += self.bce_loss(y_pred, y) * hps.log_loss
        if hps.dice_loss:
            intersection = (y_pred * y).sum()
            uwi = y_pred.sum() + y.sum()  # without intersection union
            if uwi[0] != 0:
                loss += (1 - intersection / uwi) * hps.dice_loss
        if hps.jaccard_loss:
            intersection = (y_pred * y).sum()
            union = y_pred.sum() + y.sum() - intersection
            if union[0] != 0:
                loss += (1 - intersection / union) * hps.jaccard_loss
        if hps.dist_loss:
            loss += self.mse_loss(y_pred, y_dist) * hps.dist_loss
        if hps.dist_dice_loss:
            intersection = (y_pred * y_dist).sum()
            uwi = y_pred.sum() + y_dist.sum()  # without intersection union
            if uwi[0] != 0:
                loss += (1 - intersection / uwi) * hps.dist_dice_loss
        if hps.dist_jaccard_loss:
            intersection = (y_pred * y_dist).sum()
            union = y_pred.sum() + y_dist.sum() - intersection
            if union[0] != 0:
                loss += (1 - intersection / union) * hps.dist_jaccard_loss
        loss /= (hps.log_loss + hps.dist_loss + hps.dist_jaccard_loss +
                 hps.dist_dice_loss + hps.dice_loss + hps.jaccard_loss)
        return loss

    def train(self, logdir: Path, train_ids: List[str], valid_ids: List[str],
              validation: str, no_mp: bool=False, valid_only: bool=False,
              model_path: Path=None):
        self.tb_logger = tensorboard_logger.Logger(str(logdir))
        self.logdir = logdir
        train_images = [self.load_image(im_id) for im_id in sorted(train_ids)]
        valid_images = None
        if model_path:
            self.restore_snapshot(model_path)
            start_epoch = int(model_path.name.rsplit('-', 1)[1]) + 1
        else:
            start_epoch = self.restore_last_snapshot(logdir)
        square_validation = validation == 'square'
        lr = self.hps.lr
        self.optimizer = self._init_optimizer(lr)
        for n_epoch in range(start_epoch, self.hps.n_epochs):
            if self.hps.lr_decay:
                if n_epoch % 2 == 0 or n_epoch == start_epoch:
                    lr = self.hps.lr * self.hps.lr_decay ** n_epoch
                    self.optimizer = self._init_optimizer(lr)
            else:
                lim_1, lim_2 = 25, 50
                if n_epoch == lim_1 or (
                        n_epoch == start_epoch and n_epoch > lim_1):
                    lr = self.hps.lr / 5
                    self.optimizer = self._init_optimizer(lr)
                if n_epoch == lim_2 or (
                        n_epoch == start_epoch and n_epoch > lim_2):
                    lr = self.hps.lr / 25
                    self.optimizer = self._init_optimizer(lr)
            logger.info('Starting epoch {}, step {:,}, lr {:.8f}'.format(
                n_epoch + 1, self.net.global_step[0], lr))
            subsample = 1 if valid_only else 2  # make validation more often
            for _ in range(subsample):
                if not valid_only:
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
                    self.validate_on_images(valid_images, subsample=1)
            if valid_only:
                break
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
        pre_buffer = self.hps.pre_buffer
        if mask_path.exists() and not pre_buffer:
            mask = np.load(str(mask_path))
        else:
            im_size = im_data.shape[1:]
            poly_by_type = utils.load_polygons(im_id, im_size)
            if pre_buffer:
                structures = 2
                poly_by_type[structures] = utils.to_multipolygon(
                    poly_by_type[structures].buffer(pre_buffer))
            mask = np.array(
                [utils.mask_for_polygons(im_size, poly_by_type[cls + 1])
                 for cls in range(self.hps.total_classes)],
                dtype=np.uint8)
            if not pre_buffer:
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
        mean_area = np.mean(
            [im.size[0] * im.size[1] for im in train_images])
        n_batches = int(
            mean_area / (s + b) / self.hps.batch_size / subsample / 2)

        def gen_batch(_):
            inputs, outputs, dist_outputs = [], [], []
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
                if self.hps.needs_dist:
                    dist_mask = im.dist_mask[:, x - m: x + s + m, y - m: y + s + m]

                if self.hps.augment_flips:
                    if random.random() < 0.5:
                        patch = np.flip(patch, 1)
                        mask = np.flip(mask, 1)
                        if self.hps.needs_dist:
                            dist_mask = np.flip(dist_mask, 1)
                    if random.random() < 0.5:
                        patch = np.flip(patch, 2)
                        mask = np.flip(mask, 2)
                        if self.hps.needs_dist:
                            dist_mask = np.flip(dist_mask, 2)

                if self.hps.augment_rotations:
                    assert self.hps.augment_rotations != 1  # old format
                    angle = (2 * random.random() - 1.) * self.hps.augment_rotations
                    patch = utils.rotated(patch, angle)
                    mask = utils.rotated(mask, angle)
                    if self.hps.needs_dist:
                        dist_mask = utils.rotated(dist_mask, angle)

                if self.hps.augment_channels:
                    ch_shift = np.random.normal(
                        1, self.hps.augment_channels, patch.shape[0])
                    patch = patch * ch_shift[:, None, None]

                inputs.append(patch[:, m: -m, m: -m].astype(np.float32))
                outputs.append(mask[:, m: -m, m: -m].astype(np.float32))
                if self.hps.needs_dist:
                    dist_outputs.append(
                        dist_mask[:, m: -m, m: -m].astype(np.float32))

            return (torch.from_numpy(np.array(inputs)),
                    torch.from_numpy(np.array(outputs)),
                    torch.from_numpy(np.array(dist_outputs)))

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
        log_step = 50
        im_log_step = n_batches // log_step * log_step
        map_ = (map if no_mp else
                partial(utils.imap_fixed_output_buffer, threads=4))
        for i, (x, y, dist_y) in enumerate(map_(gen_batch, range(n_batches))):
            if losses[0] and i % log_step == 0:
                for cls, ls in zip(self.hps.classes, losses):
                    self._log_value(
                        'loss/cls-{}'.format(cls), np.mean(ls[-log_step:]))
                if self.hps.has_all_classes:
                    self._log_value(
                        'loss/cls-mean', np.mean([
                            l for ls in losses for l in ls[-log_step:]]))
                pred_y = self.net(self._var(x)).data.cpu()
                self._update_jaccard(jaccard_stats, y.numpy(), pred_y.numpy())
                self._log_jaccard(jaccard_stats)
                if i == im_log_step:
                    self._log_im(
                        x.numpy(), y.numpy(), dist_y.numpy(), pred_y.numpy())
            step_losses = self.train_step(x, y, dist_y)
            for ls, l in zip(losses, step_losses):
                ls.append(l)
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
        jaccard_by_threshold = {}
        for cls, tp_fp_fn in stats.items():
            for threshold, (tp, fp, fn) in tp_fp_fn.items():
                jaccard = self._jaccard(tp, fp, fn)
                self._log_value(
                    '{}jaccard-{}/cls-{}'.format(prefix, threshold, cls),
                    jaccard)
                jaccard_by_threshold.setdefault(threshold, []).append(jaccard)
        if self.hps.has_all_classes:
            for threshold, jaccards in jaccard_by_threshold.items():
                self._log_value(
                    '{}jaccard-{}/cls-mean'.format(prefix, threshold),
                    np.mean(jaccards))

    @staticmethod
    def _jaccard(tp, fp, fn):
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

    def _log_im(self, xs: np.ndarray,
                ys: np.ndarray, dist_ys: np.ndarray,
                pred_ys: np.ndarray):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        border = np.zeros([b * 2 + s, b * 2 + s, 3], dtype=np.float32)
        border[b, b:-b, :] = border[-b, b:-b, :] = 1
        border[b:-b, b, :] = border[b:-b, -b, :] = 1
        border[-b, -b, :] = 1
        for i, (x, y, p) in enumerate(zip(xs, ys, pred_ys)):
            fname = lambda s: str(self.logdir / ('{:0>3}_{}.png'.format(i, s)))
            x = utils.scale_percentile(x.transpose(1, 2, 0))
            channels = [x[:, :, :3]]  # RGB
            if x.shape[-1] == 12:
                channels.extend([
                    x[:, :, 4:7],    # M
                    x[:, :, 3:4],    # P (will be shown below RGB)
                    # 7 and 8 from M are skipped
                    x[:, :, 9:12],   # M
                ])
            elif x.shape[-1] == 20:
                channels.extend([
                    x[:, :, 4:7],    # M
                    x[:, :, 6:9],    # M (overlap)
                    x[:, :, 9:12],   # M
                    x[:, :, 3:4],    # P (will be shown below RGB)
                    x[:, :, 12:15],  # A (overlap)
                    x[:, :, 14:17],  # A
                    x[:, :, 17:],    # A
                ])
            channels = [np.maximum(border, ch) for ch in channels]
            if len(channels) >= 4:
                n = len(channels) // 2
                img = np.concatenate(
                    [np.concatenate(channels[:n], 1),
                     np.concatenate(channels[n:], 1)], 0)
            else:
                img = np.concatenate(channels, axis=1)
            cv2.imwrite(fname('-x'), img * 255)
            for j, (cls, c_y, c_p) in enumerate(zip(self.hps.classes, y, p)):
                cv2.imwrite(fname('{}-y'.format(cls)), c_y * 255)
                cv2.imwrite(fname('{}-z'.format(cls)), c_p * 255)
                if dist_ys.shape[0]:
                    cv2.imwrite(fname('{}-d'.format(cls)), dist_ys[i, j] * 255)

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
            for xy_batch in utils.chunks(all_xy, self.hps.batch_size // 2):
                inputs = np.array(
                    [im.data[:, x - b: x + s + b, y - b: y + s + b]
                     for x, y in xy_batch]).astype(np.float32)
                outputs = np.array(
                    [im.mask[:, x: x + s, y: y + s] for x, y in xy_batch])
                outputs = outputs.astype(np.float32)
                if self.hps.needs_dist:
                    dist_outputs = np.array([im.dist_mask[:, x: x + s, y: y + s]
                                             for x, y in xy_batch])
                    dist_outputs = dist_outputs.astype(np.float32)
                else:
                    dist_outputs = np.array([])
                y_pred = self.net(self._var(torch.from_numpy(inputs)))
                step_losses = self.losses(
                    torch.from_numpy(outputs),
                    torch.from_numpy(dist_outputs),
                    y_pred)
                for ls, l in zip(losses, step_losses):
                    ls.append(l.data[0])
                y_pred_numpy = y_pred.data.cpu().numpy()
                self._update_jaccard(jaccard_stats, outputs, y_pred_numpy)
        losses = np.array(losses)
        logger.info('Valid loss: {:.3f}, Jaccard: {}'.format(
            losses.mean(), self._format_jaccard(jaccard_stats)))
        for cls, cls_losses in zip(self.hps.classes, losses):
            self._log_value('valid-loss/cls-{}'.format(cls), cls_losses.mean())
        if self.hps.has_all_classes:
            self._log_value('valid-loss/cls-mean', losses.mean())
        self._log_jaccard(jaccard_stats, prefix='valid-')

    def restore_last_snapshot(self, logdir: Path) -> int:
        average = 1  # TODO - pass
        for n_epoch in reversed(range(self.hps.n_epochs)):
            model_path = self._model_path(logdir, n_epoch)
            if model_path.exists():
                if average and average > 1:
                    self.restore_average_snapshot(
                        logdir, range(n_epoch - average + 1, n_epoch + 1))
                else:
                    self.restore_snapshot(model_path)
                return n_epoch + 1
        return 0

    def restore_snapshot(self, model_path: Path):
        logger.info('Loading snapshot {}'.format(model_path))
        state = torch.load(str(model_path))
        self.net.load_state_dict(state)

    def restore_average_snapshot(self, logdir: Path, epochs: Iterable[int]):
        epochs = list(epochs)
        logger.info('Loading averaged snapshot {} for epochs {}'
                    .format(logdir, epochs))
        states = [torch.load(str(self._model_path(logdir, n)))
                  for n in epochs]
        average_state = {key: sum(s[key] for s in states) / len(states)
                         for key in states[0].keys()}
        self.net.load_state_dict(average_state)

    def save_snapshot(self, n_epoch: int):
        model_path = self._model_path(self.logdir, n_epoch)
        logger.info('Saving snapshot {}'.format(model_path))
        torch.save(self.net.state_dict(), str(model_path))

    def _model_path(self, logdir: Path, n_epoch: int) -> Path:
        return logdir.joinpath('model-{}'.format(n_epoch))

    def predict_image_mask(self, im_data: np.ndarray,
                           rotate: bool=False,
                           no_edges: bool=False,
                           average_shifts: bool=True
                           ) -> np.ndarray:
        self.net.eval()
        c, w, h = im_data.shape
        b = self.hps.patch_border
        s = self.hps.patch_inner
        padded = np.zeros([c, w + 2 * b, h + 2 * b], dtype=im_data.dtype)
        padded[:, b:-b, b:-b] = im_data
        # mirror on the edges
        padded[:, :b, b:-b] = np.flip(im_data[:, :b, :], 1)
        padded[:, -b:, b:-b] = np.flip(im_data[:, -b:, :], 1)
        padded[:, :, :b] = np.flip(padded[:, :, b: 2 * b], 2)
        padded[:, :, -b:] = np.flip(padded[:, :, -2 * b: -b], 2)
        step = s // 3 if average_shifts else s
        margin = b if no_edges else 0
        xs = list(range(margin, w - s - margin, step)) + [w - s - margin]
        ys = list(range(margin, h - s - margin, step)) + [h - s - margin]
        all_xy = [(x, y) for x in xs for y in ys]
        out_shape = [self.hps.n_classes, w, h]
        pred_mask = np.zeros(out_shape, dtype=np.float32)
        pred_per_pixel = np.zeros(out_shape, dtype=np.int16)
        n_rot = 4 if rotate else 1

        def gen_batch(xy_batch_):
            inputs_ = []
            for x, y in xy_batch_:
                # shifted by -b to account for padding
                patch = padded[:, x: x + s + 2 * b, y: y + s + 2 * b]
                inputs_.append(patch)
                for i in range(1, n_rot):
                    inputs_.append(utils.rotated(patch, i * 90))
            return xy_batch_, np.array(inputs_, dtype=np.float32)

        for xy_batch, inputs in utils.imap_fixed_output_buffer(
                gen_batch, tqdm.tqdm(list(
                    utils.chunks(all_xy, self.hps.batch_size // (4 * n_rot)))),
                threads=2):
            y_pred = self.net(self._var(torch.from_numpy(inputs)))
            for idx, mask in enumerate(y_pred.data.cpu().numpy()):
                x, y = xy_batch[idx // n_rot]
                i = idx % n_rot
                if i:
                    mask = utils.rotated(mask, -i * 90)
                # mask = (mask >= 0.5) + 0.001
                pred_mask[:, x: x + s, y: y + s] += mask / n_rot
                pred_per_pixel[:, x: x + s, y: y + s] += 1
        if not no_edges:
            assert pred_per_pixel.min() >= 1
        pred_mask /= np.maximum(pred_per_pixel, 1)
        return pred_mask


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', help='Path to log directory')
    arg('--hps', help='Change hyperparameters in k1=v1,k2=v2 format')
    arg('--all', action='store_true',
        help='Train on all images without validation')
    arg('--validation', choices=['random', 'stratified', 'square', 'custom'],
        default='custom', help='validation strategy')
    arg('--valid-only', action='store_true')
    arg('--only',
        help='Train on this image ids only (comma-separated) without validation')
    arg('--clean', action='store_true', help='Clean logdir')
    arg('--no-mp', action='store_true', help='Disable multiprocessing')
    arg('--model-path', type=Path)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    logdir.mkdir(exist_ok=True, parents=True)
    if args.clean:
        for p in logdir.iterdir():
            p.unlink()

    if args.hps == 'load':
        hps = HyperParams.from_dir(logdir)
    else:
        hps = HyperParams()
        hps.update(args.hps)
        logdir.joinpath('hps.json').write_text(
            json.dumps(attr.asdict(hps), indent=True, sort_keys=True))
    pprint(attr.asdict(hps))

    model = Model(hps=hps)
    all_im_ids = list(utils.get_wkt_data())
    mask_stats = json.loads(Path('cls-stats.json').read_text())
    im_area = [(im_id, np.mean([mask_stats[im_id][str(cls)]['area']
                                for cls in hps.classes]))
               for im_id in all_im_ids]
    area_by_id = dict(im_area)
    valid_ids = []

    if args.only:
        train_ids = args.only.split(',')
    elif args.all:
        train_ids = all_im_ids
    elif args.validation == 'stratified':
        train_ids, valid_ids = [], []
        for idx, (im_id, _) in enumerate(
                sorted(im_area, key=lambda x: (x[1], x[0]), reverse=True)):
            (valid_ids if (idx % 4 == 1) else train_ids).append(im_id)
    elif args.validation == 'square':
        train_ids = valid_ids = all_im_ids
    elif args.validation == 'random':
        forced_train_ids = {'6070_2_3', '6120_2_2', '6110_4_0'}
        other_ids = list(set(all_im_ids) - forced_train_ids)
        train_ids, valid_ids = [[other_ids[idx] for idx in g] for g in next(
            ShuffleSplit(random_state=1, n_splits=4).split(other_ids))]
        train_ids.extend(forced_train_ids)
    elif args.validation == 'custom':
        valid_ids = ['6140_3_1', '6110_1_2', '6160_2_1', '6170_0_4', '6100_2_2']
        train_ids = [im_id for im_id in all_im_ids if im_id not in valid_ids]
    else:
        raise ValueError('Unexpected validation kind: {}'.format(args.validation))

    if args.valid_only:
        train_ids = []

    train_area_by_class, valid_area_by_class = [
        {cls: np.mean(
            [mask_stats[im_id][str(cls)]['area'] for im_id in im_ids])
         for cls in hps.classes}
        for im_ids in [train_ids, valid_ids]]

    logger.info('Train: {}'.format(' '.join(sorted(train_ids))))
    logger.info('Valid: {}'.format(' '.join(sorted(valid_ids))))
    logger.info('Train area mean: {:.6f}'.format(
        np.mean([area_by_id[im_id] for im_id in valid_ids])))
    logger.info('Train area by class: {}'.format(
        ' '.join('{}: {:.6f}'.format(cls, train_area_by_class[cls])
                 for cls in hps.classes)))
    logger.info('Valid area mean: {:.6f}'.format(
        np.mean([area_by_id[im_id] for im_id in train_ids])))
    logger.info('Valid area by class: {}'.format(
        ' '.join('cls-{}: {:.6f}'.format(cls, valid_area_by_class[cls])
                 for cls in hps.classes)))

    model.train(logdir=logdir,
                train_ids=train_ids,
                valid_ids=valid_ids,
                validation=args.validation,
                no_mp=args.no_mp,
                valid_only=args.valid_only,
                model_path=args.model_path
                )


if __name__ == '__main__':
    main()