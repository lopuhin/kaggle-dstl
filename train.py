#!/usr/bin/env python3
import argparse
import json
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
import torch.nn.functional as F
import tqdm

import utils


logger = utils.get_logger(__name__)


@attr.s(slots=True)
class HyperParams:
    cls = attr.ib()
    net = attr.ib(default='DefaultNet')
    n_channels = attr.ib(default=20)
    total_classes = 10
    thresholds = attr.ib(default=[0.2, 0.3, 0.4, 0.5, 0.6])

    patch_inner = attr.ib(default=64)
    patch_border = attr.ib(default=16)

    dropout_keep_prob = attr.ib(default=0.0)  # TODO
    jaccard_loss = attr.ib(default=0)

    n_epochs = attr.ib(default=30)
    oversample = attr.ib(default=0.0)
    lr = attr.ib(default=0.0001)
    batch_size = attr.ib(default=128)

    def update(self, hps_string: str):
        if hps_string:
            for pair in hps_string.split(','):
                k, v = pair.split('=')
                if '.' in v:
                    v = float(v)
                elif k != 'net':
                    v = int(v)
                setattr(self, k, v)


class BaseNet(nn.Module):
    def __init__(self, hps: HyperParams):
        super().__init__()
        self.hps = hps
        self.register_buffer('global_step', torch.IntTensor(1).zero_())


class MiniNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 4, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, 0, b:-b, b:-b])


class OldNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 64, 5, padding=2)
        self.conv2 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv3 = nn.Conv2d(64, 64, 5, padding=2)
        self.conv4 = nn.Conv2d(64, 1, 7, padding=3)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, 0, b:-b, b:-b])


class SmallNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 16, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, 0, b:-b, b:-b])


# UNet:
# http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/u-net-architecture.png


class SmallUNet(BaseNet):
    def __init__(self, hps):
        super().__init__(hps)
        self.conv1 = nn.Conv2d(hps.n_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x1 = self.pool(x)
        x1 = F.relu(self.conv3(x1))
        x1 = F.relu(self.conv4(x1))
        x1 = F.relu(self.conv5(x1))
        # repeat is missing: https://github.com/pytorch/pytorch/issues/440
        x1 = x1.repeat(1, 1, 2, 2)
        x = torch.cat([x, x1], 1)
        x = F.relu(self.conv6(x))
        x = self.conv7(x)
        b = self.hps.patch_border
        return F.sigmoid(x[:, 0, b:-b, b:-b])


DefaultNet = OldNet


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
        self.net = globals()[hps.net](hps)
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
        loss = self.loss(y, y_pred)
        (loss * batch_size).backward()
        self.optimizer.step()
        self.net.global_step += 1
        return loss.data[0]

    def loss(self, y, y_pred):
        loss = self.bce_loss(y_pred, y)
        if self.hps.jaccard_loss:
            intersection = (y_pred * y).sum()
            union = y_pred.sum() + y.sum()
            if union[0] != 0:
                loss += self.hps.jaccard_loss * (1 - intersection / union)
        return loss

    def train(self, logdir: Path, train_ids: List[str], valid_ids: List[str]):
        self.tb_logger = tensorboard_logger.Logger(str(logdir))
        self.logdir = logdir
        train_images = [self.load_image(im_id) for im_id in sorted(train_ids)]
        valid_images = None
        n_epoch = self.restore_snapshot(logdir)
        for n_epoch in range(n_epoch, self.hps.n_epochs):
            logger.info('Epoch {}, training'.format(n_epoch + 1))
            subsample = 4  # make validation more often
            for _ in range(subsample):
                self.train_on_images(train_images, subsample=subsample)
                if valid_images is None:
                    valid_images = [self.load_image(im_id)
                                    for im_id in sorted(valid_ids)]
                if valid_images:
                    self.validate_on_images(valid_images, subsample=subsample)
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
                [utils.mask_for_polygons(im_size, poly_by_type[poly_type + 1])
                 for poly_type in range(self.hps.total_classes)],
                dtype=np.uint8)
            with mask_path.open('wb') as f:
                np.save(f, mask)
        return Image(im_id, im_data, mask[self.hps.cls])

    def train_on_images(self, train_images: List[Image], subsample: int=1):
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
                im, (x, y) = self.sample_im_xy(train_images)
                if random.random() < self.hps.oversample:
                    for _ in range(1000):
                        if im.mask[x: x + s, y: y + s].sum():
                            break
                        im, (x, y) = self.sample_im_xy(train_images)
                patch = im.data[:, x - mb: x + s + mb, y - mb: y + s + mb]
                mask = im.mask[x - m: x + s + m, y - m: y + s + m]
                # TODO - mirror flips
                angle = random.random() * 360
                # TODO - do it without transpose?
                patch = (
                    utils.rotated(
                        patch.transpose([1, 2, 0]).astype(np.float32), angle)
                    .transpose([2, 0, 1]))
                mask = utils.rotated(mask.astype(np.float32), angle)
                inputs.append(patch[:, m: -m, m: -m])
                outputs.append(mask[m: -m, m: -m])

            return (torch.from_numpy(np.array(inputs)),
                    torch.from_numpy(np.array(outputs)))

        self._train_on_feeds(gen_batch, n_batches)

    def sample_im_xy(self, train_images):
        b = self.hps.patch_border
        s = self.hps.patch_inner
        # Extra margin for rotation
        m = int(np.ceil((np.sqrt(2) - 1) * (b + s / 2)))
        mb = m + b  # full margin
        im = random.choice(train_images)
        w, h = im.size
        return im, (random.randint(mb, w - (mb + s)),
                    random.randint(mb, h - (mb + s)))

    def _train_on_feeds(self, gen_batch, n_batches: int):
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
        log_step = 20
        im_log_step = n_batches // log_step * log_step
        with ThreadPool(processes=4) as pool:
            for i, (x, y) in enumerate(pool.imap_unordered(
                    gen_batch, range(n_batches), chunksize=10)):
                if losses and i % log_step == 0:
                    self._log_value(
                        'loss/cls-{}'.format(self.hps.cls),
                        np.mean(losses[-log_step:]))
                    pred_y = self.net(self._var(x)).data.cpu().numpy()
                    self._update_jaccard(tp_fp_fn, y.numpy(), pred_y)
                    self._log_jaccard(tp_fp_fn)
                    if i == im_log_step:
                        self._log_im(x.numpy(), y.numpy(), pred_y)
                loss = self.train_step(x, y)
                losses.append(loss)
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
        return {threshold: [[] for _ in range(3)]
                for threshold in self.hps.thresholds}

    def _update_jaccard(self, tp_fp_fn, mask, pred):
        assert len(mask.shape) == len(pred.shape)
        assert len(mask.shape) in {3, 4}
        for threshold, (tp, fp, fn) in tp_fp_fn.items():
            _tp, _fp, _fn = utils.mask_tp_fp_fn(pred, mask, threshold)
            tp.append(_tp)
            fp.append(_fp)
            fn.append(_fn)

    def _log_jaccard(self, tp_fp_fn, prefix=''):
        for threshold, (tp, fp, fn) in tp_fp_fn.items():
            self._log_value(
                '{}jaccard-{}/cls-{}'.format(prefix, threshold, self.hps.cls),
                self._jaccard(tp, fp, fn))

    def _jaccard(self, tp, fp, fn):
        if sum(tp) == 0:
            return 0
        return sum(tp) / (sum(tp) + sum(fn) + sum(fp))

    def _format_jaccard(self, tp_fp_fn):
        return ', '.join(
            'at {:.2f}: {:.3f}'.format(
                threshold, self._jaccard(tp, fp, fn))
            for threshold, (tp, fp, fn) in sorted(tp_fp_fn.items()))

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
            cv2.imwrite(fname('y'), y * 255)
            cv2.imwrite(fname('z'), p * 255)

    def _log_value(self, name, value):
        self.tb_logger.log_value(name, value, step=self.net.global_step[0])

    def validate_on_images(self, valid_images: List[Image],
                           subsample: int=1):
        self.net.eval()
        b = self.hps.patch_border
        s = self.hps.patch_inner
        losses = []
        tp_fp_fn = self._jaccard_store()
        for im in valid_images:
            w, h = im.size
            xs = range(b, w - (b + s), s)
            ys = range(b, h - (b + s), s)
            all_xy = [(x, y) for x in xs for y in ys]
            if subsample != 1:
                random.shuffle(all_xy)
                all_xy = all_xy[:len(all_xy) // subsample]
            pred_mask = np.zeros([w, h], dtype=np.float32)
            for xy_batch in utils.chunks(all_xy, self.hps.batch_size):
                inputs = np.array(
                    [im.data[:, x - b: x + s + b, y - b: y + s + b]
                     for x, y in xy_batch]).astype(np.float32)
                outputs = np.array(
                    [im.mask[x: x + s, y: y + s] for x, y in xy_batch])
                outputs = outputs.astype(np.float32)
                y_pred = self.net(self._var(torch.from_numpy(inputs)))
                loss = self.loss(
                    self._var(torch.from_numpy(outputs)), y_pred)
                losses.append(loss.data[0])
                y_pred_numpy = y_pred.data.cpu().numpy()
                for (x, y), mask in zip(xy_batch, y_pred_numpy):
                    pred_mask[x: x + s, y: y + s] = mask
                self._update_jaccard(tp_fp_fn, outputs, y_pred_numpy)
        loss = np.mean(losses)
        logger.info('Valid loss: {:.3f}, Jaccard: {}'.format(
            loss, self._format_jaccard(tp_fp_fn)))
        self._log_value(
            'valid-loss/cls-{}'.format(self.hps.cls), loss)
        self._log_jaccard(tp_fp_fn, prefix='valid-')

    def restore_snapshot(self, logdir: Path) -> int:
        for n_epoch in reversed(range(self.hps.n_epochs)):
            model_path = self._model_path(logdir, n_epoch)
            if Path(model_path).exists():
                logger.info('Loading snapshot {}'.format(model_path))
                state = torch.load(model_path)
                self.net.load_state_dict(state)
                return n_epoch + 1
        return 0

    def save_snapshot(self, n_epoch: int):
        model_path = self._model_path(self.logdir, n_epoch)
        logger.info('Saving snapshot {}'.format(model_path))
        torch.save(self.net.state_dict(), model_path)

    def _model_path(self, logdir: Path, n_epoch: int) -> str:
        return str(logdir.joinpath('model-{}'.format(n_epoch)))

    def predict_image_mask(self, im: Image) -> np.ndarray:
        self.net.eval()
        # FIXME - some copy-paste
        w, h = im.size
        b = self.hps.patch_border
        s = self.hps.patch_inner
        xs = range(b, w - (b + s), s)
        ys = range(b, h - (b + s), s)
        all_xy = [(x, y) for x in xs for y in ys]
        pred_mask = np.zeros([w, h], dtype=np.float32)
        self.net.eval()
        for xy_batch in tqdm.tqdm(
                list(utils.chunks(all_xy, self.hps.batch_size))):
            inputs = np.array(
                [im.data[:, x - b: x + s + b, y - b: y + s + b]
                 for x, y in xy_batch]).astype(np.float32)
            y_pred = self.net(self._var(torch.from_numpy(inputs)))
            for (x, y), mask in zip(xy_batch, y_pred.data.cpu().numpy()):
                pred_mask[x: x + s, y: y + s] = mask
        return pred_mask


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('cls', type=int, help='Class to train on')
    arg('logdir', type=str, help='Path to log directory')
    arg('--hps', type=str, help='Change hyperparameters in k1=v1,k2=v2 format')
    arg('--all', action='store_true',
        help='Train on all images without validation')
    arg('--stratified', type=int, default=1, help='stratified train/valid split')
    arg('--only', type=str,
        help='Train on this image ids only (comma-separated) without validation')
    arg('--clean', action='store_true', help='Clean logdir')
    args = parser.parse_args()
    hps = HyperParams(args.cls)
    hps.update(args.hps)
    pprint(attr.asdict(hps))
    logdir = Path(args.logdir)
    logdir.mkdir(exist_ok=True)
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
    elif args.stratified:
        mask_stats = utils.load_mask_stats()
        im_area = [(im_id, mask_stats[im_id][str(args.cls)]['area'])
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
    else:
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
    model.train(logdir=logdir, train_ids=train_ids, valid_ids=valid_ids)


if __name__ == '__main__':
    main()