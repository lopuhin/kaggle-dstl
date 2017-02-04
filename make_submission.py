#!/usr/bin/env python3
import argparse
import csv
import json
import gzip
from functools import partial
from pathlib import Path
from multiprocessing.pool import Pool
from typing import List, Tuple, Set

import cv2
import numpy as np
import shapely.affinity
from shapely.geometry import MultiPolygon
import shapely.wkt

import utils
from train import Model, HyperParams, Image


logger = utils.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', type=Path, help='Path to log directory')
    arg('output', type=str, help='Submission csv')
    arg('--train-only', action='store_true', help='Predict only train images')
    arg('--only', help='Only predict these image ids (comma-separated)')
    arg('--threshold', type=float, default=0.5)
    arg('--epsilon', type=float, default=5.0, help='smoothing')
    arg('--min-area', type=float, default=50.0)
    arg('--min-car-area', type=float, default=10.0)
    arg('--masks-only', action='store_true', help='Do only mask prediction')
    arg('--model-path', type=Path,
        help='Path to a specific model (if the last is not desired)')
    arg('--processes', type=int, default=30)
    arg('--debug', action='store_true', help='save masks and polygons as png')
    arg('--fix', nargs='+')
    args = parser.parse_args()
    to_fix = set(args.fix or [])
    hps = HyperParams(**json.loads(
        args.logdir.joinpath('hps.json').read_text()))

    only = set(args.only.split(',')) if args.only else set()
    with open('sample_submission.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        image_ids = [im_id for im_id, cls, _ in reader if cls == '1']

    store = args.logdir  # type: Path

    train_ids = set(utils.get_wkt_data())
    to_predict = set(train_ids)
    if not args.train_only:
        to_predict |= set(only or image_ids)
    to_predict = [im_id for im_id in to_predict
                  if not mask_path(store, im_id).exists()]

    if to_predict:
        predict_masks(args, hps, store, to_predict, args.threshold)
    if args.masks_only:
        logger.info('Was building masks only, done.')
        return

    logger.info('Building polygons')
    opener = gzip.open if args.output.endswith('.gz') else open
    with opener(args.output, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        to_output = train_ids if args.train_only else (only or image_ids)
        jaccard_stats = [[] for _ in hps.classes]
        sizes = [0 for _ in hps.classes]
        with Pool(processes=args.processes) as pool:
            for rows, js in pool.imap(
                    partial(get_poly_data,
                            store=store,
                            classes=hps.classes,
                            epsilon=args.epsilon,
                            min_area=args.min_area,
                            min_car_area=args.min_car_area,
                            debug=args.debug,
                            train_only=args.train_only,
                            to_fix=to_fix,
                            hps=hps,
                            ),
                    to_output):
                assert len(rows) == hps.n_classes
                writer.writerows(rows)
                for cls_jss, cls_js in zip(jaccard_stats, js):
                    cls_jss.append(cls_js)
                for idx, (_, _, poly) in enumerate(rows):
                    sizes[idx] += len(poly)
        if args.train_only and args.debug:
            pixel_jaccards, poly_jaccards = [], []
            for cls, cls_js in zip(hps.classes, jaccard_stats):
                pixel_jc, poly_jc = [np.array([0, 0, 0]) for _ in range(2)]
                for _pixel_jc, _poly_jc in cls_js:
                    pixel_jc += _pixel_jc
                    poly_jc += _poly_jc
                logger.info(
                    'cls-{}: pixel jaccard: {:.5f}, polygon jaccard: {:.5f}'
                    .format(cls, jaccard(pixel_jc), jaccard(poly_jc)))
                pixel_jaccards.append(jaccard(pixel_jc))
                poly_jaccards.append(jaccard(poly_jc))
            logger.info(
                'Mean pixel jaccard: {:.5f}, polygon jaccard: {:.5f}'
                .format(np.mean(pixel_jaccards), np.mean(poly_jaccards)))
        for cls, size in zip(hps.classes, sizes):
            logger.info('cls-{} size: {:,} bytes'.format(cls, size))


def mask_path(store: Path, im_id: str) -> Path:
    return store.joinpath('{}.bin-mask.gz'.format(im_id))


def predict_masks(args, hps, store, to_predict: List[str], threshold: float):
    logger.info('Predicting {} masks: {}'
                .format(len(to_predict), ', '.join(sorted(to_predict))))
    model = Model(hps=hps)
    if args.model_path:
        model.restore_snapshot(args.model_path)
    else:
        model.restore_last_snapshot(args.logdir)

    def load_im(im_id):
        return Image(id=im_id,
                     data=model.preprocess_image(utils.load_image(im_id)))

    def predict_mask(im):
        logger.info(im.id)
        return im, model.predict_image_mask(im)

    im_masks = map(predict_mask, utils.imap_fixed_output_buffer(
        load_im, sorted(to_predict), threads=2))

    for im, mask in utils.imap_fixed_output_buffer(
            lambda _: next(im_masks), to_predict, threads=1):
        assert mask.shape[1:] == im.data.shape[1:]
        with gzip.open(str(mask_path(store, im.id)), 'wb') as f:
            # TODO - maybe do (mask * 20).astype(np.uint8)
            np.save(f, mask >= threshold)


def get_poly_data(im_id, *,
                  store: Path,
                  classes: List[int],
                  epsilon: float,
                  min_area: float,
                  min_car_area: float,
                  debug: bool,
                  train_only: bool,
                  to_fix: Set[str],
                  hps: HyperParams
                  ):
    train_polygons = utils.get_wkt_data().get(im_id)
    jaccard_stats = []
    path = mask_path(store, im_id)
    if path.exists():
        logger.info(im_id)
        with gzip.open(str(path), 'rb') as f:
            masks = np.load(f)  # type: np.ndarray
        rows = []
        for cls, mask in zip(classes, masks):
            poly_type = cls + 1
            if train_only or not train_polygons:
                unscaled, pred_poly = get_polygons(
                    im_id, mask, epsilon,
                    min_area=min_car_area if cls in {8, 9} else min_area,
                    fix='{}_{}'.format(im_id, poly_type) in to_fix,
                )
                if debug:
                    poly_mask = utils.mask_for_polygons(mask.shape, unscaled)
                    cv2.imwrite(
                        str(store / '{}_{}_poly_mask.png'.format(im_id, cls)),
                        255 * poly_mask)
                    cv2.imwrite(
                        str(store / '{}_{}_pixel_mask.png'.format(im_id, cls)),
                        255 * mask)
                rows.append(
                    (im_id, str(poly_type),
                     shapely.wkt.dumps(pred_poly, rounding_precision=8)))
            elif train_polygons:
                rows.append((im_id, str(poly_type), 'MULTIPOLYGON EMPTY'))
            else:
                assert False
            if train_only and train_polygons and debug:
                train_poly = train_polygons[poly_type]
                jaccard_stats.append(
                    log_jaccard(im_id, train_poly, mask, poly_mask, hps))
    else:
        logger.info('{} empty'.format(im_id))
        rows = [(im_id, str(cls + 1), 'MULTIPOLYGON EMPTY') for cls in classes]
    return rows, jaccard_stats


def get_polygons(im_id: str, mask: np.ndarray,
                 epsilon: float, min_area: float, fix: bool
                 ) -> Tuple[MultiPolygon, MultiPolygon]:
    assert len(mask.shape) == 2
    x_scaler, y_scaler = utils.get_scalers(im_id, im_size=mask.shape)
    x_scaler = 1 / x_scaler
    y_scaler = 1 / y_scaler
    polygons = utils.mask_to_polygons(
        mask, epsilon=epsilon, min_area=min_area, fix=fix)
    return polygons, shapely.affinity.scale(
        polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


def log_jaccard(im_id, train_poly, mask, poly_mask, hps):
    im_size = mask.shape
    train_poly = shapely.wkt.loads(train_poly)
    scaled_train_poly = utils.scale_to_mask(im_id, im_size, train_poly)
    true_mask = utils.mask_for_polygons(im_size, scaled_train_poly)
    assert len(mask.shape) == 2
    square = lambda x: x[:hps.validation_square, :hps.validation_square]
    mask = square(mask)
    poly_mask = square(poly_mask)
    true_mask = square(true_mask)
    pixel_jc = utils.mask_tp_fp_fn(mask, true_mask, 0.5)
    poly_jc = utils.mask_tp_fp_fn(poly_mask, true_mask, 0.5)
    logger.info('pixel jaccard: {:.5f}, polygon jaccard: {:.5f}'
                .format(jaccard(pixel_jc), jaccard(poly_jc)))
    return pixel_jc, poly_jc


def jaccard(tp_fp_fn):
    return tp_fp_fn[0] / (sum(tp_fp_fn) + 1e-15)


if __name__ == '__main__':
    main()
