#!/usr/bin/env python3
import argparse
import csv
import json
from functools import partial
from pathlib import Path
from multiprocessing.pool import Pool
import traceback
from typing import List, Tuple

import cv2
import numpy as np
import shapely.affinity
from shapely.geometry import MultiPolygon
from shapely.geos import TopologicalError
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
    arg('--masks-only', action='store_true', help='Do only mask prediction')
    arg('--model-path', type=Path,
        help='Path to a specific model (if the last is not desired)')
    arg('--processes', type=int, default=4)
    arg('--debug', action='store_true', help='save masks and polygons as png')
    args = parser.parse_args()
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
        predict_masks(args, hps, store, to_predict)
    if args.masks_only:
        logger.info('Was building masks only, done.')
        return

    logger.info('Building polygons')
    with open(args.output, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        to_output = train_ids if args.train_only else (only or image_ids)
        jaccard_stats = [[] for _ in hps.classes]
        sizes = [0 for _ in hps.classes]
        with Pool(processes=args.processes) as pool:
            # The order will be wrong, but we don't care,
            # it's fixed in merge_submission.
            for rows, js in pool.imap_unordered(
                    partial(get_poly_data,
                            store=store,
                            threshold=args.threshold,
                            classes=hps.classes,
                            epsilon=args.epsilon,
                            min_area=args.min_area,
                            debug=args.debug,
                            train_only=args.train_only,
                            ),
                    to_output):
                assert len(rows) == hps.n_classes
                writer.writerows(rows)
                for cls_jss, cls_js in zip(jaccard_stats, js):
                    cls_jss.append(cls_js)
                for idx, (_, _, poly) in enumerate(rows):
                    sizes[idx] += len(poly)
        if args.train_only:
            for cls, cls_js in zip(hps.classes, jaccard_stats):
                intersection = union = tp = fp = fn = 0
                for (_tp, _fp, _fn), (_intersection, _union) in cls_js:
                    intersection += _intersection
                    union += _union
                    tp += _tp
                    fp += _fp
                    fn += _fn
                logger.info(
                    'cls-{} polygon jaccard: {:.5f}, mask jaccard: {:.5f}'
                    .format(cls,
                            intersection / union,
                            tp / (tp + fp + fn)))
        for cls, size in zip(hps.classes, sizes):
            logger.info('cls-{} size: {:,} bytes'.format(cls, size))


def mask_path(store: Path, im_id: str) -> Path:
    return store.joinpath('{}.mask'.format(im_id))


def predict_masks(args, hps, store, to_predict: List[str]):
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
        with mask_path(store, im.id).open('wb') as f:
            np.save(f, mask.astype(np.float16))


def get_poly_data(im_id, *,
                  store: Path,
                  classes: List[int],
                  threshold: float,
                  epsilon: float,
                  min_area: float,
                  debug: bool,
                  train_only: bool
                  ):
    train_polygons = utils.get_wkt_data().get(im_id)
    jaccard_stats = []
    if mask_path(store, im_id).exists():
        logger.info(im_id)
        masks = np.load(str(mask_path(store, im_id)))
        masks = masks > threshold  # type: np.ndarray
        rows = []
        for cls, mask in zip(classes, masks):
            poly_type = cls + 1
            unscaled, pred_poly = get_polygons(im_id, mask, epsilon, min_area)
            if debug:
                cv2.imwrite(
                    str(store / '{}_{}_poly_mask.png'.format(im_id, cls)),
                    255 * utils.mask_for_polygons(mask.shape, unscaled))
                cv2.imwrite(
                    str(store / '{}_{}_pixel_mask.png'.format(im_id, cls)),
                    255 * mask)
            if train_polygons:
                train_poly = train_polygons[poly_type]
                jaccard_stats.append(
                    log_jaccard(im_id, pred_poly, train_poly, mask, threshold))
            rows.append((im_id, str(poly_type),
                         shapely.wkt.dumps(pred_poly) if train_only else
                         'MULTIPOLYGON EMPTY'))
    else:
        logger.info('{} empty'.format(im_id))
        rows = [(im_id, str(cls + 1), 'MULTIPOLYGON EMPTY') for cls in classes]
    return rows, jaccard_stats


def get_polygons(im_id: str, mask: np.ndarray,
                 epsilon: float, min_area: float,
                 ) -> Tuple[MultiPolygon, MultiPolygon]:
    assert len(mask.shape) == 2
    x_scaler, y_scaler = utils.get_scalers(im_id, im_size=mask.shape)
    x_scaler = 1 / x_scaler
    y_scaler = 1 / y_scaler
    polygons = utils.mask_to_polygons(mask, epsilon=epsilon, min_area=min_area)
    return polygons, shapely.affinity.scale(
        polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


def log_jaccard(im_id, pred_poly, train_poly, mask, threshold):
    im_size = mask.shape
    train_poly = shapely.wkt.loads(train_poly)
    scaled_train_poly = utils.scale_to_mask(im_id, im_size, train_poly)
    true_mask = utils.mask_for_polygons(im_size, scaled_train_poly)
    assert len(mask.shape) == 2
    tp, fp, fn = tp_fp_fn = utils.mask_tp_fp_fn(mask, true_mask, threshold)
    try:
        intersection = pred_poly.intersection(train_poly).area
        union = pred_poly.union(train_poly).area
    except TopologicalError: # FIXME - wtf???
        traceback.print_exc()
        intersection = union = 0
    eps = 1e-15
    logger.info('polygon jaccard: {:.5f}, mask jaccard: {:.5f}'
                .format(intersection / (union + eps),
                        tp / (tp + fp + fn + eps)))
    return tp_fp_fn, (intersection, union)


if __name__ == '__main__':
    main()
