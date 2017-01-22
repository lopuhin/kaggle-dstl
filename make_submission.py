#!/usr/bin/env python3
import argparse
import csv
import json
from functools import partial
from pathlib import Path
import multiprocessing

import numpy as np
import shapely.affinity
from shapely.geometry import MultiPolygon
import shapely.wkt
import tensorflow as tf

import utils
from train import Model, HyperParams, Image


logger = utils.get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', type=str, help='Path to log directory')
    arg('output', type=str, help='Submission csv')
    arg('--train-only', action='store_true', help='Predict only train images')
    arg('--only', help='Only predict these image ids (comma-separated)')
    arg('--threshold', type=float, default=0.5)
    arg('--epsilon', type=float, default=5.0, help='smoothing')
    args = parser.parse_args()
    hps = HyperParams(**json.loads(
        Path(args.logdir).joinpath('hps.json').read_text()))

    only = set(args.only.split(',')) if args.only else set()
    with open('sample_submission.csv') as f:
        reader = csv.reader(f)
        header = next(reader)
        image_ids = [im_id for im_id, cls, _ in reader if cls == '1']

    store = Path(args.output.split('.csv')[0])
    store.mkdir(exist_ok=True)

    model = Model(hps=hps)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(args.logdir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        logger.info('Predicting masks')
        train_ids = set(utils.get_wkt_data())
        to_predict = train_ids if args.train_only else set(only or image_ids)
        for im_id in sorted(to_predict):
            im_path = store.joinpath(im_id)
            if im_path.exists():
                logger.info('Skip {}: already exists'.format(im_id))
                continue
            logger.info(im_id)
            im = Image(id=im_id,
                       data=model.preprocess_image(utils.load_image(im_id)))
            mask = model.image_prediction(im).astype(np.float16)
            assert mask.shape == im.data.shape[:2]
            with im_path.open('wb') as f:
                np.save(f, mask)

    logger.info('Building polygons')
    with open(args.output, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        to_output = train_ids if args.train_only else (only or image_ids)
        with multiprocessing.Pool(processes=4) as pool:
            for row in pool.imap(
                    partial(get_poly_data,
                            store=store,
                            threshold=args.threshold,
                            cls=args.cls,
                            epsilon=args.epsilon,
                            ),
                    to_output):
                writer.writerow(row)


def get_poly_data(im_id, *, store, cls: int,
                  threshold: float, epsilon: float):
    poly_type = cls + 1
    mask_path = store.joinpath(im_id)
    train_polygons = utils.get_wkt_data().get(im_id)
    if mask_path.exists():
        logger.info(im_id)
        mask = np.load(str(mask_path))
        mask = mask > threshold  # type: np.ndarray
        poly_by_type = get_polygons(im_id, mask, epsilon, cls)
        if train_polygons:
            log_jaccard(im_id, poly_by_type, train_polygons, mask,
                        cls, threshold)
            return im_id, str(poly_type), train_polygons[poly_type]
        else:
            return (im_id, str(poly_type),
                    shapely.wkt.dumps(poly_by_type[poly_type]))
    else:
        logger.info('{} empty'.format(im_id))
        return im_id, str(poly_type), 'MULTIPOLYGON EMPTY'


def get_polygons(im_id: str, mask: np.ndarray, epsilon: float, cls: int)\
        -> MultiPolygon:
    x_scaler, y_scaler = utils.get_scalers(im_id, im_size=mask.shape)
    x_scaler = 1 / x_scaler
    y_scaler = 1 / y_scaler
    polygons = utils.mask_to_polygons(mask, epsilon=epsilon)
    return shapely.affinity.scale(
        polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))


def log_jaccard(im_id, poly_by_type, train_polygons, mask, cls, threshold):
    im_size = mask.shape[:2]
    poly_type = cls + 1
    poly = poly_by_type[poly_type]
    train_poly = shapely.wkt.loads(train_polygons[poly_type])
    scaled_train_poly = utils.scale_to_mask(im_id, im_size, train_poly)
    true_mask = utils.mask_for_polygons(im_size, scaled_train_poly)
    tp, fp, fn = utils.mask_tp_fp_fn(mask, true_mask, threshold)
    eps = 1e-15
    logger.info(
        'polygon jaccard: {:.5f}, mask jaccard: {:.5f}'
        .format(poly.intersection(train_poly).area /
                (poly.union(train_poly).area + eps),
                tp / (tp + fp + fn + eps),
                ))


if __name__ == '__main__':
    main()
