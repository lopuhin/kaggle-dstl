#!/usr/bin/env python3
import argparse
import csv
from functools import partial
import logging
from pathlib import Path
import multiprocessing
from typing import Dict

import numpy as np
import shapely.affinity
from shapely.geometry import MultiPolygon
import shapely.wkt
import tensorflow as tf

import utils
from train import Model, HyperParams, Image


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] %(module)s: %(message)s'))
logger.addHandler(ch)


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', type=str, help='Path to log directory')
    arg('output', type=str, help='Submission csv')
    arg('--only', help='Only predict these image ids (comma-separated)')
    arg('--hps', type=str, help='Change hyperparameters in k1=v1,k2=v2 format')
    arg('--threshold', type=float, default=0.5)
    args = parser.parse_args()
    hps = HyperParams()
    hps.update(args.hps)

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
        for im_id in sorted(set(only or image_ids) - train_ids):
            im_path = store.joinpath(im_id)
            if im_path.exists():
                logger.info('Skip {}: already exists'.format(im_id))
                continue
            logger.info(im_id)
            im = Image(id=im_id,
                       data=model.preprocess_image(utils.load_image(im_id)))
            mask = model.image_prediction(im, sess) > args.threshold  # type: np.ndarray
            assert mask.shape[:2] == im.data.shape[:2]
            with im_path.open('wb') as f:
                np.save(f, mask)
            # cv2.imwrite(str(store.joinpath('{}_{}.png'.format(im_id, poly_type))),
            #            cls_mask.astype(np.int32) * 255)

    logger.info('Building polygons')
    with open(args.output, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        with multiprocessing.Pool(processes=4) as pool:
            for rows in pool.imap(partial(get_poly_data, store=store),
                                  image_ids):
                writer.writerows(rows)


def get_poly_data(im_id, store):
    mask_path = store.joinpath(im_id)
    train_polygons = utils.get_wkt_data().get(im_id)
    if train_polygons:
        return [(im_id, str(poly_type), poly)
                for poly_type, poly in train_polygons.items()]
    elif mask_path.exists():
        mask = np.load(str(mask_path))
        logger.info(im_id)
        poly_by_type = get_polygons(im_id, mask)
        return [(im_id, str(poly_type), shapely.wkt.dumps(polygons))
                for poly_type, polygons in sorted(poly_by_type.items())]
    else:
        logger.info('{} empty'.format(im_id))
        return [(im_id, str(cls + 1), 'MULTIPOLYGON EMPTY')
                for cls in range(10)]


def get_polygons(im_id: str, mask: np.ndarray) -> Dict[int, MultiPolygon]:
    im_size = mask.shape[:2]
    x_scaler, y_scaler = utils.get_scalers(im_id, im_size)
    x_scaler = 1 / x_scaler
    y_scaler = 1 / y_scaler
    poly_by_type = {}
    for cls in range(mask.shape[-1]):
        poly_type = cls + 1
        logger.info('{} poly_type {}'.format(im_id, poly_type))
        cls_mask = mask[:, :, cls]
        polygons = utils.mask_to_polygons(cls_mask)
        poly_by_type[poly_type] = shapely.affinity.scale(
            polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    return poly_by_type


if __name__ == '__main__':
    main()
