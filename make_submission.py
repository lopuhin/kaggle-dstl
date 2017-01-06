#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import shapely.affinity
from shapely.geometry import MultiPolygon
import shapely.wkt
import tensorflow as tf

import utils
from train import Model, HyperParams, Image


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

    debug_store = Path(args.output.split('.csv')[0])
    debug_store.mkdir(exist_ok=True)

    model = Model(hps=hps)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state(args.logdir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        submission_data = []
        for im_id in image_ids:
            print(im_id)
            if (only and im_id not in only) or im_id not in utils.get_wkt_data():
                submission_data.extend(
                    (im_id, str(cls + 1), 'MULTIPOLYGON EMPTY')
                    for cls in range(10))
            else:
                for poly_type, poly in utils.get_wkt_data()[im_id].items():
                    poly = shapely.wkt.loads(poly)
                    submission_data.append(
                        (im_id, str(poly_type), shapely.wkt.dumps(poly)))
               #im = Image(id=im_id,
               #           data=model.preprocess_image(utils.load_image(im_id)))
               #poly_by_type = get_polygons(
               #    im, model, sess, args.threshold, debug_store)
               #for poly_type, polygons in sorted(poly_by_type.items()):
               #    submission_data.append(
               #        (im_id, str(poly_type), shapely.wkt.dumps(polygons)))

    with open(args.output, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(submission_data)


def get_polygons(im: Image, model: Model, sess: tf.Session, threshold: float,
                 debug_store: Path) -> Dict[int, MultiPolygon]:
    mask = model.image_prediction(im, sess)
    im_size = im.data.shape[:2]
    x_scaler, y_scaler = utils.get_scalers(im.id, im_size)
    x_scaler = 1 / x_scaler
    y_scaler = 1 / y_scaler
    poly_by_type = {}
    for cls in range(mask.shape[-1]):
        poly_type = cls + 1
        print('poly_type', poly_type)
        cls_mask = mask[:, :, cls] > threshold
        cv2.imwrite(str(debug_store.joinpath('{}_{}.png'.format(im.id, poly_type))),
                    cls_mask.astype(np.int32) * 255)
        polygons = utils.mask_to_polygons(cls_mask)
        poly_by_type[poly_type] = shapely.affinity.scale(
            polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    return poly_by_type


if __name__ == '__main__':
    main()
