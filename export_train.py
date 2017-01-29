#!/usr/bin/env python3
import argparse
from pathlib import Path
import json

import cv2
import tabulate

import utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', help='output director')
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(exist_ok=True)
    poly_stats = {}
    for im_id in sorted(utils.get_wkt_data()):
        print(im_id)
        im_data = utils.load_image(im_id, rgb_only=True)
        im_data = utils.scale_percentile(im_data)
        cv2.imwrite(str(output.joinpath('{}.jpg'.format(im_id))), 255 * im_data)
        im_size = im_data.shape[:2]
        poly_by_type = utils.load_polygons(im_id, im_size)
        for poly_type, poly in sorted(poly_by_type.items()):
            cls = poly_type - 1
            mask = utils.mask_for_polygons(im_size, poly)
            cv2.imwrite(
                str(output.joinpath('{}_mask_{}.png'.format(im_id, cls))),
                255 * mask)
            poly_stats.setdefault(im_id, {})[cls] = {
                'area': poly.area / (im_size[0] * im_size[1]),
                'perimeter': int(poly.length),
                'number': len(poly),
            }

    output.joinpath('stats.json').write_text(json.dumps(poly_stats))

    for key in ['number', 'perimeter', 'area']:
        if key == 'area':
            fmt = '{:.4%}'.format
        else:
            fmt = lambda x: x
        print('\n{}'.format(key))
        print(tabulate.tabulate(
            [[im_id] + [fmt(s[cls][key]) for cls in range(10)]
             for im_id, s in sorted(poly_stats.items())],
            headers=['im_id'] + list(range(10))))


if __name__ == '__main__':
    main()
