#!/usr/bin/env python3
import argparse
import csv
import gzip

import utils  # for field_size_limit


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('output')
    arg('inputs', nargs='+')
    arg('--skip', nargs='+', help='{im_id}_{poly_type} format, e.g 6100_1_1_10')
    arg('--cls', type=int, nargs='+', help='leave other classes empty')
    args = parser.parse_args()
    skip = set(args.skip or [])
    all_data = {}
    for path in args.inputs:
        print('Reading {}'.format(path))
        opener = gzip.open if path.endswith('.gz') else open
        with opener(path, 'rt') as f:
            reader = csv.reader(f)
            next(reader)
            all_data.update(
                ((im_id, poly_type), poly) for im_id, poly_type, poly in reader)

    opener = gzip.open if args.output.endswith('.gz') else open
    with opener(str(args.output), 'wt') as outf:
        writer = csv.writer(outf)
        with open('sample_submission.csv') as f:
            reader = csv.reader(f)
            header = next(reader)
            writer.writerow(header)
            for im_id, poly_type, _ in reader:
                poly = 'MULTIPOLYGON EMPTY'
                cls = int(poly_type) - 1
                if ('{}_{}'.format(im_id, poly_type) not in skip
                        and (not args.cls or cls in args.cls)):
                    poly = all_data.get((im_id, poly_type)) or poly
                writer.writerow([im_id, poly_type, poly])


if __name__ == '__main__':
    main()
