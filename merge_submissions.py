#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import utils  # for field_size_limit


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('root')
    args = parser.parse_args()
    root = Path(args.root)
    out_path = root.joinpath('submission.csv')
    header = None
    all_data = []
    for path in root.glob('*.csv'):
        if path != out_path:
            print('Reading {}'.format(path))
            with path.open() as f:
                reader = csv.reader(f)
                header = next(reader)
                all_data.extend(reader)
    all_data.sort(key=lambda x: (x[0], int(x[1])))
    with out_path.open('wt') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_data)


if __name__ == '__main__':
    main()
