#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess

from train import HyperParams


def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('out_dir')
    arg('--hps')
    args, other_args = parser.parse_known_args()
    root = Path(args.out_dir)
    root.mkdir(exist_ok=True)
    for cls in range(HyperParams.total_classes):
        hps = '{}{}'.format(
            '' if not args.hps else '{},'.format(args.hps),
            'classes={}'.format(cls))
        train_args = (['./train.py', str(root.joinpath(str(cls))), '--hps', hps]
                      + other_args)
        print('\nStarting training for class {}: {}\n'
              .format(cls, ' '.join(train_args)))
        try:
            subprocess.check_call(train_args)
        except subprocess.CalledProcessError as e:
            print('Error running training for class {}: {}'.format(cls, e))


if __name__ == '__main__':
    main()
