import argparse
from typing import List

import attr
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

import utils


@attr.s(slots=True)
class HyperParams:
    input_size = attr.ib(default=20)
    n_channels = attr.ib(default=3)
    n_outputs = attr.ib(default=10)
    learning_rate = attr.ib(default=0.1)
    hidden_dim = attr.ib(default=100)
    n_epochs = attr.ib(default=10)
    batch_size = attr.ib(default=32)

    def update(self, hps_string: str):
        if hps_string:
            for pair in hps_string.split(','):
                k, v = pair.split('=')
                v = float(v) if '.' in v else int(v)
                setattr(self, k, v)


class Model:
    def __init__(self, hps: HyperParams):
        self.hps = hps
        self.x = tf.placeholder(
            tf.float32, [None, hps.input_size, hps.input_size, hps.n_channels])
        self.y = tf.placeholder(tf.float32, [None, hps.n_outputs])

        input_dim = hps.input_size ** 2 * hps.n_channels
        x = tf.reshape(self.x, [-1, input_dim])
        w0 = tf.get_variable('w0', shape=[input_dim, hps.hidden_size])
        b0 = tf.get_variable('b0', shape=[hps.hidden_size],
                             initializer=tf.zeros_initializer)
        x_hidden = tf.nn.relu(tf.nn.xw_plus_b(self.x, w0, b0))

        w1 = tf.get_variable('w1', shape=[hps.hidden_size, hps.n_outputs])
        b1 = tf.get_variable('b0', shape=[hps.n_outputs],
                             initializer=tf.zeros_initializer)
        x_logits = tf.nn.xw_plus_b(x_hidden, w1, b1)
        self.x_out = tf.nn.sigmoid(x_logits)

        self.loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(x_logits, self.y))
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=hps.learning_rate)
        self.train_op = optimizer.minimize(
            self.loss * hps.batch_size, self.global_step)

    def train(self, logdir: str, im_ids: List[str]):
        im_ids = list(im_ids)
        sv = tf.train.Supervisor(
            logdir=logdir,
            summary_op=None,
            global_step=self.global_step,
            save_summaries_secs=10,
            save_model_secs=60,
        )
        with sv.managed_session() as sess:
            for n_epoch in range(self.hps.n_epochs):
                np.random.shuffle(im_ids)
                for im_id in im_ids:
                    self.train_on_image(sess, im_id)

    def train_on_image(self, sess: tf.Session, im_id: str):
        im_data = utils.load_image(im_id)
        # TODO - get classes

    def extract_patch(self, im_data: np.ndarray, x: int, y: int) -> np.ndarray:
        # TODO - pad at the edges
        window = self.hps.input_size / 2
        x += 0.5
        y += 0.5
        return im_data[:,
                       int(x - window) : int(x + window),
                       int(y - window) : int(y + window)]




def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('logdir', type=str, help='Path to log directory')
    arg('--hps', type=str, help='Change hyperparameters in k1=v1,k2=v2 format')
    args = parser.parse_args()
    hps = HyperParams()
    hps.update(args.hps)

    model = Model(hps=hps)
    all_img_ids = list(utils.WKT_DATA)
    train_ids, valid_ids = train_test_split(all_img_ids)
    model.train(logdir=args.logdir, im_ids=train_ids)


if __name__ == '__main__':
    main()