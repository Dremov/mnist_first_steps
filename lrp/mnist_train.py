'''
@author: Vignesh Srinivasan
@author: Sebastian Lapuschkin
@author: Gregoire Montavon
@maintainer: Vignesh Srinivasan
@maintainer: Sebastian Lapuschkin 
@contact: vignesh.srinivasan@hhi.fraunhofer.de
@date: 20.12.2016
@version: 1.0+
@copyright: Copyright (c)  2016-2017, Vignesh Srinivasan, Sebastian Lapuschkin, Alexander Binder, Gregoire Montavon, Klaus-Robert Mueller, Wojciech Samek
@license : BSD-2-Clause
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
from modules.utils import Utils
from cnn_models import CNNModel
import input_data
import tensorflow as tf
import numpy as np

flags = tf.flags
logging = tf.logging

folder_name = 'models/'
data_dir = '../data/mnist'

model_index = CNNModel.selected_model
model_name = 'mnist_' + str(model_index)

path = folder_name + model_name

flags.DEFINE_integer("max_steps", 1001, 'Number of steps to run trainer.')
flags.DEFINE_integer("batch_size", 250, 'Number of steps to run trainer.')
flags.DEFINE_integer("test_every", 200, 'Number of steps to run trainer.')
flags.DEFINE_float("learning_rate", 0.01, 'Initial learning rate')
flags.DEFINE_float("dropout", 0.9, 'Keep probability for training dropout.')
flags.DEFINE_string("data_dir", data_dir, 'Directory for storing data')
flags.DEFINE_string("summaries_dir", path + '/logs','Summaries directory')
flags.DEFINE_boolean("relevance", False, 'Compute relevances')
flags.DEFINE_string("relevance_method", 'simple','relevance methods: simple/eps/w^2/alphabeta')
flags.DEFINE_boolean("save_model", True, 'Save the trained model')
flags.DEFINE_boolean("reload_model", False, 'Restore the trained model')
flags.DEFINE_string("checkpoint_dir", path + '/trained_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_reload_dir", path + '/trained_model','Checkpoint dir')

FLAGS = flags.FLAGS

def feed_dict(mnist, train):
    if train:
        xs, ys = mnist.train.next_batch(FLAGS.batch_size)
        k = FLAGS.dropout
    else:
        xs, ys = mnist.test.next_batch(FLAGS.batch_size)
        k = 1.0
    # return (2*xs)-1, ys, k
    return xs, ys, k

def train():
    # Import data
    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
    config = tf.ConfigProto(allow_soft_placement = True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 784], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
            keep_prob = tf.placeholder(tf.float32)

        with tf.variable_scope('model'):
            net = CNNModel.load_model(index=model_index, batch_size=FLAGS.batch_size)
            inp = tf.pad(tf.reshape(x, [FLAGS.batch_size,28,28,1]), [[0,0],[2,2],[2,2],[0,0]])
            op = net.forward(inp)
            y = tf.squeeze(op)
            trainer = net.fit(output=y,ground_truth=y_,loss='softmax_crossentropy',optimizer='adam', opt_params=[FLAGS.learning_rate])

        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)), tf.float32))

        tf.summary.scalar('accuracy', accuracy)

        # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        utils = Utils(sess, FLAGS.checkpoint_reload_dir)
        if FLAGS.reload_model:
            tvars = tf.trainable_variables()
            npy_files = np.load('mnist_trained_model/model.npy')
            [sess.run(tv.assign(npy_files[tt])) for tt,tv in enumerate(tvars)]

        for i in range(FLAGS.max_steps):
            if i % FLAGS.test_every == 0:  # test-set accuracy
                d = feed_dict(mnist, False)
                test_inp = {x:d[0], y_: d[1], keep_prob: d[2]}
                summary, acc, y1 = sess.run([merged, accuracy, y], feed_dict=test_inp)
                test_writer.add_summary(summary, i)
                print('Accuracy at step %s: %f' % (i, acc))

                # save model if required
                if FLAGS.save_model:
                    utils.save_model()

            else:
                d = feed_dict(mnist, True)
                inp = {x: d[0], y_: d[1], keep_prob: d[2]}
                summary, _, acc, op = sess.run([merged, trainer.train, accuracy, y], feed_dict=inp)
                print('Accuracy at step %s: %f' % (i, acc))
                train_writer.add_summary(summary, i)

    train_writer.close()
    test_writer.close()


def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
