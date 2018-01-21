'''
@author: Vignesh Srinivasan
@author: Sebastian Lapushkin
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
from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.convolution import Convolution
from modules.flatten import Flatten
from modules.maxpool import MaxPool
from modules.avgpool import AvgPool
import modules.render as render
import input_data
from modules.utils import Utils, Summaries, plot_relevances
import matplotlib.pyplot as plt
import skimage
from skimage.viewer import ImageViewer
import skimage.color as color
import skimage.transform as transform
import skimage.io as io
from PIL import Image
from array import *
from cnn_models import CNNModel

import argparse
import tensorflow as tf
import numpy as np
import pdb
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

flags = tf.flags
logging = tf.logging

read_single_images = False
data_dir = '/mnist'

model_index = 1
model_name = '/mnist_' + str(model_index)

output_path = './models' + model_name
model_path = './models' + model_name

flags.DEFINE_integer("batch_size", 300, 'Number of images to load.')
flags.DEFINE_string("data_dir", '../data' + data_dir, 'Directory for storing data')
flags.DEFINE_string("summaries_dir", model_path + '/lrp_data','Summaries directory')
flags.DEFINE_boolean("relevance", True, 'Compute relevances')
flags.DEFINE_string("checkpoint_dir", model_path + '/trained_model','Checkpoint dir')
flags.DEFINE_string("checkpoint_reload_dir", model_path + '/trained_model','Checkpoint dir')

FLAGS = flags.FLAGS
labelCount = np.zeros(10, dtype=np.int)

def rgb2gray(rgb):
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return int(gray)

def init_vars(sess):
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    try:
        saver = tf.train.Saver()
        tf.global_variables_initializer().run()
        print('Reloading ' + ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    except:
        raise ValueError('Error: Maybe model not found or error in/with model.')
    return saver


def save_images(outpath, predictions, relevances, images, ys, iter):
    n, w, h, dim = relevances.shape
    heatmaps = []

    if images is not None:
        assert relevances.shape == images.shape, 'Relevances shape != Images shape'
    for h, heat in enumerate(relevances):
        if images is not None:
            input_image = images[h]
            maps = render.hm_to_rgb(heat, input_image, scaling=3, sigma=2)
            mapsz = render.hm_to_rgb(heat, scaling=3)
        else:
            maps = render.hm_to_rgb(heat, scaling=3, sigma=2)
        heatmaps.append(maps)
        rgb = color.grey2rgb(images[h])
        s = rgb.shape
        rgb = rgb.reshape(s[0], s[1], s[3])
        rgb = transform.rescale(rgb, 3)

        for i in ys[h]:
            if i == 1.:
                num = np.nonzero(ys[h])[0]
                label = num[0]

        pred = np.argmax(predictions[h])

        preds = predictions[h][0][0]
        conf = (max(preds)/sum(preds)) * 100


        if (labelCount[label] < 10) and (label != pred):
            image_idx = str(h * (iter + 1))
            io.imsave(outpath + '/' + str(conf) + '% label-' + str(label) + '-idx-' + image_idx + '-predicted-' + str(pred) + '-norm.png', rgb)
            io.imsave(outpath + '/' + str(conf) + '% label-' + str(label) + '-idx-' + image_idx + '-predicted-' + str(pred) + '-heat.png', maps)
            io.imsave(outpath + '/' + str(conf) + '% label-' + str(label) + '-idx-' + image_idx + '-predicted-' + str(pred) + '-heatn.png', mapsz)
            labelCount[label] += 1

    num_items = 0
    for i in labelCount:
        num_items += i


    print(labelCount)
    print("Saved Images: " + str(num_items))
    print("###############################")

    if num_items >= 100:
        sys.exit()


def load_images(path):

    data_image = array('B')
    data_label = array('B')

    FileList = []
    for label in os.listdir(path):
        p = os.path.join(path, label)
        if os.path.isdir(p):
            for file in os.listdir(p):
                if file.endswith(".png"):
                    img_path = os.path.join(path, label, file)
                    FileList.append(img_path)
                    data_label.append(int(label))

    for filename in os.listdir(path):
        if filename.endswith(".png"):
            FileList.append(os.path.join(path, filename))

    for filename in FileList:
        Im = Image.open(filename)
        pixel = Im.load()
        width, height = Im.size
        for x in range(0, width):
            for y in range(0, height):
                if read_single_images:
                    gray = rgb2gray(pixel[y, x])
                    data_image.append(gray)
                else:
                    data_image.append(pixel[y, x])

    return data_image, data_label

def test():

    mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

    outpath = 'tables/out' + model_name
    shutil.rmtree(outpath, ignore_errors=True)
    os.mkdir(outpath)

    # if read_single_images:
    #     xs, ys = load_images(FLAGS.data_dir)
    #     xs = np.asarray(xs)
    #     FLAGS.batch_size = int(xs.shape[0] / 784)

    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, 784], name='input')
        with tf.variable_scope('model'):
            my_network = CNNModel.load_model(index=model_index, batch_size=FLAGS.batch_size)
            output = my_network.forward(x)
            if FLAGS.relevance:
                RELEVANCE = my_network.lrp(output, 'simple', 1.0)

        # Merge all the summaries and write them out
        merged = tf.summary.merge_all()
        test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/my_model')

        # Intialize variables and reload your model
        init_vars(sess)

        for i in range(0, 1):
            xs, ys = mnist.test.next_batch(FLAGS.batch_size, shuffle=False)

            # Pass the test data to the restored model
            xs = xs.reshape([FLAGS.batch_size, 784])
            # summary, relevance_test = sess.run([merged, RELEVANCE], feed_dict={x:(2*xs)-1})
            summary, relevance_test = sess.run([merged, RELEVANCE], feed_dict={x:xs})
            test_writer.add_summary(summary, 0)

            predictions = sess.run(output, feed_dict={x: xs})

#############
            # predictions = sess.run([model.logits], feed_dict={model.input: X_batch, model.keep_prob: 1.0})
            #
            # probs = tf.nn.softmax(logits)
            #
            # predictions = sess.run(model.probs, feed_dict=feed_dict)
#############

            # print(output)
            # Save the images as heatmaps to visualize on tensorboard
            images = xs.reshape([FLAGS.batch_size, 28, 28, 1])
            # images = (images + 1)/2.0
            relevances = relevance_test.reshape([FLAGS.batch_size, 28, 28, 1])

            print(type(output))

            # plot_relevances(relevances, images, test_writer)
            save_images(outpath, predictions, relevances, images, ys, i)

            test_writer.close()


    
def main(_):
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

    test()


if __name__ == '__main__':
    tf.app.run()
