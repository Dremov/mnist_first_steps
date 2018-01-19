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

from PIL import Image
from array import *
from cnn_models import CNNModel
import tensorflow as tf
import numpy as np
import modules.render as render
import skimage.io as io
import warnings
warnings.filterwarnings("ignore")

image_dim = 28
model_index = 1
model_name = 'mnist_' + str(model_index)

flags = tf.flags
flags.DEFINE_integer("batch_size", 1, 'Number of images to load.')
flags.DEFINE_string("summaries_dir", 'models/lrp_data', 'Summaries directory')
flags.DEFINE_string("checkpoint_dir", './models/' + model_name + '/trained_model', 'Checkpoint dir')
FLAGS = flags.FLAGS


class ComputeLRP():

    def __init__(self):
        self.RELEVANCE = None
        self.merged = None
        self.output = None
        self.sess = None
        self.x = None
        self.init()

    def rgb2gray(self, rgb):
        r = rgb[0]
        g = rgb[1]
        b = rgb[2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return int(gray)

    def init_vars(self):
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt is None:
            raise ValueError('Error: TensorFlow model not found in: ' + FLAGS.checkpoint_dir)
        try:
            saver = tf.train.Saver()
            tf.global_variables_initializer().run(session=self.sess)
            print('Reloading ' + ckpt.model_checkpoint_path)
            saver.restore(self.sess, ckpt.model_checkpoint_path)
        except:
            raise ValueError('Unknown error. Possibly model is incorrect.')
        return saver

    def save_heatmap(self, predictions, relevances):
        for h, heat in enumerate(relevances):
            heatmap = render.hm_to_rgb(heat)
            pred = np.argmax(predictions[h])
            io.imsave(str(pred) + '.png', heatmap)

    def load_image_data(self, img):
        image_data = array('B')
        pixel = img.load()
        width, height = img.size
        for x in range(0, width):
            for y in range(0, height):
                # image_data.append(pixel[y, x])
                gray = self.rgb2gray(pixel[y, x])
                image_data.append(gray)
        return np.asarray(image_data)

    def init(self):
        self.sess = tf.Session()

        self.x = tf.placeholder(tf.float32, [FLAGS.batch_size, 784], name='input')
        with tf.variable_scope('model'):
            my_network = CNNModel.load_model(index=model_index, batch_size=FLAGS.batch_size)
            self.output = my_network.forward(self.x)
            self.RELEVANCE = my_network.lrp(self.output, 'simple', 1.0)

        # Merge all the summaries
        self.merged = tf.summary.merge_all()

        # Intialize variables and reload your model
        self.init_vars()

    def compute_lrp(self, image):
        # set input image
        image = image.resize((image_dim, image_dim))
        io.imsave('input.png', image)
        xs = self.load_image_data(image)

        # Pass the test data to the restored model
        xs = xs.reshape([FLAGS.batch_size, 784])
        summary, relevance_test = self.sess.run([self.merged, self.RELEVANCE], feed_dict={self.x: xs})
        predictions = self.sess.run(self.output, feed_dict={self.x: xs})
        relevances = relevance_test.reshape([FLAGS.batch_size, 28, 28, 1])

        heatmap = render.hm_to_rgb(relevances[0])
        pred = np.argmax(predictions[0])
        
        return heatmap, pred
        #self.save_heatmap(predictions, relevances)

if __name__ == '__main__':
    lrp = ComputeLRP()
    image = Image.open("number.png")
    lrp.compute(image)
    lrp.sess.close()
