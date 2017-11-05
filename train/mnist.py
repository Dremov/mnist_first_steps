import numpy as np
import os
import struct
import sys
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K

K.set_image_dim_ordering('th')

"""
Loosely inspired by http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
which is GPL licensed.
"""


def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        print("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        yield get_img(i)


def show(image):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    from matplotlib import pyplot
    import matplotlib as mpl
    fig = pyplot.figure()
    ax = fig.add_subplot(1, 1, 1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    pyplot.show()


# print(len(training_data))
# label, pixels = training_data[765]
# print(label)
# print(pixels.shape)
# show(pixels)

""" Machine Learning Part """

nb_classes = 10

training_data = list(read(dataset='training', path='/data'))
testing_data = list(read(dataset='testing', path='/data'))

y_train, x_train = zip(*training_data)
y_test, x_test = zip(*testing_data)

x_train = np.asarray(x_train);
y_train = np.asarray(y_train);
x_test = np.asarray(x_test);
y_test = np.asarray(y_test);

x_train = x_train.reshape(60000, 1, 28, 28)
x_test = x_test.reshape(10000, 1, 28, 28)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print("Training matrix shape", x_train.shape)
print("Testing matrix shape", x_test.shape)

X_train = x_train
X_test = x_test
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(1, 28, 28)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 128
model.fit(X_train, Y_train,
          batch_size=batch_size, epochs=3,
          verbose=1, validation_data=(X_test, Y_test))

score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)

save_folder = './save/'
model.save(os.path.join(save_folder, 'mnist_model.h5'))

score_str = 'Test score: ' + str(score[0]) + '\n'
acc_str = 'Test accuracy:' + str(score[1])

print(score_str)
print(acc_str)

text_file = open(save_folder + 'MNIST_Score.txt', "w+")
text_file.write(score_str)
text_file.write(acc_str)
text_file.close()