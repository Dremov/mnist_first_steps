from modules.sequential import Sequential
from modules.linear import Linear
from modules.softmax import Softmax
from modules.relu import Relu
from modules.tanh import Tanh
from modules.convolution import Convolution
from modules.avgpool import AvgPool
from modules.maxpool import MaxPool
from modules.utils import Utils, Summaries, plot_relevances
import input_data

class CNNModel:
    img_rows = 28
    img_cols = 28
    batch_size = 128

    selected_model = 1

    @staticmethod
    def load_inputshape():
        return CNNModel.img_rows, CNNModel.img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        return x_train, x_test

    @staticmethod
    def load_model(index=1, batch_size=128, classes=10):
        if index == 1:
            return Sequential([
                Convolution(output_depth=10, input_depth=1, batch_size=batch_size, input_dim=28,
                                           act='relu', stride_size=1, pad='VALID'),
                AvgPool(),
                Convolution(output_depth=25, stride_size=1, act='relu', pad='VALID'),
                AvgPool(),
                Convolution(kernel_size=4, output_depth=100, stride_size=1, act='relu', pad='VALID'),
                AvgPool(),
                Convolution(kernel_size=1, output_depth=10, stride_size=1, pad='VALID'),
                Softmax()])

        if index == 2:
            return Sequential([
                Convolution(output_depth=32, input_depth=1, batch_size=batch_size, stride_size=1, kernel_size=5,
                            act='relu', pad='SAME', input_dim=28),
                MaxPool(),
                Convolution(output_depth=64, kernel_size=5, act='relu', pad='SAME'),
                MaxPool(),
                Convolution(output_depth=128, kernel_size=5, act='relu', pad='SAME'),
                MaxPool(),
                Linear(1024, act='relu', keep_prob=0.5),
                Linear(512, act='relu', keep_prob=0.5),
                Linear(classes, act='relu'),
                Softmax()])

        if index == 3:
            return Sequential([
                Convolution(output_depth=10, input_depth=1, batch_size=batch_size, input_dim=28,
                                           act='relu', stride_size=1, pad='VALID'),
                MaxPool(),
                Convolution(output_depth=25, stride_size=1, act='relu', pad='VALID'),
                MaxPool(),
                Convolution(kernel_size=4, output_depth=100, stride_size=1, act='relu', pad='VALID'),
                MaxPool(),
                Convolution(kernel_size=1, output_depth=10, stride_size=1, pad='VALID'),
                Softmax()])

        if index == 4:
            return Sequential([
                Convolution(output_depth=10, input_depth=1, batch_size=batch_size, input_dim=28, act='relu',
                            stride_size=1, pad='VALID'),
                MaxPool(pool_size=5),
                Convolution(output_depth=10, stride_size=2, pad='VALID'),
                Softmax()
            ])

