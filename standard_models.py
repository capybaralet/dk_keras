from keras.layers import Dense, Dropout, Activation
from keras.layers import Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam, SGD
from keras.utils import np_utils


def keras_cnn(dataset, num_filters=32, dropout=True):
    if dataset == 'cifar10':
        return keras_cifar10_cnn(num_filters, dropout)
    if dataset == 'mnist':
        return keras_mnist_cnn(num_filters, dropout)


def keras_mnist_cnn(num_filters=32, dropout=True):
    print num_filters
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    model = Sequential()

    model.add(Convolution2D(num_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(num_filters, kernel_size[0], kernel_size[1]))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=pool_size))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))
    return model


def keras_cifar10_cnn(num_filters=32, dropout=True):
    print num_filters
    model = Sequential()
    shape = (40000, 32, 32, 3)

    model.add(Convolution2D(num_filters, 3, 3, border_mode='same',
                            input_shape=shape[1:]))
    model.add(Activation('relu'))
    model.add(Convolution2D(num_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Convolution2D(2 * num_filters, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Convolution2D(2 * num_filters, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if dropout:
        model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    return model



