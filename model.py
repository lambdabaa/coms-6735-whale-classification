#!/usr/bin/env python3
import PIL
import json
import numpy
import pandas
import sys
from tensorflow import keras

count = 0


def load_image(filename, train=True):
    global count
    count += 1
    sys.stdout.write('\r[%d] Loading image %s...' % (count, filename))
    sys.stdout.flush()
    dirname = 'train' if train else 'test'
    with PIL.Image.open('./data/%s/%s' % (dirname, filename)) as image:
        return numpy.array(image)


def create_alexnet(height, width, classes):
    model = keras.models.Sequential()
    # Layer 1.
    model.add(
        keras.layers.Conv2D(
            activation='relu',
            filters=96,
            input_shape=(height, width, 3),
            kernel_size=(11, 11),
            strides=(4, 4),
            padding='valid'))
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Layer 2.
    model.add(
        keras.layers.Conv2D(
            activation='relu',
            filters=256,
            kernel_size=(5, 5),
            strides=(1, 1),
            padding='valid'))
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(keras.layers.BatchNormalization())

    # Layer 3.
    model.add(
        keras.layers.Conv2D(
            activation='relu',
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid'))
    model.add(keras.layers.BatchNormalization())

    # Layer 4.
    model.add(
        keras.layers.Conv2D(
            activation='relu',
            filters=384,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid'))
    model.add(keras.layers.BatchNormalization())

    # Layer 5.
    model.add(
        keras.layers.Conv2D(
            activation='relu',
            filters=256,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid'))
    model.add(
        keras.layers.MaxPooling2D(
            pool_size=(2, 2), strides=(2, 2), padding='valid'))
    model.add(keras.layers.BatchNormalization())

    # Layer 6.
    model.add(keras.layers.Flatten())
    model.add(
        keras.layers.Dense(4096, 'relu',
            input_shape=(height * width * 3,)))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.BatchNormalization())

    # Layer 7.
    model.add(keras.layers.Dense(4096, 'relu'))
    model.add(keras.layers.Dropout(0.4))
    model.add(keras.layers.BatchNormalization())

    # Layer 8.
    model.add(keras.layers.Dense(classes, 'softmax'))
    return model


class Model:

    def __init__(self, X, Y):
        self._color = None
        self._grayscale = None
        self._classes = set(Y)
        self._idx2class = {i: x for i, x in enumerate(self._classes)}
        self._class2idx = {x: i for i, x in self._idx2class.items()}
        self._max_2d_width = self._get_max_dimension(X, 2, 1)
        self._max_2d_height = self._get_max_dimension(X, 2, 0)
        self._max_3d_width = self._get_max_dimension(X, 3, 1)
        self._max_3d_height = self._get_max_dimension(X, 3, 0)


    def transform(self, X, Y):
        X_t = []
        print('Padding training images...')
        for x in X:
            X_t.append(self._transform_grayscale(x) if x.ndim == 2
                       else self._transform_color(x))
        Y_t = [keras.utils.to_categorical(
            self._class2idx[y], len(self._classes)) for y in Y]
        return (X_t, Y_t)

    def fit(self, X, Y):
        X_grayscale, Y_grayscale, X_color, Y_color = self._split_color_grayscale(X, Y)
        self._fit_grayscale(X_grayscale, Y_grayscale)
        self._fit_color(X_color, Y_color)

    def _fit_grayscale(self, X, Y):
        self._grayscale = create_alexnet(self._max_2d_height, self._max_2d_width, len(self._classes))
        self._grayscale.summary()
        self._grayscale.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'top_k_categorical_accuracy'])
        self._color.fit(
            X, Y, batch_size=64, epochs=1, validation_split=0.1, shuffle=True)

    def _fit_color(self, X, Y):
        self._color = create_alexnet(self._max_3d_height, self._max_3d_width, len(self._classes))
        self._color.summary()
        self._color.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy', 'top_k_categorical_accuracy'])
        self._color.fit(
            X, Y, batch_size=64, epochs=1, validation_split=0.1, shuffle=True)

    def _transform_grayscale(self, x):
        result = numpy.zeros((self._max_2d_height, self._max_2d_width, 3))
        for height in range(0, x.shape[0]):
            for width in range(0, x.shape[1]):
                result[height, width, 0] = x[height, width]
                result[height, width, 1] = x[height, width]
                result[height, width, 2] = x[height, width]
        return result

    def _transform_color(self, x):
        result = numpy.zeros((self._max_3d_height, self._max_3d_width, 3))
        for height in range(0, x.shape[0]):
            for width in range(0, x.shape[1]):
                for rgb in range(0, 3):
                    result[height, width, rgb] = x[height, width, rgb]
        return result

    def _get_max_dimension(self, X, ndim, aspect):
        result = 0
        for x in X:
            if x.ndim != ndim:
                continue
            result = max(result, x.shape[aspect])
        return result

    def _split_color_grayscale(self, X, Y):
        X_grayscale = []
        Y_grayscale = []
        X_color = []
        Y_color = []
        for idx, x in enumerate(X):
            if x.ndim == 2:
                X_grayscale.append(x)
                Y_grayscale.append(Y[idx])
            elif x.ndim == 3:
                X_color.append(x)
                Y_color.append(Y[idx])
        return (
            numpy.array(X_grayscale),
            numpy.array(Y_grayscale),
            numpy.array(X_color),
            numpy.array(Y_color))


    def __str__(self):
        return json.dumps({
            'max_2d_width': self._max_2d_width,
            'max_2d_height': self._max_2d_height,
            'max_3d_width': self._max_3d_width,
            'max_3d_height': self._max_3d_height,
        })

def main():
    dataset = pandas.read_csv('./data/train.csv')
    X = [load_image(x) for x in dataset['Image'].values]
    Y = dataset['Id'].values
    print()
    model = Model(X, Y)
    print(str(model))
    X, Y = model.transform(X, Y)
    model.fit(X, Y)

if __name__ == '__main__':
    main()
