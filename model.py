#!/usr/bin/env python3
"""Implements AlexNet (Krizhevsky et al., 2012)."""

import PIL
import json
import math
import numpy
import pandas
import sys
from tensorflow import keras

_BATCH_SIZE = 64
_SIDE = 224

count = 0
model = None
X = None
Y = None


def load_image(filename, train=True):
    global count
    count += 1
    sys.stdout.write('\r[%d] Loading image %s...' % (count, filename))
    sys.stdout.flush()
    dirname = 'train' if train else 'test'
    with PIL.Image.open('./data/%s/%s' % (dirname, filename)) as image:
        rgb = PIL.Image.new('RGBA', image.size)
        rgb.paste(image)
        width, height = image.size
        rgb.thumbnail(
            (_SIDE, math.floor(_SIDE * height / width)) if width > height else (math.floor(_SIDE * width / height), _SIDE))
        return numpy.array(rgb)


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

    # Layer 6.
    model.add(keras.layers.Flatten())
    model.add(
        keras.layers.Dense(4096, 'relu',
            input_shape=(height * width * 3,)))
    model.add(keras.layers.BatchNormalization())

    # Layer 7.
    model.add(keras.layers.Dense(4096, 'relu'))
    model.add(keras.layers.BatchNormalization())

    # Layer 8.
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(classes, 'softmax'))
    #return keras.utils.multi_gpu_model(model, gpus=2)
    return model

class Sequence(keras.utils.Sequence):
    def __init__(self, X, Y, height, width, class2idx, classes):
        self._X = X
        self._Y = Y
        self._height = height
        self._width = width
        self._class2idx = class2idx
        self._classes = classes

    def __getitem__(self, index):
        start = index * _BATCH_SIZE
        end = start + _BATCH_SIZE
        return self._transform(self._X[start:end], self._Y[start:end])

    def __len__(self):
        return int(numpy.floor(self._Y.size / _BATCH_SIZE))

    def _transform(self, X, Y):
        Xt = []
        Yt = []
        for idx, x in enumerate(X):
            Xt.append(self._transform_color(x))
            Yt.append(keras.utils.to_categorical(self._class2idx[Y[idx]], len(self._classes)))
        return (numpy.array(Xt), numpy.array(Yt))

    def _transform_color(self, x):
        result = numpy.zeros((self._height, self._width, 3))
        for height in range(0, x.shape[0]):
            for width in range(0, x.shape[1]):
                for rgb in range(0, 3):
                    result[height, width, rgb] = x[height, width, rgb]
        return result


class Model:

    def __init__(self, X, Y):
        self._color = None
        self._classes = set(Y)
        self._idx2class = {i: x for i, x in enumerate(self._classes)}
        self._class2idx = {x: i for i, x in self._idx2class.items()}
        self._max_3d_width = self._get_max_dimension(X, 3, 1)
        self._max_3d_height = self._get_max_dimension(X, 3, 0)

    def fit(self, X, Y):
        self._color = create_alexnet(self._max_3d_height, self._max_3d_width, len(self._classes))
        self._color.summary()
        self._color.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.SGD(momentum=0.9, decay=0.0005, lr=0.01),
            metrics=['accuracy', 'top_k_categorical_accuracy'])
        split = math.floor(0.05 * len(Y))
        X_train = X[split:]
        Y_train = Y[split:]
        X_validation = X[:split]
        Y_validation = Y[:split]
        generator = Sequence(X_train, Y_train, self._max_3d_height, self._max_3d_width, self._class2idx, self._classes)
        validation = Sequence(X_validation, Y_validation, self._max_3d_height, self._max_3d_width, self._class2idx, self._classes)
        self._color.fit_generator(generator, epochs=10, validation_data=validation, shuffle=True, use_multiprocessing=True, workers=2)

    def _get_max_dimension(self, X, ndim, aspect):
        result = 0
        for x in X:
            if x.ndim != ndim:
                continue
            result = max(result, x.shape[aspect])
        return result

    def save(self):
        self._color.save('model.h5')
        with open('class2idx.json', 'w') as writer:
            writer.write(json.dumps(self._class2idx))

    def __str__(self):
        return json.dumps({
            'max_3d_width': self._max_3d_width,
            'max_3d_height': self._max_3d_height,
        })


def main():
    global model, X, Y
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
