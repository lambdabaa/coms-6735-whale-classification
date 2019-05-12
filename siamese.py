# !/usr/bin/env python3
"""Implements Siamese Neural Networks for One-shot Image Recognition (Koch et al., 2016)."""

import PIL
import json
import math
import numpy
import pandas
import random
import sys
from tensorflow import keras

_BATCH_SIZE = 32
_SIDE = 224

model = None
count = 0
X = None
Y = None

def get_abs_diff(vects):
    x, y = vects
    return keras.backend.abs(x - y)

def make_conv_layer(filters, kernel_size, strides=(1, 1)):
    return keras.layers.Conv2D(
        activation='relu',
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding='valid')

def make_pool_layer():
    return keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')

def create_twin():
    start = keras.layers.Input((_SIDE, _SIDE, 3))
    conv1 = make_conv_layer(96, (11, 11), (4, 4))(start)
    norm1 = keras.layers.BatchNormalization()(conv1)
    pool1 = make_pool_layer()(norm1)
    conv2 = make_conv_layer(256, (5, 5))(pool1)
    norm2 = keras.layers.BatchNormalization()(conv2)
    pool2 = make_pool_layer()(norm2)
    conv3 = make_conv_layer(384, (3, 3))(pool2)
    norm3 = keras.layers.BatchNormalization()(conv3)
    conv4 = make_conv_layer(384, (3, 3))(norm3)
    norm4 = keras.layers.BatchNormalization()(conv4)
    conv5 = make_conv_layer(256, (3, 3))(conv4)
    norm5 = keras.layers.BatchNormalization()(conv5)
    pool5 = make_pool_layer()(norm5)
    flatten = keras.layers.Flatten()(pool5)
    dense1 = keras.layers.BatchNormalization()(
        keras.layers.Dense(4096, activation='relu')(flatten))
    dense2 = keras.layers.BatchNormalization()(
        keras.layers.Dense(4096, activation='relu')(dense1))
    return (start, dense2)

def create_siamese_network():
    input1, twin1 = create_twin()
    input2, twin2 = create_twin()
    dist = keras.layers.Lambda(get_abs_diff)([twin1, twin2])
    dropout = keras.layers.Dropout(0.5)(dist)
    output = keras.layers.Dense(1, activation='sigmoid')(dropout)
    return keras.utils.multi_gpu_model(keras.models.Model(inputs=[input1, input2], outputs=output), gpus=4)

def pairs(X, Y):
    positive = 0
    negative = 0
    total = len(X) ** 2
    shuffle = list(range(0, len(X)))
    Xp = []
    Yp = []
    for idx1, x1 in enumerate(X):
        random.shuffle(shuffle)
        for idx2 in shuffle:
            x2 = X[idx2]
            if idx1 == idx2:
                continue
            if Y[idx1] == Y[idx2]:
                Xp.append((x1, x2))
                Yp.append(1)
                positive += 1
            elif negative <= positive:
                Xp.append((x1, x2))
                Yp.append(0)
                negative += 1
        sys.stdout.write('\rCreating siamese example %d / %d...' % (positive + negative, total))
        sys.stdout.flush()
    return (Xp, Yp)

class Sequence(keras.utils.Sequence):
    def __init__(self, X, Y):
        self._X = X
        self._Y = Y

    def on_epoch_end(self):
        Xs, Ys = shuffle_unison(self._X, self._Y)
        self._X = Xs
        self._Y = Ys

    def __getitem__(self, idx):
        """Expands a single batch."""
        start = idx * _BATCH_SIZE
        end = start + _BATCH_SIZE
        return self._transform(self._X[start:end], self._Y[start:end])

    def _transform(self, X, Y):
        Xt1 = []
        Xt2 = []
        for x in X:
            Xt1.append(self._transform_image(x[0]))
            Xt2.append(self._transform_image(x[1]))
        return ([numpy.array(Xt1), numpy.array(Xt2)], numpy.array(Y))

    def _transform_image(self, x):
        result = numpy.zeros((_SIDE, _SIDE, 3))
        for height in range(0, x.shape[0]):
            for width in range(0, x.shape[1]):
                for rgb in range(0, 3):
                    result[height, width, rgb] = x[height, width, rgb]
        return result

    def __len__(self):
        return int(numpy.floor(len(self._Y) / _BATCH_SIZE))

class Model:
    def fit(self, X, Y):
        self._model = create_siamese_network()
        self._model.summary()
        self._model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        split = math.floor(0.1 * len(Y))
        X_train = X[split:]
        Y_train = Y[split:]
        X_validation = X[:split]
        Y_validation = Y[:split]
        generator = Sequence(X_train, Y_train)
        validation = Sequence(X_validation, Y_validation)
        self._model.fit_generator(generator, epochs=20, validation_data=validation, use_multiprocessing=True, workers=64, max_queue_size=64)

    def save(self):
        self._model.save('model.h5')

    def load(self):
        self._model = keras.models.load_model('model.h5')

    def predict(self, x):
        return self._model.predict(x)

def load_image(filename, train=True):
    global count
    count += 1
    sys.stdout.write('\r[%d] Loading image %s...' % (count, filename))
    sys.stdout.flush()
    dirname = 'train' if train else 'test'
    with PIL.Image.open('./data/%s/%s' % (dirname, filename)) as image:
        image.thumbnail((_SIDE, _SIDE))
        rgb = PIL.Image.new('RGB', (_SIDE, _SIDE), (255, 255, 255))
        rgb.paste(image)
        return numpy.array(rgb)

def shuffle_unison(X, Y):
    shuffle = list(range(0, len(X)))
    random.shuffle(shuffle)
    Xt = []
    Yt = []
    for idx in shuffle:
        Xt.append(X[idx])
        Yt.append(Y[idx])
    return (Xt, Yt)

def main():
    global model, X, Y
    dataset = pandas.read_csv('./data/train.csv')
    X0 = [load_image(x) for x in dataset['Image'].values]
    Y0 = dataset['Id'].values.tolist()
    X = []
    Y = []
    for idx, y in enumerate(Y0):
        if y != 'new_whale':
            X.append(X0[idx])
            Y.append(y)
    print()
    # Shuffle X and Y.
    X, Y = shuffle_unison(X, Y)
    X, Y = pairs(X, Y)
    model = Model()
    model.fit(X, Y)
    model.save()

if __name__ == '__main__':
    main()
