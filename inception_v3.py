"""
Inception V3 model for Keras.

Note that the input image format for this model is different than for
the VGG16 and ResNet models (299x299 instead of 224x224),
and that the input preprocessing function is also different (same as Xception).

# Reference

- [Rethinking the Inception Architecture for Computer Vision](
    http://arxiv.org/abs/1512.00567) (CVPR 2016)

COPYRIGHT

Copyright (c) 2016 - 2018, the respective contributors.
All rights reserved.

Each contributor holds copyright over their respective contributions.
The project versioning (Git) records all such contribution source information.
The initial code of this file came from https://github.com/keras-team/keras-applications/edit/master/keras_applications/inception_v3.py
(the Keras repository), hence, for author information regarding commits
that occured earlier than the first commit in the present repository,
please see the original Keras repository.

The original file from above link was modified. Modifications can be tracked via
git commits at https://github.com/joaopauloschuler/k-neural-api/blob/master/cai/inception_v3.py

LICENSE

The MIT License (MIT)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import layers
from layers import conv2d_bn
import utils
from tensorflow import keras
from tensorflow.keras.models import Model


from tensorflow.keras.layers import (Input, Conv2D, UpSampling2D,
                                     Concatenate, MaxPooling2D, Activation, Add, AveragePooling2D
                                     )

import tensorflow as tf



def custom_inception_block(x, filters, kType=0):
    x = kInceptionPointwise(x, filters=80, name='single_path3', kType=kType)
    x = conv2d_bn(x, filters, 3, 3, padding='valid', groups=1)  # 保持分组卷积

    x = conv2d_bn(x, filters, 3, 3, padding='same', groups=1)
    x = conv2d_bn(x, filters, 3, 3, padding='same', groups=1)

    shortcut = AveragePooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    shortcut = Conv2D(filters, kernel_size=(1, 1), strides=(1, 1), padding='valid', groups=1)(shortcut)
    shortcut = UpSampling2D(size=(2, 2))(shortcut)  # 调整上采样大小以匹配步幅

    x = Add()([shortcut, x])
    x = Activation('relu')(x)

    return x

def flexible_downsampling(x, target_shape):
    input_shape = tf.keras.backend.int_shape(x)[1:3]
    downsampling_factor = tf.divide(input_shape, target_shape)
    resized_x = tf.image.resize(x, size=target_shape, method=tf.image.ResizeMethod.BILINEAR)

    return resized_x


def interpolate(inputs, grid):
    grid = tf.cast(grid, tf.float32)
    ph = grid[:, :, 0]
    pw = grid[:, :, 1]
    a1 = gather_point(inputs, tf.math.ceil(ph), tf.math.ceil(pw))
    a2 = gather_point(inputs, tf.math.ceil(ph), tf.math.floor(pw))
    a3 = gather_point(inputs, tf.math.floor(ph), tf.math.ceil(pw))
    a4 = gather_point(inputs, tf.math.floor(ph), tf.math.floor(pw))
    return tf.reduce_sum([a1, a2, a3, a4], axis=-1)

def gather_point(inputs, ph, pw):
    a1 = bilinear(inputs, ph, pw, 0)
    a2 = bilinear(inputs, ph, pw, 1)
    a3 = bilinear(inputs, ph, pw, 2)
    a4 = bilinear(inputs, ph, pw, 3)
    return tf.reduce_max(tf.stack([a1, a2, a3, a4], axis=-1), axis=-1)  # Max pooling for simplicity

def bilinear(inputs, ph, pw, index):
    if index == 0:
        indices = tf.cast(tf.stack([ph, pw], axis=-1), tf.int32)
        return tf.gather(inputs, indices)
    elif index == 1:
        indices = tf.cast(tf.stack([ph, pw + 1], axis=-1), tf.int32)
        return tf.gather(inputs, indices)
    elif index == 2:
        indices = tf.cast(tf.stack([ph + 1, pw], axis=-1), tf.int32)
        return tf.gather(inputs, indices)
    elif index == 3:
        indices = tf.cast(tf.stack([ph + 1, pw + 1], axis=-1), tf.int32)
        return tf.gather(inputs, indices)


def InceptionV3(include_top=True,
                weights='imagenet',
                input_tensor=None,
                input_shape=None,
                pooling=None,
                classes=1000,
                **kwargs):

    img_input = keras.layers.Input(shape=input_shape)

    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    x = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                                strides=(1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed0')

    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                                strides=(1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')

    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                                strides=(1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    branch3x3 = conv2d_bn(x, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(
        branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = keras.layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed3')

    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                                strides=(1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed4')

    for i in range(2):
        branch1x1 = conv2d_bn(x, 192, 1, 1)

        branch7x7 = conv2d_bn(x, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

        branch7x7dbl = conv2d_bn(x, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

        branch_pool = keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))

    branch1x1 = conv2d_bn(x, 192, 1, 1)

    branch7x7 = conv2d_bn(x, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = keras.layers.AveragePooling2D((3, 3),
                                                strides=(1, 1),
                                                padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
    x = keras.layers.concatenate(
        [branch1x1, branch7x7, branch7x7dbl, branch_pool],
        axis=channel_axis,
        name='mixed7')

    branch3x3 = conv2d_bn(x, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(
        branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = keras.layers.concatenate(
        [branch3x3, branch7x7x3, branch_pool],
        axis=channel_axis,
        name='mixed8')

    for i in range(2):
        branch1x1 = conv2d_bn(x, 320, 1, 1)

        branch3x3 = conv2d_bn(x, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = keras.layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed9_' + str(i))

        branch3x3dbl = conv2d_bn(x, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = keras.layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = keras.layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)
        x = keras.layers.concatenate(
            [branch1x1, branch3x3, branch3x3dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(9 + i))
    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)

    inputs = img_input
    model = keras.models.Model(inputs, x, name='inception_v3')

    return model


def kInceptionPointwise(last_tensor, filters=32, channel_axis=3, name=None, activation='relu', has_batch_norm=True,
                        has_batch_scale=False, use_bias=False, kType=0):
    return layers.kPointwiseConv2D(last_tensor, filters=filters, channel_axis=channel_axis, name=name,
                                       activation=activation, has_batch_norm=has_batch_norm,
                                       has_batch_scale=has_batch_scale, use_bias=use_bias, kType=kType)


def create_inception_v3_mixed_layer(x, id, name='', channel_axis=3, bottleneck_compression=1, compression=1, kType=0):
    if id == 0:
        # mixed 0: 35 x 35 x 256
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression * 64), name=name + '_11a', kType=kType)
        # branch5x5 = conv2d_bn(x, int(bottleneck_compression*48), 1, 1, name=name + '_11b')
        branch5x5 = kInceptionPointwise(x, filters=int(bottleneck_compression * 48), name=name + '_11b', kType=kType)
        branch5x5 = conv2d_bn(branch5x5, int(compression * 64), 5, 5, name=name + '_55b')
        # branch3x3dbl = conv2d_bn(x, int(compression*64), 1, 1, name=name + '_11c')
        branch3x3dbl = kInceptionPointwise(x, filters=int(compression * 64), name=name + '_11c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression * 96), 3, 3, name=name + '_33c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression * 96), 3, 3, name=name + '_33cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*32), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression * 32), name=name + '_avg11',
                                          kType=kType)
        x = keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 1:
        # mixed 1: 35 x 35 x 288
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression * 64), name=name + '_11a', kType=kType)
        # branch5x5 = conv2d_bn(x, int(bottleneck_compression*48), 1, 1, name=name + '_11b')
        branch5x5 = kInceptionPointwise(x, filters=int(bottleneck_compression * 48), name=name + '_11b', kType=kType)
        branch5x5 = conv2d_bn(branch5x5, int(compression * 64), 5, 5, name=name + '_55b')
        # branch3x3dbl = conv2d_bn(x, int(compression*64), 1, 1, name=name + '_11c')
        branch3x3dbl = kInceptionPointwise(x, filters=int(compression * 64), name=name + '_11c', kType=kType)
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression * 96), 3, 3, name=name + '_33c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression * 96), 3, 3, name=name + '_33cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*64), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression * 64), name=name + '_avg11',
                                          kType=kType)
        x = keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 2:
        # mixed 2: 35 x 35 x 288
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression * 64), name=name + '_11a', kType=kType)
        # branch5x5 = conv2d_bn(x, int(bottleneck_compression*48), 1, 1, name=name + '_11b')
        branch5x5 = kInceptionPointwise(x, filters=int(bottleneck_compression * 48), name=name + '_11b', kType=kType)
        branch5x5 = conv2d_bn(branch5x5, int(compression * 64), 5, 5, name=name + '_55b')
        # branch3x3dbl = conv2d_bn(x, int(compression*64), 1, 1, name=name + '_11c')
        branch3x3dbl = kInceptionPointwise(x, filters=int(compression * 64), name=name + '_11c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression * 96), 3, 3, name=name + '_33c')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression * 96), 3, 3, name=name + '_33bb')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*64), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression * 64), name=name + '_avg11')
        x = keras.layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 3:
        # mixed 3: 17 x 17 x 768
        branch3x3 = conv2d_bn(x, int(bottleneck_compression * 384), 3, 3, strides=(2, 2), padding='valid',
                              name=name + '_33a')
        # branch3x3dbl = conv2d_bn(x, int(bottleneck_compression*64), 1, 1, name=name + '_11b')
        branch3x3dbl = kInceptionPointwise(x, filters=int(bottleneck_compression * 64), name=name + '_11b', kType=kType)
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression * 96), 3, 3, name=name + '_33b')
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression * 96), 3, 3, strides=(2, 2), padding='valid')
        branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name=name + '_max')(x)
        x = keras.layers.concatenate([branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name=name)

    if id == 4:
        # mixed 4: 17 x 17 x 768
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression * 192), name=name + '_11a', kType=kType)
        # branch7x7 = conv2d_bn(x, int(bottleneck_compression*128), 1, 1, name=name + '_11b')
        branch7x7 = kInceptionPointwise(x, filters=int(bottleneck_compression * 128), name=name + '_11b', kType=kType)
        branch7x7 = conv2d_bn(branch7x7, int(compression * 128), 1, 7, name=name + '_17b')
        branch7x7 = conv2d_bn(branch7x7, int(compression * 192), 7, 1, name=name + '_71b')
        # branch7x7dbl = conv2d_bn(x, int(bottleneck_compression*128), 1, 1, name=name + '_11c')
        branch7x7dbl = kInceptionPointwise(x, filters=int(bottleneck_compression * 128), name=name + '_11c',
                                           kType=kType)
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 128), 7, 1, name=name + '_71c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 128), 1, 7, name=name + '_17c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 128), 7, 1, name=name + '_71cc')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 192), 1, 7, name=name + '_17cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression * 192), name=name + '_avg11',
                                          kType=kType)
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name=name)

    if ((id == 5) or (id == 6)):
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression * 192), name=name + '_11a', kType=kType)
        # branch7x7 = conv2d_bn(x, int(bottleneck_compression*160), 1, 1, name=name + '_11b')
        branch7x7 = kInceptionPointwise(x, filters=int(bottleneck_compression * 160), name=name + '_11b', kType=kType)
        branch7x7 = conv2d_bn(branch7x7, int(compression * 160), 1, 7, name=name + '_17b')
        branch7x7 = conv2d_bn(branch7x7, int(compression * 192), 7, 1, name=name + '_71b')
        # branch7x7dbl = conv2d_bn(x, int(bottleneck_compression*160), 1, 1, name=name + '_11c')
        branch7x7dbl = kInceptionPointwise(x, filters=int(bottleneck_compression * 160), name=name + '_11c',
                                           kType=kType)
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 160), 7, 1, name=name + '_71c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 160), 1, 7, name=name + '_17c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 160), 7, 1, name=name + '_71cc')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 192), 1, 7, name=name + '_17cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression * 192), name=name + '_avg11',
                                          kType=kType)
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name=name)

    if id == 7:
        # mixed 7: 17 x 17 x 768
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression * 192), name=name + '_11a', kType=kType)
        # branch7x7 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11b')
        branch7x7 = kInceptionPointwise(x, filters=int(bottleneck_compression * 192), name=name + '_11b', kType=kType)
        branch7x7 = conv2d_bn(branch7x7, int(compression * 192), 1, 7, name=name + '_17b')
        branch7x7 = conv2d_bn(branch7x7, int(compression * 192), 7, 1, name=name + '_71b')
        # branch7x7dbl = conv2d_bn(x, int(compression*192), 1, 1, name=name + '_11c')
        branch7x7dbl = kInceptionPointwise(x, filters=int(bottleneck_compression * 192), name=name + '_11c',
                                           kType=kType)
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 192), 7, 1, name=name + '_71c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 192), 1, 7, name=name + '_17c')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 192), 7, 1, name=name + '_71cc')
        branch7x7dbl = conv2d_bn(branch7x7dbl, int(compression * 192), 1, 7, name=name + '_17cc')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression * 192), name=name + '_avg11',
                                          kType=kType)
        x = keras.layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=channel_axis, name=name)

    if id == 8:
        # mixed 8: 8 x 8 x 1280
        # branch3x3 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11a')
        branch3x3 = kInceptionPointwise(x, filters=int(bottleneck_compression * 192), name=name + '_11a', kType=kType)
        branch3x3 = conv2d_bn(branch3x3, int(compression * 320), 3, 3, strides=(2, 2), padding='valid',
                              name=name + '_33a')
        # branch7x7x3 = conv2d_bn(x, int(bottleneck_compression*192), 1, 1, name=name + '_11b')
        branch7x7x3 = kInceptionPointwise(x, filters=int(bottleneck_compression * 192), name=name + '_11b', kType=kType)
        branch7x7x3 = conv2d_bn(branch7x7x3, int(compression * 192), 1, 7, name=name + '_17b')
        branch7x7x3 = conv2d_bn(branch7x7x3, int(compression * 192), 7, 1, name=name + '_71b')
        branch7x7x3 = conv2d_bn(branch7x7x3, int(compression * 192), 3, 3, strides=(2, 2), padding='valid',
                                name=name + '_33b')
        branch_pool = keras.layers.MaxPooling2D((3, 3), strides=(2, 2), name=name + '_max')(x)
        x = keras.layers.concatenate([branch3x3, branch7x7x3, branch_pool], axis=channel_axis, name=name)

    if (id == 9) or (id == 10):
        # mixed 9: 8 x 8 x 2048
        # branch1x1 = conv2d_bn(x, int(bottleneck_compression*320), 1, 1, name=name + '_11')
        branch1x1 = kInceptionPointwise(x, filters=int(bottleneck_compression * 320), name=name + '_11', kType=kType)
        # branch3x3 = conv2d_bn(x, int(bottleneck_compression*384), 1, 1, name=name + '_11a')
        branch3x3 = kInceptionPointwise(x, filters=int(bottleneck_compression * 384), name=name + '_22', kType=kType)
        branch3x3_1 = conv2d_bn(branch3x3, int(compression * 384), 1, 3, name=name + '_11a')
        branch3x3_2 = conv2d_bn(branch3x3, int(compression * 384), 3, 1, name=name + '_31a')
        branch3x3 = keras.layers.concatenate([branch3x3_1, branch3x3_2], axis=channel_axis, name=name + '_pa')
        # branch3x3dbl = conv2d_bn(x, int(bottleneck_compression*448), 1, 1, name=name + '_11b')
        branch3x3dbl = kInceptionPointwise(x, filters=int(bottleneck_compression * 448), name=name + '_11b',
                                           kType=kType)
        branch3x3dbl = conv2d_bn(branch3x3dbl, int(compression * 384), 3, 3, name=name + '_33b')
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, int(compression * 384), 1, 3, name=name + '_13b')
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, int(compression * 384), 3, 1, name=name + '_31b')
        branch3x3dbl = keras.layers.concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis, name=name + '_pb')
        branch_pool = keras.layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same', name=name + '_avg')(x)
        # branch_pool = conv2d_bn(branch_pool, int(bottleneck_compression*192), 1, 1, name=name + '_avg11')
        branch_pool = kInceptionPointwise(branch_pool, filters=int(bottleneck_compression * 192), name=name + '_avg11',
                                          kType=kType)
        x = keras.layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=channel_axis, name=name)
    return x


def create_inception_path(last_tensor, compression=0.5, channel_axis=3, name=None, activation=None, has_batch_norm=True,
                          kType=0):
    output_tensor = last_tensor
    prev_layer_channel_count = keras.backend.int_shape(last_tensor)[channel_axis]
    channel_count = int(prev_layer_channel_count * compression)
    output_tensor = kInceptionPointwise(output_tensor, filters=channel_count, name=name, activation=activation,
                                        has_batch_norm=has_batch_norm, kType=kType)
    return output_tensor


def create_inception_v3_two_path_mixed_layer(x, id, name='', channel_axis=3, bottleneck_compression=0.5,
                                             compression=0.655, has_batch_norm=False, kType=0):
    if name == '':
        name = 'mixed'
    interleaved = layers.InterleaveChannels(2, name=name + '_interleaved')(x)
    a = create_inception_path(last_tensor=interleaved, compression=bottleneck_compression, channel_axis=channel_axis,
                              name=name + '_ta', activation=None, has_batch_norm=has_batch_norm, kType=kType)
    b = create_inception_path(last_tensor=interleaved, compression=bottleneck_compression, channel_axis=channel_axis,
                              name=name + '_tb', activation=None, has_batch_norm=has_batch_norm, kType=kType)
    a = create_inception_v3_mixed_layer(a, id=id, name=name + 'a', bottleneck_compression=bottleneck_compression,
                                        compression=compression, kType=kType)
    b = create_inception_v3_mixed_layer(b, id=id, name=name + 'b', bottleneck_compression=bottleneck_compression,
                                        compression=compression, kType=kType)
    return keras.layers.Concatenate(axis=channel_axis, name=name)([a, b])


def two_path_inception_v3(
        include_top=True,
        weights=None,  # 'two_paths_plant_leafs'
        input_shape=(224, 224, 3),
        pooling=None,
        classes=1000,
        two_paths_partial_first_block=0,
        two_paths_first_block=False,
        two_paths_second_block=False,
        deep_two_paths=False,
        deep_two_paths_compression=0.655,
        deep_two_paths_bottleneck_compression=0.5,
        l_ratio=0.5,
        ab_ratio=0.5,
        max_mix_idx=10,
        max_mix_deep_two_paths_idx=-1,
        model_name='two_path_inception_v3',
        kType=0,
        **kwargs):

    img_input = keras.layers.Input(shape=input_shape)
    if (deep_two_paths):  max_mix_deep_two_paths_idx = max_mix_idx

    if keras.backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    if two_paths_partial_first_block == 3:
        two_paths_partial_first_block = 0
        two_paths_first_block = True
        two_paths_second_block = False

    if two_paths_partial_first_block > 3:
        two_paths_partial_first_block = 0
        two_paths_first_block = True
        two_paths_second_block = True

    if (two_paths_second_block):
        two_paths_first_block = True

    include_first_block = True
    if (two_paths_partial_first_block == 1) or (two_paths_partial_first_block == 2):
        two_paths_second_block = False
        two_paths_first_block = False
        include_first_block = False

        if (two_paths_partial_first_block == 1):
            if (l_ratio > 0):
                l_branch = layers.CopyChannels(0, 1)(img_input)
                l_branch = conv2d_bn(l_branch, int(round(32 * l_ratio)), 3, 3, strides=(2, 2), padding='valid')

            if (ab_ratio > 0):
                ab_branch = layers.CopyChannels(1, 2)(img_input)
                ab_branch = conv2d_bn(ab_branch, int(round(32 * ab_ratio)), 3, 3, strides=(2, 2), padding='valid')

            if (l_ratio > 0):
                if (ab_ratio > 0):
                    single_branch = keras.layers.Concatenate(axis=channel_axis, name='concat_partial_first_block1')(
                        [l_branch, ab_branch])
                else:
                    single_branch = l_branch
            else:
                single_branch = ab_branch

            single_branch = conv2d_bn(single_branch, 32, 3, 3, padding='valid')
            single_branch = conv2d_bn(single_branch, 64, 3, 3)
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(single_branch)

        if (two_paths_partial_first_block == 2):
            if (l_ratio > 0):
                l_branch = layers.CopyChannels(0, 1)(img_input)
                l_branch = conv2d_bn(l_branch, int(round(32 * l_ratio)), 3, 3, strides=(2, 2), padding='valid')
                l_branch = conv2d_bn(l_branch, int(round(32 * l_ratio)), 3, 3, padding='valid')

            if (ab_ratio > 0):
                ab_branch = layers.CopyChannels(1, 2)(img_input)
                ab_branch = conv2d_bn(ab_branch, int(round(32 * ab_ratio)), 3, 3, strides=(2, 2), padding='valid')
                ab_branch = conv2d_bn(ab_branch, int(round(32 * ab_ratio)), 3, 3, padding='valid')

            if (l_ratio > 0):
                if (ab_ratio > 0):
                    single_branch = keras.layers.Concatenate(axis=channel_axis, name='concat_partial_first_block2')(
                        [l_branch, ab_branch])
                else:
                    single_branch = l_branch
            else:
                single_branch = ab_branch

            single_branch = conv2d_bn(single_branch, 64, 3, 3)
            x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(single_branch)

    if include_first_block:
        if two_paths_first_block:
            if (l_ratio > 0):
                l_branch = layers.CopyChannels(0, 1)(img_input)
                l_branch = conv2d_bn(l_branch, int(round(32 * l_ratio)), 3, 3, strides=(2, 2), padding='valid')
                l_branch = conv2d_bn(l_branch, int(round(32 * l_ratio)), 3, 3, padding='valid')
                l_branch = conv2d_bn(l_branch, int(round(64 * l_ratio)), 3, 3)
                l_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(l_branch)

            if (ab_ratio > 0):
                ab_branch = layers.CopyChannels(1, 2)(img_input)
                ab_branch = conv2d_bn(ab_branch, int(round(32 * ab_ratio)), 3, 3, strides=(2, 2), padding='valid')
                ab_branch = conv2d_bn(ab_branch, int(round(32 * ab_ratio)), 3, 3, padding='valid')
                ab_branch = conv2d_bn(ab_branch, int(round(64 * ab_ratio)), 3, 3)
                ab_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(ab_branch)

            if (l_ratio > 0):
                if (ab_ratio > 0):
                    x = keras.layers.Concatenate(axis=channel_axis, name='concat_first_block')(
                        [l_branch, ab_branch])
                else:
                    x = l_branch
            else:
                x = ab_branch
        else:
            single_branch = conv2d_bn(img_input, 32, 3, 3, strides=(2, 2), padding='valid')
            single_branch = conv2d_bn(single_branch, 32, 3, 3, padding='valid')
            single_branch = conv2d_bn(single_branch, 64, 3, 3)
            single_branch = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(single_branch)
            x = single_branch

    if (two_paths_second_block):
        # l_branch    = conv2d_bn(x, int(round(80*deep_two_paths_bottleneck_compression)), 1, 1, padding='valid', name='second_block_ta', activation=None, has_batch_norm=True)
        # ab_branch = conv2d_bn(x, int(round(80*deep_two_paths_bottleneck_compression)), 1, 1, padding='valid', name='second_block_tb', activation=None, has_batch_norm=True)
        l_branch = create_inception_path(last_tensor=x, compression=deep_two_paths_bottleneck_compression,
                                         channel_axis=channel_axis, name='second_block_ta', activation=None,
                                         has_batch_norm=True, kType=kType)
        ab_branch = create_inception_path(last_tensor=x, compression=deep_two_paths_bottleneck_compression,
                                          channel_axis=channel_axis, name='second_block_tb', activation=None,
                                          has_batch_norm=True, kType=kType)

        # l_branch    = conv2d_bn(l_branch,    int(round(80 *deep_two_paths_compression)), 1, 1, padding='valid')
        l_branch = kInceptionPointwise(l_branch, filters=int(round(80 * deep_two_paths_compression)),
                                       name='l_branch_path', kType=kType)
        l_branch = conv2d_bn(l_branch, int(round(192 * deep_two_paths_compression)), 3, 3, padding='valid')
        # ab_branch = conv2d_bn(ab_branch, int(round(80 *deep_two_paths_compression)), 1, 1, padding='valid')
        ab_branch = kInceptionPointwise(ab_branch, filters=int(round(80 * deep_two_paths_compression)),
                                        name='ab_branch_path', kType=kType)
        ab_branch = conv2d_bn(ab_branch, int(round(192 * deep_two_paths_compression)), 3, 3, padding='valid')

        x = keras.layers.Concatenate(axis=channel_axis, name='concat_second_block')(
            [l_branch, ab_branch])
        x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    else:
        x1 = kInceptionPointwise(x, filters=80, name='single_path', kType=kType)
        x1 = conv2d_bn(x1, 192, 3, 3, padding='valid')
        x1 = flexible_downsampling(x1, target_shape=(13, 13))

        x2 = kInceptionPointwise(x, filters=80, name='single_path2', kType=kType)
        x2 = custom_inception_block(x2, filters=192, kType=kType)
        x2 = flexible_downsampling(x2, target_shape=(13, 13))

        x = Add()([x1, x2])

    if max_mix_idx >= 0:
        for id_layer in range(max_mix_idx + 1):
            if (max_mix_deep_two_paths_idx >= id_layer):
                x = create_inception_v3_two_path_mixed_layer(x, id=id_layer, name='mixed' + str(id_layer),
                                                             channel_axis=channel_axis,
                                                             bottleneck_compression=deep_two_paths_bottleneck_compression,
                                                             compression=deep_two_paths_compression,
                                                             has_batch_norm=True, kType=kType)
            else:
                x = create_inception_v3_mixed_layer(x, id=id_layer, name='mixed' + str(id_layer),
                                                    channel_axis=channel_axis, kType=kType)

    if include_top:
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)  # glob.avg.pool
        x = keras.layers.Dense(classes, activation='softmax', name='predictions')(x)  # dense
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)

    inputs = img_input
    model = keras.models.Model(inputs, x, name=model_name)
    return model


def compiled_full_two_path_inception_v3(
        input_shape=(224, 224, 3),
        classes=1000,
        max_mix_idx=10,
        model_name='two_path_inception_v3',
        optimizer=None):

    return compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=0,
        two_paths_first_block=True,
        two_paths_second_block=True,
        deep_two_paths=True,
        deep_two_paths_compression=0.655,
        max_mix_idx=max_mix_idx,
        model_name='deep_two_path_inception_v3',
        optimizer=optimizer
    )


def compiled_inception_v3(
        input_shape=(224, 224, 3),
        classes=1000,
        max_mix_idx=10,
        model_name='two_path_inception_v3',
        optimizer=None):

    return compiled_two_path_inception_v3(
        input_shape=input_shape,
        classes=classes,
        two_paths_partial_first_block=0,
        two_paths_first_block=False,
        two_paths_second_block=False,
        deep_two_paths=False,
        max_mix_idx=max_mix_idx,
        model_name='two_path_inception_v3',
        optimizer=optimizer
    )


def compiled_two_path_inception_v3(
        input_shape=(224, 224, 3),
        classes=1000,
        two_paths_partial_first_block=0,
        two_paths_first_block=False,
        two_paths_second_block=False,
        deep_two_paths=False,
        deep_two_paths_compression=0.655,
        deep_two_paths_bottleneck_compression=0.5,
        l_ratio=0.5,
        ab_ratio=0.5,
        max_mix_idx=10,
        max_mix_deep_two_paths_idx=-1,
        model_name='two_path_inception_v3',
        optimizer=None
):

    base_model = two_path_inception_v3(
        include_top=False,
        weights=None,
        input_shape=input_shape,
        pooling=None,
        classes=classes,
        two_paths_partial_first_block=two_paths_partial_first_block,
        two_paths_first_block=two_paths_first_block,
        two_paths_second_block=two_paths_second_block,
        deep_two_paths=deep_two_paths,
        deep_two_paths_compression=deep_two_paths_compression,
        deep_two_paths_bottleneck_compression=deep_two_paths_bottleneck_compression,
        l_ratio=l_ratio,
        ab_ratio=ab_ratio,
        max_mix_idx=max_mix_idx,
        max_mix_deep_two_paths_idx=max_mix_deep_two_paths_idx,
        model_name=model_name
    )
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(classes, name='preprediction')(x)
    predictions = keras.layers.Activation('softmax', name='prediction')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    if optimizer is None:
        opt = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    else:
        opt = optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'])
    return model
