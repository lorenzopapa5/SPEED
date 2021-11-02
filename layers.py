from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras import Model
from tensorflow.python.keras import backend

import tensorflow as tf


def pyramid_pooling_block(input_tensor, bin_sizes, w, h, filters):
    concat_list = [input_tensor]

    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(w // bin_size, h // bin_size),
                                             strides=(w // bin_size, h // bin_size))(input_tensor)
        x = tf.keras.layers.SeparableConv2D(filters, 3, strides=1, padding='same')(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w, h)))(x)

        concat_list.append(x)

    return tf.keras.layers.concatenate(concat_list)


def my_pyramid_pooling_block(input_tensor, bin_sizes, w, h, filters, name):
    concat_list = []

    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(w // bin_size, h // bin_size), strides=(w // bin_size, h // bin_size),
                             name=name + '_avgpool_' + str(bin_size))(input_tensor)
        x = SeparableConv2D(filters=filters // 4, kernel_size=3, padding='same',
                            name=name + '_upconv_1_' + str(bin_size), use_bias=False)(x)
        x = tf.keras.layers.Lambda(lambda x: tf.image.resize(x, (w, h)))(x)

        concat_list.append(x)

    print(bin_sizes)

    x_even = Concatenate()([concat_list[0], concat_list[2]])
    x_even = ReLU()(x_even)
    x_even = SeparableConv2D(filters=filters // 2, kernel_size=3, padding='same', name=name + '_upconv_2_odd',
                             use_bias=False)(x_even)
    x_even = ReLU()(x_even)

    x_odd = Concatenate()([concat_list[1], concat_list[3]])
    x_odd = ReLU()(x_odd)
    x_odd = SeparableConv2D(filters=filters // 2, kernel_size=3, padding='same', name=name + '_upconv_2_even',
                            use_bias=False)(x_odd)
    x_odd = ReLU()(x_odd)

    x = Concatenate()([x_even, x_odd, input_tensor])

    return x


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)

    x = layers.Conv2D(filters,
                      kernel,
                      padding='same',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(inputs)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    x = layers.ReLU(6., name='conv1_relu')(x)

    return x


def _depthwise_conv_block(inputs,
                          pointwise_conv_filters,
                          alpha,
                          depth_multiplier=1,
                          strides=(1, 1),
                          block_id=1):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((0, 1), (0, 1)), name='conv_pad_%d' % block_id)(inputs)

    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv_pw_%d_bn' % block_id)(x)
    x = layers.ReLU(6., name='conv_pw_%d_relu' % block_id)(x)

    return x


def SPP_Encoder(input_shape, alpha=1.0, depth_multiplier=1):
    img_input = layers.Input(shape=input_shape)

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(
        x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(
        x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(
        x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)

    x = pyramid_pooling_block(x, [2, 4, 6, 8], x.shape[1], x.shape[2])
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    # x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=13)  # 1024

    # Create model.
    model = Model(img_input, x, name='SPP_encoder')

    return model


def upsample_layer(tensor, filters, name, concat_with, base_model):
    def HPO2(filters_value):
        for i in range(filters_value, 0, -1):
            if ((i & (i - 1)) == 0):
                return i

    if name == 'up1':
        up_i = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', dilation_rate=1,
                               name=name + '_upconv', use_bias=False)(tensor)
    else:
        up_i = my_pyramid_pooling_block(input_tensor=tensor, bin_sizes=[2, 4, 6, 8], w=tensor.shape[1],
                                        h=tensor.shape[2], filters=filters, name=name)
        up_i = Conv2DTranspose(filters=filters, kernel_size=3, strides=2, padding='same', dilation_rate=1,
                               name=name + '_upconv_final', use_bias=False)(up_i)

    up_i = Concatenate(name=name + '_concat')([up_i, base_model.get_layer(concat_with).output])  # Skip connection
    up_i = ReLU()(up_i)

    up_i = SeparableConv2D(filters=HPO2((up_i.shape[-1]) // 4),
                           kernel_size=3,
                           padding='same',
                           use_bias=False,
                           name=name + '_sep_conv')(up_i)
    up_i = ReLU()(up_i)

    return up_i
