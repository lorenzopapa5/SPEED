from tensorflow.keras.layers import *
from tensorflow.keras import Model
from layers import SPP_Encoder, upsample_layer

import tensorflow as tf


def create_SPEED_model(input_shape):
    encoder = SPP_Encoder(input_shape=input_shape)

    encoder.summary()
    print('Base model loaded\n')
    print('Number of layers in the base model: {}\n'.format(len(encoder.layers)))

    # Starting point for decoder
    base_model_output_shape = encoder.layers[-1].output.shape
    decode_filters = 256

    # Decoder Layers
    decoder_0 = Conv2D(filters=decode_filters,
                       kernel_size=1,
                       padding='same',
                       input_shape=base_model_output_shape,
                       name='conv_Init_decoder')(encoder.output)

    decoder_1 = upsample_layer(decoder_0, int(decode_filters / 2), 'up1', concat_with='conv_dw_6', base_model=encoder)
    decoder_2 = upsample_layer(decoder_1, int(decode_filters / 4), 'up2', concat_with='conv_dw_4', base_model=encoder)
    decoder_3 = upsample_layer(decoder_2, int(decode_filters / 8), 'up3', concat_with='conv_dw_2', base_model=encoder)

    convDepthF = Conv2D(filters=1,
                        kernel_size=3,
                        padding='same',
                        name='convDepthF')(decoder_3)

    # Create the model
    model = Model(inputs=encoder.input, outputs=convDepthF)

    print('Model created\n')

    return model
