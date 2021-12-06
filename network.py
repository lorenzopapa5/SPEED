import sys
from tensorflow.keras.layers import *
from tensorflow.keras import Model, models
from layers import SPEED_Encoder, upsample_layer
from loss import accurate_obj_boundaries_loss


def create_SPEED_model(input_shape, existing=''):
    if len(existing) == 0:
        encoder = SPEED_Encoder(input_shape=input_shape)
        # encoder.summary()
        print('Number of layers in the encoder: {}'.format(len(encoder.layers)))

        # Starting point for decoder
        base_model_output_shape = encoder.layers[-1].output.shape
        decode_filters = 256

        # Decoder Layers
        decoder_0 = Conv2D(filters=decode_filters,
                           kernel_size=1,
                           padding='same',
                           input_shape=base_model_output_shape,
                           name='conv_Init_decoder')(encoder.output)

        decoder_1 = upsample_layer(decoder_0, int(decode_filters / 2), 'up1', concat_with='conv_dw_6',
                                   base_model=encoder)
        decoder_2 = upsample_layer(decoder_1, int(decode_filters / 4), 'up2', concat_with='conv_dw_4',
                                   base_model=encoder)
        decoder_3 = upsample_layer(decoder_2, int(decode_filters / 8), 'up3', concat_with='conv_dw_2',
                                   base_model=encoder)

        convDepthF = Conv2D(filters=1,
                            kernel_size=3,
                            padding='same',
                            name='convDepthF')(decoder_3)

        model = Model(inputs=encoder.input, outputs=convDepthF)
        print('Number of layers in the SPEED model: {}'.format(len(model.layers)))
        model.summary()

    else:
        if not existing.endswith('.h5'):
            sys.exit('Please provide a correct model file when using [existing] argument.')

        custom_objects = {'accurate_obj_boundaries_loss': accurate_obj_boundaries_loss}
        model = models.load_model(existing, custom_objects=custom_objects)

        for layer in model.layers:
            layer.trainable = True

        print('Number of layers in the SPEED model: {}'.format(len(model.layers)))
        print('Existing model loaded.\n')

    return model