import numpy as np


def get_batch(bs_type, batch_size, img_shape, d_shape, colors, dts_train, dts_test, aug, dts=0):
    while True:
        # Initialize arrays that will contain the images
        input = np.zeros((batch_size, img_shape[0], img_shape[1], img_shape[2]))
        target = np.zeros((batch_size, d_shape[0], d_shape[1], d_shape[2]))

        if bs_type == 'train':
            dataset_Path = dts_train
        elif bs_type == 'valid':
            dataset_Path = dts_test
        else:
            raise ValueError('Incorrect dataset chosen.')

        for i in range(batch_size):
            img, depth_img, _ = dataset_Path[dts].load_image(img_shape[0], d_shape[0], colors)
            img, depth_img = aug(img, depth_img)
            input[i, :, :, :] = np.expand_dims(img, axis=0)
            target[i, :, :, :] = np.expand_dims(depth_img, axis=0)

        yield input, target


def get_batch_evolution(bs, num_test, dts_test, img_shape, d_shape, colour, dts=0):
    input = np.zeros((bs, img_shape[0], img_shape[1], img_shape[2]))
    target = np.zeros((bs, d_shape[0], d_shape[1], d_shape[2]))

    for i in range(bs):
        J = int(num_test - (num_test / 3 ** i))
        img, depth_img, _ = dts_test[dts].load_image(img_shape[0], d_shape[0], colour, J)
        input[i, :, :, :] = np.expand_dims(img, axis=0)
        target[i, :, :, :] = np.expand_dims(depth_img, axis=0)

    return input, target


def get_batch_evaluation(bs, num_test, dts_test, img_shape, d_shape, colour=True, dts=0, type=None, index=0):
    input = np.zeros((bs, img_shape[0], img_shape[1], img_shape[2]))
    target = np.zeros((bs, d_shape[0], d_shape[1], d_shape[2]))

    for i in range(bs):
        if type == 'sample_img':
            position = int((num_test - 1) / 15)
            img, depth_img, _ = dts_test[dts].load_image(img_shape[0], d_shape[0], colour, i * position)
        else:
            img, depth_img, _ = dts_test[dts].load_image(img_shape[0], d_shape[0], colour, index + i)
        input[i, :, :, :] = np.expand_dims(img, axis=0)
        target[i, :, :, :] = np.expand_dims(depth_img, axis=0)

    return input, target

