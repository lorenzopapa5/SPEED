import numpy as np
import tensorflow as tf
import keras.backend as K
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_depth_map(dm):
    MIN_DEPTH = 0.0
    MAX_DEPTH = min(1000, np.percentile(dm, 99))

    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    cmap = plt.cm.plasma_r

    return dm, cmap, MIN_DEPTH, MAX_DEPTH


def resize_keeping_aspect_ratio(img, base, interpolation='area'):  # bilinear di default
    '''
    Resize the image to a defined length manteining its proportions
    Scaling the shortest side of the image to a fixed 'base' length'
    '''

    if img.shape[0] <= img.shape[1]:
        basewidth = int(base)
        wpercent = (basewidth / float(img.shape[0]))
        hsize = int((float(img.shape[1]) * float(wpercent)))
        img = tf.image.resize(img, [basewidth, hsize], method=interpolation, antialias=True)
    else:
        baseheight = int(base)
        wpercent = (baseheight / float(img.shape[1]))
        wsize = int((float(img.shape[0]) * float(wpercent)))
        img = tf.image.resize(img, [wsize, baseheight], method=interpolation, antialias=True)

    return np.array(img)


def accuracy(y_true, y_pred, thr=0.05):
    correct = K.maximum((y_true / y_pred), (y_pred / y_true)) < (1 + thr)

    return 100. * K.mean(correct)


def print_img(dataset, img_h=480, d_h=480, augment=False):
    img, depth, index = dataset.load_image(img_h, d_h, colors_image=True)
    print('Depth {} -> Shape = {}, max = {}, min = {}, mean = {}'.format(index, depth.shape, np.max(depth),
                                                                         np.min(depth), np.mean(depth)))
    print('IMG {} -> Shape = {}, max = {}, min = {}, mean = {}\n'.format(index, img.shape, np.max(img), np.min(img),
                                                                         np.mean(img)))

    if augment:
        # TO DO
        pass

    fig = plt.figure(figsize=(15, 2))
    plt.subplot(1, 3, 1)
    plt.title('Original image')
    plt.imshow(tf.squeeze(img), cmap='gray', vmin=0.0, vmax=1.0)
    if False:
        plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title('Grayscale DepthMap')
    plt.imshow(tf.squeeze(depth), cmap='gray')
    plt.colorbar()
    if False:
        plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title('Colored DepthMap')
    depth, cmap_dm, vmin, vmax = plot_depth_map(depth)
    plt.imshow(tf.squeeze(depth), cmap=cmap_dm, vmin=vmin, vmax=vmax)
    plt.colorbar()
    if False:
        plt.axis('off')

    plt.show()
