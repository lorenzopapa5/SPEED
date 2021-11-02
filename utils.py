import numpy as np
import matplotlib as plt
import tensorflow as tf
import keras as K


def plot_depth_map(dm):
    MIN_DEPTH = 0.0
    MAX_DEPTH = min(1000, np.percentile(dm, 99))

    dm = np.clip(dm, MIN_DEPTH, MAX_DEPTH)
    cmap = plt.cm.plasma_r  # jet
    cmap.set_bad(color='black')

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


def get_zipped_dataset(dataset_path_zipped, flag):
    if flag == True:
        from zipfile import ZipFile
        with ZipFile(dataset_path_zipped, 'r') as zip:
            print('Extracting all the files now...')
            zip.extractall()
            print('Done!\n')
        return True

    return False


def accuracy(y_true, y_pred, thr=0.05, types='percentual'):
    correct = K.maximum((y_true / y_pred), (y_pred / y_true)) < (1 + thr)

    return 100. * K.mean(correct)

