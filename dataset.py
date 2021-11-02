import random, os
import numpy as np
import tensorflow as tf
from utils import resize_keeping_aspect_ratio

"""
## NYU
  * Indoor img (480, 640, 3) depth (480, 640, 1) both in png -> range between 0.5 to 10 meters
  * 654 Test and 7268 Train images (NYUv2_Labelled)  [Dataset Parts](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html#raw_parts)
  * 654 Test and 50688 Train images (NYUv1)
"""


class NYU2_DataLoader():

    def __init__(self, path, dts_type):
        self.dataset = path
        self.x = []
        self.y = []
        self.info = 0
        self.dts_type = dts_type

    def shuffle_dts(self):
        tmp_list = list(zip(self.x, self.y))
        random.shuffle(tmp_list)
        x_tmp, y_tmp = zip(*tmp_list)
        self.x = list(x_tmp)
        self.y = list(y_tmp)

        return self.x, self.y

    def get_dataset(self, size=0):
        if 'test' in self.dataset:
            elem = os.listdir(self.dataset)
            for el in elem:
                if 'colors' in str(el):
                    self.x.append(str(el))
                elif 'depth' in str(el):
                    self.y.append(str(el))
                else:
                    raise SystemError('Problem in get_dataset (test)')

        elif 'train' in self.dataset:
            scenarios = os.listdir(self.dataset)
            for scene in scenarios:
                elem = os.listdir(self.dataset + scene)
                for el in elem:
                    if 'jpg' in el:
                        self.x.append(scene + '/' + el)
                    elif 'png' in el:
                        self.y.append(scene + '/' + el)
                    else:
                        raise SystemError('Type image error (train)')
        else:
            raise SystemError('Problem in the path')

        if len(self.x) != len(self.y):
            raise SystemError('Problem with Img and Gt, no same size')

        self.x.sort()
        self.y.sort()

        if size != 0:
            self.x = self.x[:size]
            self.y = self.y[:size]

        self.info = len(self.x)

        return self.info

    def get_test_dataset(self):
        img_path = self.dataset + 'eigen_test_rgb.npy'
        depth_path = self.dataset + 'eigen_test_depth.npy'
        crop = self.dataset + 'eigen_test_crop.npy'

        rgb = np.load(img_path)
        depth = np.load(depth_path)
        crop = np.load(crop)

        self.x = rgb
        self.y = depth

        self.info = rgb.shape[0]

        return self.info

    def load_image(self, W_IMG_reduced_size=480, W_D_reduced_size=480, colors_image=False, index=None):
        if index is None:
            index = np.random.randint(0, self.info)

        # Load Image
        if self.dts_type == 'test':
            if colors_image:
                img = self.x[index]
            else:
                img = self.x[index]
                r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
                img = 0.2989 * r + 0.5870 * g + 0.1140 * b
                img = np.expand_dims(img, axis=-1)
        else:
            if colors_image:
                img = tf.io.read_file(self.dataset + self.x[index])
                img = tf.io.decode_png(img, channels=3)
            else:
                img = tf.io.read_file(self.dataset + self.x[index])
                img = tf.io.decode_png(img, channels=1)

        # Load Depth Image
        if self.dts_type == 'test':
            depth = np.expand_dims(self.y[index], axis=-1)
        else:
            depth = tf.io.read_file(self.dataset + self.y[index])
            depth = tf.io.decode_png(depth, channels=1)

        # Resize the image to a square image manteining the proportion
        img = resize_keeping_aspect_ratio(img, W_IMG_reduced_size)
        depth = resize_keeping_aspect_ratio(depth, W_D_reduced_size)

        if self.dts_type == 'test':
            depth = depth * 100
        else:
            depth /= 255
            depth = tf.clip_by_value(depth * 1000, 50, 1000)

        return img/255, depth, index