import os
import random
import numpy as np
import tensorflow as tf
from PIL import Image

from utils import resize_keeping_aspect_ratio


class NYU2_DataLoader:

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
        img_path = self.dataset + 'test_rgb.npy'
        depth_path = self.dataset + 'test_depth.npy'

        rgb = np.load(img_path)
        depth = np.load(depth_path)

        self.x = rgb
        self.y = depth

        self.info = rgb.shape[0]

        return self.info

    def load_image(self, W_IMG_reduced_size=480, W_D_reduced_size=480, colors_image=False, index=None):
        if index is None:
            index = np.random.randint(0, self.info)

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

        if self.dts_type == 'test':
            depth = np.expand_dims(self.y[index], axis=-1)
        else:
            depth = tf.io.read_file(self.dataset + self.y[index])
            depth = tf.io.decode_png(depth, channels=1)

        img = resize_keeping_aspect_ratio(img, W_IMG_reduced_size)
        depth = resize_keeping_aspect_ratio(depth, W_D_reduced_size)

        if self.dts_type == 'test':
            depth = depth * 100
        else:
            depth /= 255
            depth = tf.clip_by_value(depth * 1000, 50, 1000)

        return img/256, depth, index


class DIML_DataLoader:

    def __init__(self, path, type_dts):
        self.dataset = path
        self.type_dts = type_dts
        self.x = []
        self.y = []
        self.info = 0

    def shuffle_dts(self):
        tmp_list = list(zip(self.x, self.y))
        random.shuffle(tmp_list)
        x_tmp, y_tmp = zip(*tmp_list)
        self.x = list(x_tmp)
        self.y = list(y_tmp)

        return self.x, self.y

    def get_train_dataset(self):
        skipped_img = 0
        folders = os.listdir(self.dataset)
        for folder in folders:
            scenarios = os.listdir(self.dataset + folder)
            for scene in scenarios:
                color_img = os.listdir(self.dataset + folder + '/' + scene + '/col/')
                depth_img = os.listdir(self.dataset + folder + '/' + scene + '/up_png/')
                depth_img.sort()
                color_img.sort()
                for col, dep in zip(color_img, depth_img):
                    if col.split('_')[2:5] != dep.split('_')[2:5]:
                        skipped_img += 1
                        pass
                    else:
                        self.x.append(folder + '/' + scene + '/col/' + col)
                        self.y.append(folder + '/' + scene + '/up_png/' + dep)

        self.x.sort()
        self.y.sort()

        self.info = len(self.x)

        return self.info, skipped_img

    def get_test_dataset(self):
        skipped_img = 0
        folders = os.listdir(self.dataset)
        rgb_img = os.listdir(self.dataset + folders[0])
        d_img = os.listdir(self.dataset + folders[1])
        rgb_img.sort()
        d_img.sort()
        for col, dep in zip(rgb_img, d_img):
            if col.split('_')[2:5] != dep.split('_')[2:5]:
                skipped_img += 1
                pass
            else:
                self.x.append(folders[0] + '/' + col)
                self.y.append(folders[1] + '/' + dep)

        self.x.sort()
        self.y.sort()

        self.info = len(self.x)

        return self.info, skipped_img

    def load_image(self, W_IMG_reduced_size=792, W_D_reduced_size=792, colors_image=True, index=None):
        if index is None:
            index = np.random.randint(0, self.info)

        depth = Image.open(self.dataset + self.y[index])
        depth = np.expand_dims(np.array(depth), axis=-1)

        if colors_image:
            img = Image.open(self.dataset + self.x[index]).convert('RGB')
            img = np.array(img)
        else:
            img = tf.io.read_file(self.dataset + self.x[index])
            img = tf.io.decode_png(img, channels=1)

        img = resize_keeping_aspect_ratio(img, W_IMG_reduced_size)
        depth = resize_keeping_aspect_ratio(depth, W_D_reduced_size)

        img = img / 256
        depth = depth / 10

        return img, depth, index
