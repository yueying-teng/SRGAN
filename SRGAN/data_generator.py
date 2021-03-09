
import numpy as np
import math
import os
import random
import tensorflow as tf
import glob
import cv2
from random import seed
from SRGAN import utils
np.random.seed(2020)
seed(2020)


class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, img_path_list, crop_size, batch_size, scale=2, shuffle=True,
                crop=True, keep_dim=False, flip=False, normalize=True):
        """
        img_path_list: path that contains the images(original RGB images)
        keep_dim: if True, LR and HR images have the same spatial dimension
                  set to True for Res34Unet and SRCNN
                  False for EDSR
        scale: x2 enhancement
        """
        self.crop_size = crop_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.normalize = normalize
        # NOTE: images are not resized in this implementation. the valid np array 
        # output relies on using the randon_crop function to keep the dimension
        # consistent 
        self.crop = crop
        self.flip = flip
        self.scale = scale
        self.img_path_list = img_path_list 
        self.keep_dim = keep_dim

    def __len__(self):
        # number of batched per epoch
        return int(math.ceil(len(self.img_path_list) / float(self.batch_size)))

    def on_epoch_end(self):
        # shuffle the indices after each epoch during training
        self.index = range(len(self.img_path_list))
        if self.shuffle == True:
            self.index = random.sample(self.index, len(self.index))
    
    def make_border(self, img, top_pad=0, bottom_pad=0, right_pad=0, left_pad=0):
        white = [255,255,255]
        img = cv2.copyMakeBorder(img, top=top_pad, bottom=bottom_pad,
                                left=left_pad, right=right_pad,
                                borderType=cv2.BORDER_CONSTANT, value=white)
        return img

    def random_crop(self, lr_img, hr_img):
        lr_crop_size = self.crop_size // self.scale if not self.keep_dim else self.crop_size

        # if lr_img_shape.any() < crop size, pad both lr_img and hr_img first then crop
        if lr_img.shape[0] < lr_crop_size: # height too small
            lr_img = self.make_border(lr_img, top_pad=self.crop_size, bottom_pad=self.crop_size,
                                      right_pad=0, left_pad=0)
            hr_img = self.make_border(hr_img, top_pad=self.crop_size, bottom_pad=self.crop_size,
                                      right_pad=0, left_pad=0)

        if lr_img.shape[1] < lr_crop_size: # width too short
            lr_img = self.make_border(lr_img, top_pad=0, bottom_pad=0,
                                      right_pad=self.crop_size, left_pad=self.crop_size)
            hr_img = self.make_border(hr_img, top_pad=0, bottom_pad=0,
                                      right_pad=self.crop_size, left_pad=self.crop_size)

        lr_img_shape = lr_img.shape[:2]
        lr_w = np.random.randint(0, lr_img_shape[1] - lr_crop_size + 1)
        lr_h = np.random.randint(0, lr_img_shape[0] - lr_crop_size + 1)
        if self.keep_dim:
            hr_w = lr_w
            hr_h = lr_h
        else:
            hr_w = lr_w * self.scale
            hr_h = lr_h * self.scale

        lr_img_cropped = lr_img[lr_h:lr_h + lr_crop_size, lr_w:lr_w + lr_crop_size]
        hr_img_cropped = hr_img[hr_h:hr_h + self.crop_size, hr_w:hr_w + self.crop_size]

        return lr_img_cropped, hr_img_cropped

    def random_flip(self, lr_img, hr_img):
        n = random.randint(0,3)
        flipped_lr_img = np.rot90(lr_img, n)
        flipped_hr_img = np.rot90(hr_img, n)

        return flipped_lr_img, flipped_hr_img

    def __getitem__(self, idx):
        batch_hr_img =[]
        batch_lr_img = []
        batch_img_path = self.img_path_list[idx * self.batch_size: (idx + 1) * self.batch_size]
        for path in batch_img_path:
            # print(path)
            hr_img = cv2.imread(path)
            if hr_img is None:
                continue

            hr_img = utils.make_len_divisible(path)
            h, w, _ = hr_img.shape
            lr_img = cv2.resize(hr_img, (w // self.scale, h // self.scale)) # bilinear interpolation is the default

            if self.keep_dim:
                lr_img = cv2.resize(lr_img, (w, h))
            
            if self.flip:
                lr_img, hr_img = self.random_flip(lr_img, hr_img)

            if self.crop:
                lr_img, hr_img = self.random_crop(lr_img, hr_img)

            # print('after cropping ...')
            # if hr_img.shape != (self.crop_size, self.crop_size, 3) or lr_img.shape != (self.crop_size, self.crop_size, 3):
            #     print(cv2.imread(path).shape)
            #     print(hr_img.shape, lr_img.shape)
            batch_hr_img.append(hr_img)
            batch_lr_img.append(lr_img)

        if self.normalize:
            data, label = np.array(batch_lr_img).astype('float32')/255, np.array(batch_hr_img).astype('float32')/255
        else:
            # data, label = np.array(batch_lr_img).astype('float32'), np.array(batch_hr_img).astype('float32')
            data, label = np.array(batch_lr_img), np.array(batch_hr_img)
        
        return data, label

 