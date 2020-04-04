# ------------------------------------------------------------------------------
# --coding='utf-8'--
# Written by czifan (czifan@pku.edu.cn)
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from torch.utils.data import Dataset
import gzip
import random
import torch

class MovingMNISTDataset(Dataset):
    def __init__(self, config, split='train'):
        super().__init__()
        with gzip.open(config.data_file, 'rb') as f:
            self.datas = np.frombuffer(f.read(), np.uint8, offset=16)
            self.datas = self.datas.reshape(-1, *config.image_size)
        self.datas = self.datas[eval('config.{}_dataset[0]'.format(split)):eval('config.{}_dataset[1]'.format(split))]
        self.image_size = config.image_size
        self.input_size = config.input_size
        self.step_length = config.step_length
        self.num_objects = config.num_objects
        self.num_frames_input = config.num_frames_input
        self.num_frames_output = config.num_frames_output
        self.num_frames = config.num_frames_input + config.num_frames_output
        print('Loaded {} samples ({})'.format(self.__len__(), split))

    def _get_random_trajectory(self, seq_length):
        assert self.input_size[0] == self.input_size[1]
        assert self.image_size[0] == self.image_size[1]
        canvas_size = self.input_size[0] - self.image_size[0]
        x = random.random()
        y = random.random()
        theta = random.random() * 2 * np.pi
        v_y = np.sin(theta)
        v_x = np.cos(theta)

        start_y = np.zeros(seq_length)
        start_x = np.zeros(seq_length)
        for i in range(seq_length):
            y += v_y * self.step_length
            x += v_x * self.step_length

            if x <= 0.: x=0.; v_x=-v_x;
            if x >= 1.: x=1.; v_x=-v_x
            if y <= 0.: y=0.; v_y=-v_y;
            if y >= 1.: y=1.; v_y=-v_y

            start_y[i] = y
            start_x[i] = x

        start_y = (canvas_size * start_y).astype(np.int32)
        start_x = (canvas_size * start_x).astype(np.int32)

        return start_y, start_x

    def _generate_moving_mnist(self, num_digits=2):
        data = np.zeros((self.num_frames, *self.input_size), dtype=np.float32)
        for n in range(num_digits):
            start_y, start_x = self._get_random_trajectory(self.num_frames)
            ind = np.random.randint(0, self.__len__())
            digit_image = self.datas[ind]
            for i in range(self.num_frames):
                top = start_y[i]
                left = start_x[i]
                bottom = top + self.image_size[0]
                right = left + self.image_size[1]
                data[i, top:bottom, left:right] = np.maximum(data[i, top:bottom, left:right], digit_image)
        data = data[..., np.newaxis]
        return data

    def __getitem__(self, item):
        num_digits = random.choice(self.num_objects)
        images = self._generate_moving_mnist(num_digits)

        inputs = torch.from_numpy(images[:self.num_frames_input]).permute(0, 3, 1, 2).contiguous()
        outputs = torch.from_numpy(images[self.num_frames_input:]).permute(0, 3, 1, 2).contiguous()

        inputs = inputs / 255.
        outputs = outputs / 255.

        return inputs, outputs

    def __len__(self):
        return self.datas.shape[0]