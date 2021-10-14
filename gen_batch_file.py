"""
生成 batch 文件
"""
import os

import numpy as np
from tensorflow_core.python.keras.utils.np_utils import to_categorical



def train_gen(batch_size):
    top_dir = "./row_data"
    dirs = os.listdir(top_dir)
    for dir in dirs:
        train_file_path = top_dir + "/" + dir + "/train.data.npz"
        print(train_file_path)
        with np.load(train_file_path) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_train = to_categorical(x_train, 4)
            y_train = to_categorical(y_train, 2)
            batch_nums = int(len(x_train)/batch_size)
            for i in range(batch_nums):
                yield x_train[i*batch_size:(i+1)*batch_size], y_train[i*batch_size:(i+1)*batch_size]

def val_gen(batch_size):
    top_dir = "./row_data"
    dirs = os.listdir(top_dir)
    for dir in dirs:
        train_file_path = top_dir + "/" + dir + "/test.data.npz"
        with np.load(train_file_path) as f:
            x_train, y_train = f["x_train"], f["y_train"]
            x_train = to_categorical(x_train, 4)
            y_train = to_categorical(y_train, 2)
            batch_nums = int(len(x_train) / batch_size)
            for i in range(batch_nums):
                yield x_train[i * batch_size:(i + 1) * batch_size], y_train[i * batch_size:(i + 1) * batch_size]
