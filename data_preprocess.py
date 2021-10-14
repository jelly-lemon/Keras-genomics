"""
数据预处理

.fa 文件转 .npz
"""
import os

import numpy as np

# 转换规则
trans_dict = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3
}


def convert2npz(fa_file_path, save_file_path):
    # 提取 x_train 和 y_train
    x_train = []
    y_train = []
    with open(fa_file_path) as f:
        line = f.readline()
        while line is not "":
            line = line.split()
            x = line[1]
            x = [trans_dict[e] if e in trans_dict else 0 for e in x]
            x = np.array(x)
            x_train.append(x)
            y = int(line[2])
            y_train.append(y)
            line = f.readline()
    dir, name = os.path.split(save_file_path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    np.savez(save_file_path, x_train=x_train, y_train=y_train)


top_dir = "./ChIP-seq_690"
dirs = os.listdir(top_dir)
for dir in dirs:
    sub_dir = top_dir + "/" + dir
    names = os.listdir(sub_dir)
    for name in names:
        file_path = sub_dir + "/" + name
        save_path = "./row_data" + "/" + dir + "/" + name + ".npz"
        convert2npz(file_path, save_path)
