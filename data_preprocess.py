"""
数据预处理

.fa 文件转 .npz
"""
import numpy as np

# 转换规则
trans_dict = {
    "A": 0,
    "T": 1,
    "C": 2,
    "G": 3
}

# FA 文件
fa_file_path = "./row_data/train.data"

# npz 保存路径
save_file_path = "./row_data/DNA.npz"

# 提取 x_train 和 y_train
x_train = []
y_train = []
with open(fa_file_path) as f:
    line = f.readline()
    while line is not "":
        line = line.split()
        x = line[1]
        x = [trans_dict[e] for e in x]
        x = np.array(x)
        x_train.append(x)
        y = int(line[2])
        y_train.append(y)
        line = f.readline()
np.savez(save_file_path, x_train=x_train, y_train=y_train)

