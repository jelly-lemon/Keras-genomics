import os
import h5py
import numpy as np
from enum import Enum


class DATA_LOAD_MODE(Enum):
    MEMORY = 0      # 加载全部文件到内存
    GENERATOR = 1   # 批生成方式


def get_files_by_prefix(file_prefix: str) -> list:
    """
    获取某目录下指定前缀的文件列表

    如：
    file_prefix: /batch_files/train.h5.batch
    返回：[/batch_files/train.h5.batch1, /batch_files/train.h5.batch2, /batch_files/train.h5.batch3]

    :param file_prefix:文件前缀
    :return:包含该前缀的文件列表
    """
    target_file_paths = []
    data_dir, file_prefix = os.path.split(file_prefix)
    # TODO 优化文件查找
    for _1, _2, all_files in os.walk(data_dir):
        break
    for file_name in all_files:
        if file_name.find(file_prefix) != -1:
            target_file_paths.append(data_dir + "/" + file_name)

    return target_file_paths


def probe_data(file_prefix: str) -> tuple:
    """
    获取文件数量、样本数量

    :param file_prefix:
    :return:
    """
    allfiles = get_files_by_prefix(file_prefix)
    file_count = 0
    sample_count = 0
    for x in allfiles:
        if x.split(file_prefix)[1].isdigit():
            file_count += 1
            data = h5py.File(x, 'r')
            sample_count += len(data['label'])
    return file_count, sample_count


def batch_generator(batch_size, file_prefix, shuf=True):
    """
    生成批数据

    :param self:
    :param batch_size:
    :param file_prefix:
    :param shuf:
    :return:
    """
    allfiles = get_files_by_prefix(file_prefix)
    cache = []
    while True:
        idx2use = np.random.permutation(range(len(allfiles))) if shuf else range(len(allfiles))
        for i in idx2use:
            data1f = h5py.File(file_prefix + str(i + 1), 'r')
            data1 = data1f['batch_files'][()]
            label = data1f['label'][()]
            datalen = len(data1)
            if shuf:
                reorder = np.random.permutation(range(datalen))
                data1 = data1[reorder]
                label = label[reorder]
            minibatch_size = batch_size or datalen
            idx = 0
            if len(cache) != 0:
                idx = minibatch_size - len(cache)
                yield ([np.vstack((cache[0], data1[:idx])), np.vstack((cache[1], label[:idx]))])
            while idx + minibatch_size <= datalen:
                idx += minibatch_size
                yield ([data1[(idx - minibatch_size):idx], label[(idx - minibatch_size):idx]])
            if idx < datalen:
                cache = [data1[idx:], label[idx:]]


def read_data(data_prefix) -> tuple:
    """
    读取数据

    :param self:
    :param data_prefix:
    :return:
    """
    allfiles = get_files_by_prefix(data_prefix)
    batch_file_count = 0
    sample_count = 0
    for file_path in allfiles:
        print("load batch file:", file_path)
        if file_path.split(data_prefix)[1].isdigit():
            batch_file_count += 1
            dataall = h5py.File(file_path, 'r')
            if batch_file_count == 1:
                label = np.asarray(dataall['label'])
                data = np.asarray(dataall['batch_file'])
            else:
                label = np.vstack((label, dataall['label']))
                data = np.vstack((data, dataall['batch_file']))
    return label, data
