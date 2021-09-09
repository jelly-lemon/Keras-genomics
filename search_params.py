"""
搜索参数
"""
from __future__ import print_function

import argparse
import h5py
import os
import pickle
import shutil
import sys
import time
from pprint import pprint
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

import DataHelper
from Hyperband import Hyperband
from models.ModelHelper import *
import json

# 当前工作目录
cwd = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="Keras + Hyperband for genomics")
    parser.add_argument("--save_tag", dest="save_tag", default="", action='store_true', help="Output directory tag")
    parser.add_argument("-d", "--data_dir", dest="data_dir", default='./batch_files',
                        help="The batch_files directory")
    parser.add_argument("-m", "--model_name", dest="model_name", help="Model name")
    parser.add_argument("-o", "--out_dir", dest="out_dir", default='./saved_models',
                        help="Model output directory")
    parser.add_argument("-te", "--trainepoch", default=20, type=int, help="The number of epochs to train for")
    parser.add_argument("-pa", "--patience", default=10, type=int,
                        help="number of epochs with no improvement after which training will be stopped.")
    parser.add_argument("-bs", "--batchsize", default=100, type=int, help="Batchsize in SGD-based training")
    parser.add_argument("-w", "--weightfile", default=None, help="Weight file for the best model")
    parser.add_argument("-l", "--lweightfile", default=None, help="Weight file after training")
    parser.add_argument("-r", "--retrain", default=None, help="codename for the retrain run")
    parser.add_argument("-rw", "--rweightfile", default='', help="Weight file to load for retraining")
    parser.add_argument("-dm", "--data_mode", default='memory',
                        help="whether to load batch_files into memory ('memory') or using a generator('generator')")
    parser.add_argument("-ei", "--evalidx", dest='evalidx', default=0, type=int,
                        help="which output neuron (0-based) to calculate 2-class auROC for")
    parser.add_argument("--epochratio", default=1, type=float,
                        help="when training with batch_files generator, optionally shrink each epoch size by this factor to enable more frequen evaluation on the valid set")
    parser.add_argument("-shuf", default=1, type=int,
                        help="whether to shuffle the batch_files at the begining of each epoch (1/0)")

    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 获取模型名
    model_name = args.model_name
    if model_name is None:
        raise Exception("please input model name")

    # 加载模型
    baseModel = get_model(model_name, args.out_dir, args.save_tag)

    # 获取输出目录
    out_dir = baseModel.save_dir

    # 设置模型结构等保存路径
    architecture_file = os.path.join(out_dir, model_name + '_best_archit.json')
    optimizer_file = os.path.join(out_dir, model_name + '_best_optimer.json')
    weight_file_out_path = os.path.join(out_dir,
                                        model_name + '_bestmodel_weights.h5') if args.weightfile is None else args.weightfile
    last_weight_file = os.path.join(out_dir,
                                    model_name + '_lastmodel_weights.h5') if args.lweightfile is None else args.lweightfile
    eval_out_path = os.path.join(out_dir, model_name + '_eval.txt')

    # 加载数据
    print(args)
    x_train, y_train = DataHelper.read_data(args.data_dir + "/train.h5.batch")
    x_valid, y_valid = DataHelper.read_data(args.data_dir + "/valid.h5.batch")


    # 搜索最优参数
    hb = Hyperband(baseModel)
    results = hb.search(x_train, y_train, x_valid, y_valid,
                        skip_last=1)

    # 保存最优参数到文件 architecture_file
    best_result = sorted(results, key=lambda x: x['val_loss'])[0]
    print(best_result)
    json.dump(best_result['model'], open(architecture_file, 'w'))
    json.dump(best_result["params"], open(optimizer_file, 'w'))


