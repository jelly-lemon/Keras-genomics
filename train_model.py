import argparse

from tensorflow_core.python.keras.api import keras

import DataHelper
from ModelHelper import *


def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="train model")
    parser.add_argument("-m", "--model_name", dest="model_name", help="Model name")

    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    print(args)
    output_dir = "./saved_models"
    data_dir = "./batch_files"

    # 加载模型
    if args.model_name is None:
        raise Exception("please input model name")
    baseModel = get_base_model(args.model_name, output_dir)

    # mnist
    (x_train, y_train), (x_val, y_val) = keras.datasets.mnist.load_data()
    y_train = keras.utils.to_categorical(y_train)
    y_val = keras.utils.to_categorical(y_val)
    x_train = x_train.reshape(60000, 28, 28, 1).astype("float32") / 255
    x_val = x_val.reshape(10000, 28, 28, 1).astype("float32") / 255

    # 加载数据
    # x_train, y_train = DataHelper.read_data(data_dir + "/train.h5.batch")
    # x_val, y_val = DataHelper.read_data(data_dir + "/valid.h5.batch")

    # 训练
    baseModel.train(x_train, y_train, x_val, y_val)
