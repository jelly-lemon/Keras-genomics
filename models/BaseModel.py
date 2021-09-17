import json
import os
import sys
from abc import abstractmethod

import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import shutil

from Logger import Logger


class BaseModel:
    """
    模型基类
    """

    def __init__(self, save_dir: str):
        # 创建保存文件夹
        self.save_dir = save_dir + "/" + self.__class__.__name__
        if os.path.exists(self.save_dir):
            c = input(self.save_dir + "  exists" + ", delete it? [y/n] ")
            if c == "y":
                shutil.rmtree(self.save_dir)
            else:
                exit()
        os.makedirs(self.save_dir)
        print("create dir:", self.save_dir, "succeed")

        # 设置模型结构等保存路径
        self.architecture_path = self.save_dir + "/" + self.__class__.__name__ + '_architecture.json'
        self.weight_path = self.save_dir + "/" + self.__class__.__name__ + '_weight.h5'
        self.history_path = self.save_dir + "/" + self.__class__.__name__ + '_history.json'
        self.log_path = self.save_dir + "/" + self.__class__.__name__ + "_log.txt"

        # 保存日志到文件
        sys.stdout = Logger(self.log_path)

        # 创建模型
        self.model = self._get_model()
        with open(self.architecture_path, "w") as architecture_file:
            architecture_file.write(self.model.to_json())

    @abstractmethod
    def _get_model(self) -> Model:
        """
        获取模型
        """

    def _train_model_by_gen(self, epochs: int, batch_size: int,
                            train_gen, train_samples: int,
                            val_gen, val_samples: int):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

        train_epoch_step = train_samples / batch_size
        valid_epoch_step = val_samples / batch_size
        input_shape = train_gen.next()[0].shape[1:]

        model = self._get_model()

        model.fit_generator(
            train_gen,
            steps_per_epoch=train_epoch_step,
            epochs=epochs,
            validation_data=val_gen,
            validation_steps=valid_epoch_step,
            callbacks=[early_stopping])
        score, acc = model.evaluate_generator(val_gen, steps=valid_epoch_step)

        result = {'val_loss': score, 'model': (model.to_json(), optimizer, optimizer.get_config(), loss_func)}

        return result

    def train_by_gen(self):
        pass

    @abstractmethod
    def train(self, x_train, y_train, x_val, y_val):
        """
        训练模型

        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :return:
        """


    def _train_model(self, x_train, y_train, x_val, y_val,
                     train_params:dict, train_callbacks):
        """
        训练模型

        train_params 需要提供：
        batch_size
        epochs

        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :param train_params:
        :param train_callbacks:
        :return:
        """
        # 训练模型
        history = self.model.fit(
            x_train, y_train,
            batch_size=train_params['batch_size'],
            epochs=train_params['epochs'],
            validation_split=0.1,
            #validation_data=(x_val, y_val),
            callbacks=train_callbacks)
        self.model.evaluate(x_val, y_val)


        # 保存训练结果到文件
        with open(self.history_path, "w") as history_file:
            history_file.write(str(history.history))

        return history

    @abstractmethod
    def predict(self, x_predict:np.ndarray)->list:
        """
        预测未知数据

        :param x_predict: 待预测数据
        :return:标签
        """

