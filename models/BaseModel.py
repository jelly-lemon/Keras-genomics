import json
import os
from abc import abstractmethod

from hyperopt.pyll.stochastic import sample
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle

import shutil

from tensorflow_core.python.keras.saving.model_config import model_from_json


class BaseModel:
    """
    模型基类
    """

    def __init__(self, search_space: dict, save_dir: str, is_load_saved_model:bool=False):
        self.search_space = search_space

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
        self.architecture_path = self.save_dir + "/" + self.__class__.__name__ + '_architech.json'
        self.params_path = self.save_dir + "/" + self.__class__.__name__ + '_params.json'
        self.weight_path = self.save_dir + "/" + self.__class__.__name__ + '_weight.h5'
        self.eval_path = self.save_dir + "/" + self.__class__.__name__ + '_eval.txt'

        if is_load_saved_model:
            self.model = self._load_saved_model()
        else:
            self.model = self._get_model()

    @abstractmethod
    def _get_model(self) -> Model:
        """
        获取模型
        """

    def get_params(self) -> dict:
        """
        获取待尝试参数
        """
        params = sample(self.search_space)  # 返回一个字典，随机选择，如：{'DELTA': 0.0001, 'DROPOUT': 0.05, 'MOMENT': 0.99}

        # 有些数值应该是 int 类型，但是前面得到的却是 float 类型，
        # 这里转换一下
        new_params = {}
        for k, v in params.items():
            if type(v) == float and int(v) == v:
                new_params[k] = int(v)
            else:
                new_params[k] = v

        return new_params

    def _train_model_by_gen(self, epochs: int, batch_size: int,
                            train_gen, train_samples: int,
                            val_gen, val_samples: int):
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

        train_epoch_step = train_samples / batch_size
        valid_epoch_step = val_samples / batch_size
        input_shape = train_gen.next()[0].shape[1:]

        model = self._get_model(input_shape, params)

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
                     train_params, train_callbacks):
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
        history_callback = self.model.fit(
            x_train, y_train,
            batch_size=train_params['batch_size'],
            epochs=train_params['epochs'],
            validation_data=(x_val, y_val),
            callbacks=train_callbacks)
        val_loss, val_acc = self.model.evaluate(x_val, y_val)

        return history_callback

    def try_params(self, x_train, y_train, x_val, y_val,
                   params):
        """
        尝试参数
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
        callbacks = [early_stopping]

        result, history_callback = self._train_model(x_train, y_train, x_val, y_val,
                                                     train_params=params, train_callbacks=callbacks)

        return result, history_callback

    def try_params_by_gen(self):
        self._train_model_by_gen(train_gen=train_gen, train_samples=train_samples,
                                 val_gen=val_gen, val_samples=val_samples,
                                 epochs=epochs, batch_size=batch_size)

    def _load_saved_model(self, isLoadArchitecture: bool = True, isLoadWeight: bool = True,
                          isLoadParams: bool = True) -> Model:
        """
        加载保存的模型


        :return:模型
        """
        # 加载超参数
        params = None
        if isLoadParams:
            if os.path.exists(self.params_path):
                params = json.load(open(self.params_path))
                print("load params succeed:", params)

        # 加载模型结构
        if isLoadArchitecture:
            if os.path.exists(self.architecture_path):
                model = model_from_json(open(self.architecture_path).read())
                print("model_from_json succeed:", self.architecture_path)
            else:
                model = self._get_model(params)
                print(self.architecture_path, "not exists, use default model")

        # 加载保存的权重
        if isLoadWeight:
            if os.path.exists(self.weight_path):
                model.load_weights(self.weight_path)

        return model
