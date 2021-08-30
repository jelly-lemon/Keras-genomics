import os
from abc import abstractmethod

import numpy as np
from hyperopt.pyll.stochastic import sample
from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow_core.python.keras.saving.model_config import model_from_json
import pickle

import DataHelper


class BaseModel:
    """
    自己封装的模型基类
    """
    def __init__(self, search_space: dict, save_dir:str, save_tag:str=""):
        self.search_space = search_space

        # 创建保存文件夹
        self.save_dir = save_dir + "/" + self.__class__.__name__ + "." + save_tag
        if os.path.exists(self.save_dir):
            save_number = 1
            self.save_dir = self.save_dir + str(save_number)
            while os.path.exists(self.save_dir):
                save_number = save_number + 1
                self.save_dir = self.save_dir + str(save_number)
        os.makedirs(self.save_dir)

    @abstractmethod
    def get_model(self, input_shape: tuple, params: dict) -> Model:
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

    def try_params(self, epoch, params, data, data_mode='memory'):
        """
        尝试参数

        :param epoch:
        :param params:
        :param data:
        :param data_mode:
        :return:
        """
        print("try_params:", params)

        # 回调函数
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

        # 读取超参数
        batch_size = params['batch_size']
        loss_func = params["loss_func"]
        optimizer = params["optimizer"]

        # 训练模型
        if data_mode == 'memory':
            X_train, Y_train = data['train']
            X_valid, Y_valid = data['valid']
            input_shape = X_train.shape[1:]

            model = self.get_model(input_shape, params)

            model.fit(
                X_train,
                Y_train,
                batch_size=batch_size,
                epochs=epoch,
                validation_data=(X_valid, Y_valid),
                callbacks=[early_stopping])
            score, acc = model.evaluate(X_valid, Y_valid)

        else:
            train_generator = data['train']['gen_func'](batch_size, data['train']['path'])
            valid_generator = data['valid']['gen_func'](batch_size, data['valid']['path'])
            train_epoch_step = data['train']['n_sample'] / batch_size
            valid_epoch_step = data['valid']['n_sample'] / batch_size
            input_shape = data['train']['gen_func'](batch_size, data['train']['path']).next()[0].shape[1:]

            model = self.get_model(input_shape, params)

            model.fit_generator(
                train_generator,
                steps_per_epoch=train_epoch_step,
                epochs=epoch,
                validation_data=valid_generator,
                validation_steps=valid_epoch_step,
                callbacks=[early_stopping])
            score, acc = model.evaluate_generator(valid_generator, steps=valid_epoch_step)

        result = {'val_loss': score, 'model': (model.to_json(), optimizer, optimizer.get_config(), loss_func)}

        return result

    def train_model(self, model: Model, weight_save_path: str, data_dir: str,
                    batch_size: int, epochs: int, data_mode: str = "memory",
                    steps_ratio: float = 1, shuffle: bool = True) -> tuple:
        """
        训练模型

        :param model:模型
        :param weight_save_path: 权重保存路径
        """
        checkpointer = ModelCheckpoint(filepath=weight_save_path, verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)

        # 加载数据、设置超参数
        if data_mode == 'memory':
            Y_train, traindata = DataHelper.read_data(data_dir + '/train.h5.batch')
            Y_valid, validdata = DataHelper.read_data(data_dir + '/valid.h5.batch')
            history_callback = model.fit(
                traindata,
                Y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(validdata, Y_valid),
                callbacks=[checkpointer, early_stopping],
                shuffle=shuffle)
        else:
            trainbatch_num, train_size = DataHelper.probe_data(data_dir + '/train.h5.batch')
            validbatch_num, valid_size = DataHelper.probe_data(data_dir + '/valid.h5.batch')
            history_callback = model.fit_generator(
                DataHelper.batch_generator(batch_size, data_dir + '/train.h5.batch', shuf=shuffle),
                steps_per_epoch=int((train_size / batch_size) * steps_ratio),
                epochs=epochs,
                validation_data=DataHelper.batch_generator(batch_size, data_dir + '/valid.h5.batch', shuf=shuffle),
                validation_steps=np.ceil(float(valid_size) / batch_size),
                callbacks=[checkpointer, early_stopping])

        return model, history_callback

    def load_saved_model(self, architecture_file: str = None, weightfile2load: str = None,
                         optimizer_file: str = None) -> Model:
        """
        加载模型

        :param architecture_file:
        :param optimizer_file:
        :param weightfile2load:权重文件路径 [in]
        :return:模型
        """
        if os.path.exists(architecture_file) is False:
            info = "architecture_file not exists:" + architecture_file + ", Please search for super parameters first"
            raise Exception(info)

        # 加载模型结构
        model = model_from_json(open(architecture_file).read())
        print("model_from_json succeed:", architecture_file)

        # 加载保存的权重
        if weightfile2load:
            model.load_weights(weightfile2load)

        # 配置超参数
        best_optimizer, best_optimizer_config, best_loss = pickle.load(open(optimizer_file, 'rb'))
        best_optimizer = best_optimizer.from_config(best_optimizer_config)
        model.compile(loss=best_loss, optimizer=best_optimizer, metrics=METRICS)

        return model
