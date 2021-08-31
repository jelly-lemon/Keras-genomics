import json
import os
from abc import abstractmethod

from hyperopt.pyll.stochastic import sample
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import pickle

import shutil


class BaseModel:
    """
    自己封装的模型基类
    """

    def __init__(self, search_space: dict, save_dir: str, save_tag: str = ""):
        self.search_space = search_space

        # 创建保存文件夹
        if save_tag:
            self.save_dir = save_dir + "/" + self.__class__.__name__ + "." + save_tag
        else:
            self.save_dir = save_dir + "/" + self.__class__.__name__
        if os.path.exists(self.save_dir):
            c = input(self.save_dir + "  exists" + ", delete it? [y/n] ")
            if c == "y":
                shutil.rmtree(self.save_dir)
                print("deleted")
            else:
                exit()
        os.makedirs(self.save_dir)
        print("create dir:", self.save_dir)

    @abstractmethod
    def get_model(self, params: dict) -> Model:
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

        model = self.get_model(input_shape, params)

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

    def train(self, x_train, y_train, x_val, y_val,
              params):
        checkpoint = ModelCheckpoint(filepath=self.save_dir, verbose=1, save_best_only=True)
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
        callbacks = [checkpoint, early_stopping]

        return self._train_model(x_train, y_train, x_val, y_val, params, callbacks)

    def _train_model(self, x_train, y_train, x_val, y_val,
                     params, callbacks):
        print("_train_model:", params)

        # 获取模型
        input_shape = x_train.shape[1:]
        params["input_shape"] = input_shape
        model = self.get_model(params)

        # 训练模型
        history_callback = model.fit(
            x_train, y_train,
            batch_size=params['batch_size'],
            epochs=params['epochs'],
            validation_data=(x_val, y_val),
            callbacks=callbacks)
        val_loss, val_acc = model.evaluate(x_val, y_val)

        result = {'val_loss': val_loss, 'val_acc': val_acc,
                  'model': json.loads(model.to_json()),
                  'params': params}

        return result, history_callback

    def try_params(self, x_train, y_train, x_val, y_val,
                   params):
        """
        尝试参数
        """
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=0)
        callbacks = [early_stopping]


        result, history_callback = self._train_model(x_train, y_train, x_val, y_val,
                                                     params=params, callbacks=callbacks)

        return result, history_callback

    def try_params_by_gen(self):
        self._train_model_by_gen(train_gen=train_gen, train_samples=train_samples,
                                 val_gen=val_gen, val_samples=val_samples,
                                 epochs=epochs, batch_size=batch_size)

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
        model.compile(loss=best_loss, optimizer=best_optimizer, metrics=['acc'])

        return model


    def get_optimizer(self, name, params):
        if name == "Adam":
            return Adam()