"""
用 softmax/categorical_crossentropy ，acc 一下就上来了。
"""

from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from common_defs import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, GlobalMaxPooling2D

from models.BaseModel import BaseModel


class EasyModel(BaseModel):
    def __init__(self, save_dir: str, is_load_saved_model:bool=False):
        #
        # 超参数搜索空间
        #
        search_space = {
            'dropout': hp.choice('dropout', (0.05, 0.1, 0.2)),
            # 'DELTA': hp.choice('delta', (1e-04, 1e-06, 1e-08)),
            # 'MOMENT': hp.choice('moment', (0.9, 0.99, 0.999)),
            'batch_size': hp.choice('batch_size', (32,)),
            'loss_func': hp.choice('loss_func', ('binary_crossentropy',)),
            'optimizer': hp.choice('optimizer', ('Adam',)),
            'activation': hp.choice('activation', ('softmax',)),
            'optimizer_config': hp.choice('optimizer_config', ({},))
        }
        super().__init__(search_space, save_dir, is_load_saved_model)

    def train(self, x_train, y_train, x_val, y_val):
        # ModelCheckpoint(filepath=self.save_dir, verbose=1, save_best_only=True)
        # EarlyStopping(monitor='val_loss', patience=3, verbose=0)
        callbacks = []

        train_params = {
            "batch_size": 32,
            "epochs": 10,
        }
        print("train_params:", train_params)

        return self._train_model(x_train, y_train, x_val, y_val, train_params, callbacks)

    def _get_model(self) -> Model:
        """
        获取模型
        """
        # 默认参数
        model_params = {
            "input_shape": (4, 1, 101),
            "dropout": 0.2,
            "activation": "softmax",
            "loss_func": "categorical_crossentropy",
            "optimizer": Adam()
        }

        # 超参数
        print("model_params:", model_params)

        #
        # 构建模型
        #
        model = Sequential()
        model.add(Conv2D(128, (1, 24), padding='same', input_shape=model_params["input_shape"],
                         activation='relu'))  # input_shape 不包含 batch_size
        model.add(GlobalMaxPooling2D())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(model_params["dropout"]))
        model.add(Dense(2))
        model.add(Activation(activation=model_params["activation"]))

        #
        # 超参数
        #
        model.compile(loss=model_params["loss_func"], optimizer=model_params["optimizer"], metrics=['accuracy'])

        return model
