from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.optimizers import Adam

from common_defs import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, GlobalMaxPooling2D

from models.BaseModel import BaseModel


class EasyModel(BaseModel):
    def __init__(self, save_dir: str, save_tag: str = ""):
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
        super().__init__(search_space, save_dir, save_tag)

    def get_model(self, params: dict = None) -> Model:
        """
        获取模型

        :param params:
        :return:
        """
        # 超参数
        if params is not None:
            input_shape = params["input_shape"]
            dropout = params["dropout"]
            activation = params["activation"]
            loss_func = params["loss_func"]
        else:
            input_shape = (4, 1, 101)
            dropout = 0.2
            activation = "sigmoid"
            loss_func = "binary_crossentropy"

        #
        # 构建模型
        #
        model = Sequential()
        model.add(Conv2D(128, (1, 24), padding='same', input_shape=input_shape,
                         activation='relu'))  # input_shape 不包含 batch_size
        model.add(GlobalMaxPooling2D())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout))
        model.add(Dense(2))
        model.add(Activation(activation=activation))

        #
        # 超参数
        #
        optimizer = self.get_optimizer(params["optimizer"], params["optimizer_config"])
        model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])

        return model
