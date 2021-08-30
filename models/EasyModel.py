from tensorflow_core.python.keras.optimizer_v2.adadelta import Adadelta

from common_defs import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, GlobalMaxPooling2D
from tensorflow_core.python.keras import Model

from models.BaseModel import BaseModel


class EasyModel(BaseModel):
    def __init__(self, save_dir: str, save_tag: str = ""):
        #
        # 超参数搜索空间
        #
        search_space = {
            'DROPOUT': hp.choice('drop', (0.05, 0.1, 0.2)),
            'DELTA': hp.choice('delta', (1e-04, 1e-06, 1e-08)),
            'MOMENT': hp.choice('moment', (0.9, 0.99, 0.999)),
            'batch_size': hp.choice('batch_size', (32,)),
        }
        super(EasyModel, self).__init__(search_space, save_dir, save_tag)

    def get_model(self, input_shape: tuple, params: dict) -> Model:
        """
        获取模型

        :param params:
        :return:
        """
        dropout = params["dropout"]

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
        model.add(Activation('softmax'))

        #
        # 超参数
        #
        optimizer = Adadelta(epsilon=params['DELTA'], rho=params['MOMENT'])
        loss_func = 'categorical_crossentropy'
        model.compile(loss=loss_func, optimizer=optimizer, metrics=['accuracy'])

        return model










