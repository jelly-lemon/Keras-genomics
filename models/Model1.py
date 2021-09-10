"""
用 softmax/categorical_crossentropy ，acc 一下就上来了。
"""

from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from common_defs import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Conv2D, GlobalMaxPooling2D, BatchNormalization

from models.BaseModel import BaseModel


class Model1(BaseModel):
    def __init__(self, save_dir: str, is_load_saved_model:bool=False):
        #
        # 超参数搜索空间
        #
        search_space = {
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
        model.add(BatchNormalization())
        model.add(Dropout(model_params["dropout"]))
        model.add(Dense(2))
        model.add(Activation(activation=model_params["activation"]))

        #
        # 超参数
        #
        model.compile(loss=model_params["loss_func"], optimizer=model_params["optimizer"], metrics=['accuracy'])

        return model
