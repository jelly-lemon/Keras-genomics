"""
验证集上可以达到 acc 0.9052

在一个 data 上训练，去预测另外一个 data，不能得到好的结果。
"""
from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers import *
from tensorflow_core.python.keras.utils.np_utils import to_categorical
import numpy as np

# DNA
with np.load("./row_data/wgEncodeAwgTfbsBroadDnd41CtcfUniPk/train.data.npz") as f:
    x_train, y_train = f["x_train"], f["y_train"]
    x_train = to_categorical(x_train, 4)
    y_train = to_categorical(y_train, 2)
with np.load("./row_data/wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk/test.data.npz") as f:
    x_val, y_val = f["x_train"], f["y_train"]
    x_val = to_categorical(x_val, 4)
    y_val = to_categorical(y_val, 2)
inputs = Input(shape=x_train[0].shape)

# 创建模型
conv1D_1 = Conv1D(128, 24, activation='relu')(inputs)
pool1D_1 = MaxPool1D(2)(conv1D_1)
flatten_1 = Flatten()(pool1D_1)
dense_1 = Dense(32, activation="relu")(flatten_1)
drop_1 = Dropout(0.2)(dense_1)
outputs = Dense(2, activation="sigmoid")(drop_1)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["acc"])

#model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
model.fit_generator()
model.evaluate(x_val, y_val, batch_size=64)