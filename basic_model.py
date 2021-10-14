"""
把 690 个数据集喂给模型进行训练
训练集 acc: 0.8556
验证集 acc: 0.7357

"""
from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers import *
from tensorflow_core.python.keras.utils.np_utils import to_categorical
import numpy as np
from gen_batch_file import *

# DNA
# with np.load("./row_data/wgEncodeAwgTfbsBroadDnd41CtcfUniPk/train.data.npz") as f:
#     x_train, y_train = f["x_train"], f["y_train"]
#     x_train = to_categorical(x_train, 4)
#     y_train = to_categorical(y_train, 2)
# with np.load("./row_data/wgEncodeAwgTfbsBroadDnd41Ezh239875UniPk/test.data.npz") as f:
#     x_val, y_val = f["x_train"], f["y_train"]
#     x_val = to_categorical(x_val, 4)
#     y_val = to_categorical(y_val, 2)
inputs = Input(shape=(101, 4))

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

# 喂单个文件形式
# model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)
# model.evaluate(x_val, y_val, batch_size=64)

# generator 形式
batch_size = 64
model.fit(x=train_gen(batch_size))
model.evaluate(x=val_gen(batch_size))