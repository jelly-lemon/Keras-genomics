from tensorflow_core.python.keras import Model
from tensorflow_core.python.keras.layers import *
from tensorflow_core.python.keras.utils.np_utils import to_categorical
import numpy as np

# IMDB

# DNA
with np.load("./row_data/DNA.npz") as f:
    x_train, y_train = f["x_train"], f["y_train"]
    y_train = to_categorical(y_train, 2)
inputs = Input(shape=(101,))

# 创建模型
embed_1 = Embedding(input_dim=4, output_dim=20)(inputs)
conv1D_1 = Conv1D(16, 24, activation='relu')(embed_1)
pool1D_1 = MaxPool1D(2)(conv1D_1)
flatten_1 = Flatten()(pool1D_1)
dense_1 = Dense(16, activation="relu")(flatten_1)
drop_1 = Dropout(0.2)(dense_1)
dense_2 = Dense(8, activation="relu")(dense_1)
drop_2 = Dropout(0.2)(dense_2)
outputs = Dense(2, activation="sigmoid")(drop_2)
model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["acc"])

model.fit(x_train, y_train, batch_size=64, epochs=10, validation_split=0.2)