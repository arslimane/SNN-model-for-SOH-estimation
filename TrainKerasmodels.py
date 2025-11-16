import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    LSTM,
    Dense,
    Dropout,
    Flatten,
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import numpy as np
import itertools
import struct
import pandas as pd
from tqdm import tqdm
from tensorflow.keras.models import load_model
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pynvml
import time
import gc
from sklearn.metrics import mean_absolute_error, mean_squared_error
import scipy.stats as st


def ci(data):
        se = np.std(data, ddof=1) / np.sqrt(len(data))
        return st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=se),np.std(data)

def build_LSTM_model(inputShape, output):
    # define model
    model2 = Sequential()
    model2.add(
        LSTM(64, activation="relu", input_shape=inputShape, return_sequences=True)
    )
    model2.add(LSTM(32, activation="relu", return_sequences=False))
    model2.add(Dense(output, activation="sigmoid"))

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model2.compile(
        optimizer=opt,
        loss="mae",
        metrics=[
            "mae",
            "mse",
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MeanAbsolutePercentageError(),
        ],
    )
    return model2


def build_CNNLSTM_model(input_shape, output_shape):
    model = Sequential(
        [
            Input(shape=input_shape),
            Conv1D(filters=64, kernel_size=5, activation="relu"),
            MaxPooling1D(pool_size=2),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dense(output_shape, activation="sigmoid"),
        ]
    )
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        optimizer=opt,
        loss="mae",
        metrics=[
            "mae",
            "mse",
            tf.keras.metrics.RootMeanSquaredError(),
            tf.keras.metrics.MeanAbsolutePercentageError(),
        ],
    )
    return model


def TestLSTM(Steps_in, Steps_out, maxlength=1500):
    for stpsin, stpout in tqdm(zip(Steps_in, Steps_out)):
        data = np.load(f"Jobs\\Data\\Xtrain_normalized_{stpsin}_{stpout}.npz")
        X1 = data["Data"]
        data = np.load(f"Jobs\\Data\\Ytrain_{stpsin}_{stpout}.npz")
        Y1 = data["y"]
        data = np.load(f"Jobs\\Data\\Xtest_normalized_{stpsin}_{stpout}.npz")
        X2 = data["Data"]
        data = np.load(f"Jobs\\Data\\Ytest_{stpsin}_{stpout}.npz")
        Y2 = data["y"]
        X1 = np.float32(X1)
        Y1 = np.float32(Y1)
        X2 = np.float32(X2)
        Y2 = np.float32(Y2)

        input_shape = (maxlength * stpsin, 3)
        hidden_size1 = 256
        hidden_size2 = 128
        hidden_size3 = 100
        code_size = 64

        input_layer = Input(shape=(input_shape))

        hidden_layer_0 = Flatten()(input_layer)
        hidden_layer_1 = Dense(hidden_size1, activation="relu")(hidden_layer_0)
        hidden_layer_2 = Dense(hidden_size2, activation="relu")(hidden_layer_1)
        hidden_layer_3 = Dense(hidden_size3, activation="relu")(hidden_layer_2)
        bottleneck_layer = Dense(code_size * stpsin, activation="sigmoid")(
            hidden_layer_3
        )

        # Define the decoder network
        hidden_layer_5 = Dense(hidden_size1, activation="relu")(bottleneck_layer)
        hidden_layer_6 = Dense(hidden_size2, activation="relu")(hidden_layer_5)
        hidden_layer_7 = Dense(hidden_size1, activation="relu")(hidden_layer_6)
        hidden_layer_8 = Dense(maxlength * stpsin * 3, activation="sigmoid")(
            hidden_layer_7
        )
        output_layer = Reshape((maxlength * stpsin, 3))(hidden_layer_8)

        autoencoder = Model(input_layer, output_layer)
        opt = Adam(learning_rate=0.0001)
        autoencoder.compile(
            optimizer=opt,
            loss="mae",
            metrics=["mae", "mse", tf.keras.metrics.RootMeanSquaredError()],
        )
        early_stopping = EarlyStopping(monitor="val_loss", patience=5, mode="min")
        autoencoder.fit(
            X1,
            X1,
            epochs=50,
            batch_size=32,
            validation_data=(X2, X2),
            callbacks=[early_stopping],
            verbose=0,
        )
        print("Encoder done")
        encoder = Model(input_layer, bottleneck_layer)
        encoder.compile(optimizer="adam", loss="mae")
        X11 = encoder.predict(X1)
        X22 = encoder.predict(X2)
        X11 = np.float32(np.reshape(X11, (len(X11), stpsin, 64)))
        X22 = np.float32(np.reshape(X22, (len(X22), stpsin, 64)))
        input_shape = (stpsin, 64)
        model2 = build_LSTM_model(input_shape, stpout)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                f"Jobs\\Models\\ELSTM_{stpsin}_{stpout}.keras", save_best_only=True
            ),
        ]

        history = model2.fit(
            X11,
            Y1,
            validation_split=0.1,
            epochs=4,  # to be changed
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )
        test_loss = model2.evaluate(X22, Y2)
        ypredict=model2.predict(X22)
        print("Model done")
        input_shape = (maxlength * stpsin, 3)
        fsize = 64

        input_layer = Input(shape=(input_shape))
        hidden_layer_0 = encoder(input_layer)
        hidden_layer_1 = Reshape((stpsin, fsize))(hidden_layer_0)
        output_layer = model2(hidden_layer_1)

        model3 = Model(input_layer, output_layer)
        model3.compile(optimizer="adam", loss="mae")

        model3.save(f"Jobs\\Models\\ELSTMt_{stpsin}_{stpout}.keras")
        
        del X1, X2, Y1, Y2, X11, X22, encoder, autoencoder, model2, model3
        gc.collect()
        tf.keras.backend.clear_session()


def TestCNN(Steps_in, Steps_out, maxlength=2000):
    for stpsin, stpout in tqdm(zip(Steps_in, Steps_out)):
        data = np.load(f"Jobs\\Data\\Xtrain_normalized_{stpsin}_{stpout}.npz")
        X1 = data["Data"]
        data = np.load(f"Jobs\\Data\\Ytrain_{stpsin}_{stpout}.npz")
        Y1 = data["y"]
        X1 = np.float32(X1)
        Y1 = np.float32(Y1)
        
        input_shape = (maxlength * stpsin, 3)
        model2 = build_CNNLSTM_model(input_shape, stpout)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
            ModelCheckpoint(
                f"Jobs\\Models\\CNNLSTM_{stpsin}_{stpout}.keras", save_best_only=True
            ),
        ]

        history = model2.fit(
            X1,
            Y1,
            validation_split=0.1,
            epochs=4,  # to be changed
            batch_size=32,
            callbacks=callbacks,
            verbose=1,
        )
       
        del X1,Y1, model2, history
        gc.collect()
        tf.keras.backend.clear_session()


Steps_in = [1,10, 20, 25, 25, 25, 25]
Steps_out = [1,5, 10, 25, 50, 100, 200]

TestLSTM(Steps_in, Steps_out, maxlength=1500)
TestCNN(Steps_in, Steps_out, maxlength=1500)
