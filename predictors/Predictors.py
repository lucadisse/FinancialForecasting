import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import date, timedelta, datetime

class Predictor:

    CONV_WIDTH = 3

    def __init__(self, window):
        self.window = window
        self.stock_df = window.stock_df
        self.MAX_EPOCHS = 10
        self.OUT_STEPS = window.label_width
        self.num_features = len(window.stock_df.columns)
        self.train_end_index = window.train_end_index


    @property
    def lstm(self):
        self.model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units]
            tf.keras.layers.LSTM(16, return_sequences=False),
            tf.keras.layers.Dropout(0.3),
            #tf.keras.layers.LSTM(2, return_sequences=False),
            tf.keras.layers.Dense(1000, activation='relu'),
            #tf.keras.layers.Dropout(0.3),
            #tf.keras.layers.LSTM(6, return_sequences=False),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(self.OUT_STEPS,
                                  kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
            # tf.keras.layers.Reshape([self.OUT_STEPS])
        ])

        return self.model


    @property
    def conv1D(self):
        self.model = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
            tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
            # Shape => [batch, 1, conv_units]
            tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
            # Shape => [batch, 1,  out_steps*features]
            tf.keras.layers.Dense(self.OUT_STEPS * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([self.OUT_STEPS, num_features])
        ])
        return self.model

    def compile_and_fit(self, model, patience=2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                          patience=patience,
                                                          mode='min')

        model.compile(loss=tf.losses.MeanSquaredError(),
                      optimizer=tf.optimizers.Adam(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

        history = model.fit(self.window.train, epochs=self.MAX_EPOCHS,
                            validation_data=self.window.test)
                            #,callbacks=[early_stopping])
        return model


    def predict(self):
        prediction = self.model.predict(self.window.test).reshape((-1,self.window.label_width))

        # shape --> [number of predicted sequences, length of total time series]
        padded_prediction = np.empty(shape=(prediction.shape[0], len(self.stock_df)))
        padded_prediction[:] = np.NaN

        # Anchoring the initial prediction value at the last input value

        for batch_id in range(prediction.shape[0]):
            stride  = self.train_end_index + self.window.input_width + int(self.window.input_width * (1-self.window.overlap)) * batch_id

            prediction[batch_id,:] = self.stock_df["Close"][stride] + prediction[batch_id,:] - prediction[batch_id,0]

            padding = [np.NaN for p in range(stride)]

            #if len(padding)+prediction.shape[1] > ts_length:
            #    break

            padded_prediction[batch_id, :len(padding)+prediction.shape[1]] = np.concatenate((padding, prediction[batch_id,:]))

        batch_names = ["Batch " + str(i + 1) for i in range(prediction.shape[0])]

        prediction_df = pd.DataFrame(padded_prediction.T,
                                    index=self.window.stock_df.index,
                                    columns=batch_names)

        return prediction_df

    def make_money(self):
        # TODO perform prediction on very last input window
        future = self.model.predict(self.window.future)
        print(future)

