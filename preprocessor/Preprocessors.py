import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

class Preprocessor:

    def __init__(self, ts, window_length):
        self.stock_df = ts
        self.window_length = window_length


    def unscale_ts(self, prediction):
        # TODO impelemnt some kind of unscaling
        b=0


    def window_scaling(self):
        # scale input by window
        scaler = StandardScaler()
        stock_array = self.stock_df.values

        for ti in range(0, stock_array.shape[0], self.window_length):
            scaler.fit(stock_array[ti:ti + self.window_length, :])
            stock_array[ti:ti + self.window_length, :] = scaler.transform(stock_array[ti:ti + self.window_length, :])

        self.stock_df  = pd.DataFrame(stock_array,
                                     columns=self.stock_df.columns,
                                     index=self.stock_df.index)


    def smoothing(self, alpha=0):
        prev_value = 0
        # alpha = % of new obervation
        stock_array = self.stock_df.values

        for col in range(stock_array.shape[1]):
            for ti in range(stock_array.shape[0]):
                stock_array[ti, col] = alpha * stock_array[ti, col] + (1 - alpha) * prev_value
                prev_value = stock_array[ti, col]

        print(self.stock_df.columns)
        self.stock_df  = pd.DataFrame(stock_array,
                                     columns=self.stock_df.columns,
                                     index=self.stock_df.index)


class WindowGenerator():
    def __init__(self, input_width, label_width, ts_set, train_test_ratio=0.8, target="Close", overlap=0):

        # Store the raw data.
        self.stock_df = ts_set
        self.train_end_index = int(len(ts_set)*train_test_ratio)
        self.target = target
        # Window parameters
        self.input_width = input_width
        self.label_width = label_width

        self.total_window_size = input_width + label_width

        # Overlapping of two consecutive windows
        self.overlap = overlap
        # start = 0, stop = input_width of time window batch
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

        self.test_label_times = self.stock_df.index[self.labels_slice]

        # Training hyperparameters
        self.BATCH_SIZE = 6

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


    def split_window(self, ds_batch):
        # [batches, rows, columns]
        inputs = ds_batch[:, self.input_slice, :]

        # TODO Currently only one target variable is possible
        target_col_index = list(self.stock_df.columns).index(self.target)
        labels = ds_batch[:, self.labels_slice, target_col_index]


        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width])

        return inputs, labels

    def make_training_dataset(self, data_df, validation=True):
        # Create windows from the whole time series
        # Manually splits training and test sequence via mapping in the end

        data = np.array(data_df, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=int(self.input_width * (1-self.overlap)),
            shuffle=True,
            batch_size=self.BATCH_SIZE,
            end_index=self.train_end_index)

        ds = ds.map(self.split_window)

        return ds


    def make_validation_dataset(self, data_df):
        data = np.array(data_df, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=int(self.input_width * (1-self.overlap)),
            shuffle=False,
            batch_size=1,
            start_index=self.train_end_index+1)

        ds = ds.map(self.split_window)

        return ds

    def make_prediction_dataset(self, data_df):
        data = np.array(data_df, dtype=np.float32)

        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.input_width,
            #sequence_stride=self.input_width,
            shuffle=False,
            batch_size=1)

        return ds


    @property
    def train(self):
        return self.make_training_dataset(self.stock_df)

    @property
    def test(self):
        return self.make_validation_dataset(self.stock_df)

    @property
    def future(self):
        return self.make_prediction_dataset(self.stock_df.tail(self.input_width))

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result