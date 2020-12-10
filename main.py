import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from predictors.Predictors import Predictor
from preprocessor.Preprocessors import WindowGenerator, Preprocessor
from data_loader.DataLoader import DataLoader
from visualization.visuals import ts_plot, volume_vs_price, prediction_plot

if __name__ == "__main__":

    #"NFLX"
    #"MRK
    #"DJIA"
    #"ALV"

    data = DataLoader("NFLX")
    data.gtrends()

    # Visualize data
    #ts_plot(data.hist)
    #volume_vs_price(data)

    overlap = 0.0
    train_test_ratio = 0.8
    input_width = 100
    label_width =100
    processed = Preprocessor(data.hist, window_length=input_width+label_width)
    processed.window_scaling()
    processed.smoothing(alpha=0.7)

    #ts_plot(processed.stock_df, start="2019-11-11")

    # TODO apply LOESS smoothing the data, but only the training data
    # Define batches, time span per sample and prediction sequence
    w1 = WindowGenerator(input_width=input_width,
                         label_width=label_width,
                         train_test_ratio=train_test_ratio,
                         ts_set=processed.stock_df,
                         target="Close",
                         overlap=overlap)

    for example_inputs, example_labels in w1.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')

    predict = True
    if predict:
        w1.make_training_dataset(data.hist)

        pred = Predictor(w1)
        pred.compile_and_fit(pred.lstm)

        test_prediction = pred.predict()
        #pred.make_money()

        prediction_plot(processed.stock_df, test_prediction, plot_start=w1.train_end_index)

    # TODO
    # Define Graph of influences from news ticker between companies. Determine their time series
    # cross correlation coefficient to tell if they run parallel or antiparallel
