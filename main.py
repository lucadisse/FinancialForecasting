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
    stock_abrev = "NFLX"
    data = DataLoader(stock_abrev)
    #data.gtrends()
    # Visualize data
    #ts_plot(data.hist)
    #volume_vs_price(data)

    overlap = 0.0
    train_test_ratio = 0.8
    input_width = 100
    label_width =100

    prep = Preprocessor()
    stock = prep.get_technical_indicators(ts=data.hist)
    #print(stock.iloc[3:,:])
    stock = prep.window_scaling(ts=stock, window_length=input_width+label_width)
    #print(stock.iloc[3:,:])
    stock = prep.smoothing(ts=stock, alpha=0.8)
    #print(stock.iloc[3:,:])
    #ts_plot(stock, start="2019-11-11")

    # TODO apply LOESS smoothing the data, but only the training data
    # Define batches, time span per sample and prediction sequence
    w1 = WindowGenerator(input_width=input_width,
                         label_width=label_width,
                         train_test_ratio=train_test_ratio,
                         ts_set=stock,
                         target="Close",
                         overlap=overlap)

    for example_inputs, example_labels in w1.train.take(1):
        print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
        print(f'Labels shape (batch, time, features): {example_labels.shape}')


    #print(stock)
    predict = True
    train = True
    if predict:
        w1.make_training_dataset(stock)

        pred = Predictor(w1)
        if train:
            #if any(stock.isna()):
                #raise ValueError("Data contains NaN.")
            model = pred.compile_and_fit(pred.lstm)
            model.save_weights('./trained_models/latest/'+stock_abrev)

        else:
            model = pred.lstm
            model.load_weights('./trained_models/latest/'+stock_abrev)
            pred.model = model
        test_prediction = pred.predict()
        #pred.make_money()
        #print(test_prediction)
        prediction_plot(stock, test_prediction, plot_start=w1.train_end_index)

    # TODO
    # Define Graph of influences from news ticker between companies. Determine their time series
    # cross correlation coefficient to tell if they run parallel or antiparallel
