import seaborn as sns
import matplotlib.pyplot as plt

def ts_plot(data, start=None, y_value="Close"):
    if start is not None:
        data = data.truncate(before=start, copy=True)

    ts = data.reset_index(col_fill="Date", inplace=False)
    sns.lineplot(x="Date", y=y_value, data=ts)
    plt.show()


def volume_vs_price(data):
    plt.hist2d(data.hist['Close'], data.hist['Volume'], bins=(50,50), vmax=15)
    plt.xlabel('Closing Price')
    plt.ylabel('Volume')
    plt.show()

def prediction_plot(stock_df, prediction_df, plot_start=0):
    #trunc_history = stock_df.truncate(before=str(future_df.index[0].date()), copy=True)

    # Create column form index for plotting
    history = stock_df.reset_index(col_fill="Date", inplace=False)
    future  = prediction_df.reset_index(col_fill="Date", inplace=False)

    plt.plot('Date', 'Close', data=history, marker='', color='olive', linewidth=1)
    plt.legend()

    for prediction_id in range(future.shape[1]-1):
        plt.plot('Date', future.columns[prediction_id+1], data=future)
        plt.legend()

    plt.xlim((future['Date'][plot_start],None))
    plt.show()
