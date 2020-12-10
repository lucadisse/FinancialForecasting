import yfinance
from pytrends.request import TrendReq
import seaborn as sns
import matplotlib.pyplot as plt


class DataLoader:

    def __init__(self, key):
        self.stock = yfinance.Ticker(key)
        self.hist = self.stock.history(period="max")
        self.feature_selection()
        self.pytrend = TrendReq()

    def printhist(self):
        print(self.hist)

    def gtrends(self):
        # form google trends
        search_keys = ["china", "trump"]
        self.pytrend.build_payload(kw_list=search_keys,
                                   timeframe='all')
        interest_over_time = self.pytrend.interest_over_time()
        ts = interest_over_time.reset_index(col_fill="date", inplace=False)
        sns.lineplot(x="date", y=search_keys[0], data=ts)
        plt.show()

    def feature_selection(self):
        self.hist.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)
        #self.hist[search_keys] = interest
