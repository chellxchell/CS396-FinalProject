# Temporal Relational Ranking for Stock Prediction
Relational Stock Ranking (RSR) is a new deep learning solution for stock prediction developed by the authors of [this paper](https://arxiv.org/pdf/1809.09441.pdf). There are traditional solutions for stock predictions that are based on time-series analysis, but these methods are stochastic, and hard to optimize without special knowledge of finance. There are also existing neural-network solutions, but these treat stocks as independent of each other and ignore relationships between different stocks in the same industry. RSR outperforms all of these methods with an average return ratio of 98& and 71% on NYSE and NASDAQ data.

## The NYSE and NASDAQ Data
To justify the method proposed by the authors, it was employed on two real-world markets, New York Stock Exchange (NYSE) and NASDAQ Stock Market (NASDAQ). Stocks from these markets that have transaction records between 01/02/2013 and 12/08/2017 were collected. Any stocks that didn't meet these conditions were filtered out of the data set: stocks must have been traded on more than 98% of trading days since 01/02/2013 to ensure no abnormal patterns occur; stocks must have never been traded at less than $5 per share during the collection period to ensure that the selected stocks are not penny stocks. Filtering out the stocks that did not meet these conditions resulted in 1,026 NASDAQ and 1,737 NYSE stocks. Three kinds of data were collected for these stocks: historical price data, sector-industry relations, and Wiki between their companies (ex. supplier-consumer relation).
For sequential data, the authors aimed to predict a ranking list of stocks for the following trading day, based on the daily historical data in the last _S_ trading days. The code below loads the eod (end-of-day) data used in training:
```
def load_EOD_data(data_path, market_name, tickers, steps=1):
    eod_data = []
    masks = []
    ground_truth = []
    base_price = []
    for index, ticker in enumerate(tickers):
        single_EOD = np.genfromtxt(
            os.path.join(data_path, market_name + '_' + ticker + '_1.csv'),
            dtype=np.float32, delimiter=',', skip_header=False
        )
        if market_name == 'NASDAQ':
            # remove the last day since lots of missing data
            single_EOD = single_EOD[:-1, :]
        if index == 0:
            print('single EOD data shape:', single_EOD.shape)
            eod_data = np.zeros([len(tickers), single_EOD.shape[0],
                                 single_EOD.shape[1] - 1], dtype=np.float32)
            masks = np.ones([len(tickers), single_EOD.shape[0]],
                            dtype=np.float32)
            ground_truth = np.zeros([len(tickers), single_EOD.shape[0]],
                                    dtype=np.float32)
            base_price = np.zeros([len(tickers), single_EOD.shape[0]],
                                  dtype=np.float32)
        for row in range(single_EOD.shape[0]):
            if abs(single_EOD[row][-1] + 1234) < 1e-8:
                masks[index][row] = 0.0
            elif row > steps - 1 and abs(single_EOD[row - steps][-1] + 1234) \
                    > 1e-8:
                ground_truth[index][row] = \
                    (single_EOD[row][-1] - single_EOD[row - steps][-1]) / \
                    single_EOD[row - steps][-1]
            for col in range(single_EOD.shape[1]):
                if abs(single_EOD[row][col] + 1234) < 1e-8:
                    single_EOD[row][col] = 1.1
        eod_data[index, :, :] = single_EOD[:, 1:]
        base_price[index, :] = single_EOD[:, -1]
    return eod_data, masks, ground_truth, base_price
 ```
