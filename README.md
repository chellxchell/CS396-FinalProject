# Temporal Relational Ranking for Stock Prediction
Relational Stock Ranking (RSR) is a new deep learning solution for stock prediction developed by the authors of [this paper](https://arxiv.org/pdf/1809.09441.pdf). The paper contributes the following:
* A proposal for a novel neural network-based framework, RSR, to solve the stock prediction problem in a learning-to-rank fashion
* A proposal for a new component in the neural network modeling, called Temporal Graph Convolution (TGC) that explicitly and quickly captures the domain knowledge of stock relations
* A demonstration of the effectiveness on these proposals on two real-world stock markets, NYSE and NASDAQ, that will be described later in this blog.  

In this blog post, we will discuss the relevance of the paper, introduce the data used and how it was loaded into the codebase, and describe the methodology used in the experiment. After this, we will describe how to train a model of Rank LSTM and a model of RSR, then evaluate those models, using the code from the paper as a guide.

## Table of Contents
* [Relevance](#relevance)
* [The NYSE and NASDAQ Data](#the-nyse-and-nasdaq-data)
    * [Sequential Data](#sequential-data)
    * [Relational Data](#relational-data)
* [Methodology](#methodology)
* [Training the Model](#training-the-model)
    * [Training the Rank_LSRM Model](#training-the-rank_lsrm-model)
    * [Training the RSR Model](#training-the-relational-stock-ranking-model)
* [Evaluating the Model](#evaluating-the-model)
* [Summary[(#summary)

## Relevance
There are existing traditional solutions for stock predictions that are based on time-series analysis, but these methods are stochastic, and hard to optimize without special knowledge of finance. There are also existing neural-network solutions, but these treat stocks as independent of each other and ignore relationships between different stocks in the same industry. RSR outperforms all of these methods with an average return ratio of 98& and 71% on NYSE and NASDAQ data.

## The NYSE and NASDAQ Data
To justify the method proposed by the authors, it was employed on two real-world markets, New York Stock Exchange (NYSE) and NASDAQ Stock Market (NASDAQ). Stocks from these markets that have transaction records between 01/02/2013 and 12/08/2017 were collected. Any stocks that didn't meet these conditions were filtered out of the data set: stocks must have been traded on more than 98% of trading days since 01/02/2013 to ensure no abnormal patterns occur; stocks must have never been traded at less than $5 per share during the collection period to ensure that the selected stocks are not penny stocks. Filtering out the stocks that did not meet these conditions resulted in 1,026 NASDAQ and 1,737 NYSE stocks. Three kinds of data were collected for these stocks: historical price data, sector-industry relations, and Wiki between their companies (ex. supplier-consumer relation).

### Sequential Data
The authors aimed to predict a ranking list of stocks for the following trading day, based on the daily historical data in the last _S_ trading days. The code below loads the eod (end-of-day) data used in training:

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
### Relational Data
As discussed, one of the benefits to RSR is that it takes into account the relations between stocks in the same industry. To observe the trends that stocks under the same industry are similar influenced by, the sector-industry relation between stocks was collected. In NASDAQ and NYSE, each stock in the dataset was classified into a sector and industry.  
Additionally, the knowledge base Wikidata contains first-order and second-order company relations. A company _i_ has a first-order relation with company _j_ if there is a statement that _i_ and _j_ are the subject and object, respectively. A company _i_ has a second-order relation with the company _j_ if there is a statement that they share the same object. The example below loads the dataset and summarizes the shape of the loaded dataset.

```
def load_relation_data(relation_file):
    relation_encoding = np.load(relation_file)
    print('relation encoding shape:', relation_encoding.shape)
    rel_shape = [relation_encoding.shape[0], relation_encoding.shape[1]]
    mask_flags = np.equal(np.zeros(rel_shape, dtype=int),
                          np.sum(relation_encoding, axis=2))
    mask = np.where(mask_flags, np.ones(rel_shape) * -1e9, np.zeros(rel_shape))
    return relation_encoding, mask
```
<img src="/blog_images/industry_relation.png" alt="Schematic of Industry Relation" width="500">
<img src="/blog_images/wikidata_relation.png" alt="Schematic of Wikidata Relation" width="500">


## Methodology
When conducting their experiment, the authors aimed to answer the following research questions:
1. How is the utility of formulating the stock prediction as a ranking task? Can the RSR solution outperform other prediction solutions?
1. Do stock relations enhance the neural network-based solution for stock prediction? How effective is the proposed TGC component compared to conventional graph-based learning?
1. How does the RSR solution perform under different back-testing strategies?  

A buy-hold-sell trading strategy was adopted to evaluate the performance of stock prediction methods regarding revenue. The target of this experiment was to accurately predict the return ratio of stocks and rank the relative order of stocks. Mean Square Error (MSE), Mean Reciprocal Rank (MRR), and cumulative investment return ratio (IRR) were used to report model performance.  

## Training the Model

### Training the Rank_LSRM Model

### Training the Relational Stock Ranking Model

## Evaluating the Model

## Summary
