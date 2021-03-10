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
* [Summary](#summary)

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
Before describing how to train the models, let's go over the proposed RSR framework. First, historical time series data of each stock is fed into the Long Short-Term Memory (LSTM) network. The LSTM network is a special type of  Recurrent Neural Networks (RNNs) used in the proposed RSR model to capture the sequential dependencies and learn a stock-wise sequential embedding. Next, a Temporal Graph Convolution (TGC) is devised to account for stock relations in a time-sensitive way. Finally, the concatenation of sequential embeddings and relational embeddings is fed into a fully connected layer to obtain the ranking score of stocks.
<img src="/blog_images/RSR.png" alt="RSR Framework" width="500">


### Training the Rank_LSRM Model
To answer research question one, the Rank_LSTM method is used. However for the purposes of this experiment, the relational embedding layer was removed for this model, i.e. this Rank_LSTM method ignores stock relations. This is done in order to see a basic solution and study primarily stock ranking formulation without the relation aspect.  

The following is the complete constructor for the RankLSTM class:
```
    def __init__(self, data_path, market_name, tickers_fname, parameters,
                 steps=1, epochs=50, batch_size=None, gpu=False):
        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)
        ### DEBUG
        # self.tickers = self.tickers[0: 10]
        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)

        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        if batch_size is None:
            self.batch_size = len(self.tickers)
        else:
            self.batch_size = batch_size

        self.valid_index = 756
        self.test_index = 1008
        self.trade_dates = self.mask_data.shape[1]
        self.fea_dim = 5

        self.gpu = gpu
```
Then, add a function for the batch:
```
    def get_batch(self, offset=None):
        if offset is None:
            offset = random.randrange(0, self.valid_index)
        seq_len = self.parameters['seq']
        mask_batch = self.mask_data[:, offset: offset + seq_len + self.steps]
        mask_batch = np.min(mask_batch, axis=1)
        return self.eod_data[:, offset:offset + seq_len, :], \
               np.expand_dims(mask_batch, axis=1), \
               np.expand_dims(
                   self.price_data[:, offset + seq_len - 1], axis=1
               ), \
               np.expand_dims(
                   self.gt_data[:, offset + seq_len + self.steps - 1], axis=1
               )
```

Add the function for training the model:
```
def train(self):
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()

            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 1])
            mask = tf.placeholder(tf.float32, [self.batch_size, 1])
            feature = tf.placeholder(tf.float32,
                [self.batch_size, self.parameters['seq'], self.fea_dim])
            base_price = tf.placeholder(tf.float32, [self.batch_size, 1])
            all_one = tf.ones([self.batch_size, 1], dtype=tf.float32)

            lstm_cell = tf.contrib.rnn.BasicLSTMCell(
                self.parameters['unit']
            )

            initial_state = lstm_cell.zero_state(self.batch_size,
                                                 dtype=tf.float32)
            outputs, _ = tf.nn.dynamic_rnn(
                lstm_cell, feature, dtype=tf.float32,
                initial_state=initial_state
            )

            seq_emb = outputs[:, -1, :]
            # One hidden layer
            prediction = tf.layers.dense(
                seq_emb, units=1, activation=leaky_relu, name='reg_fc',
                kernel_initializer=tf.glorot_uniform_initializer()
            )

            return_ratio = tf.div(tf.subtract(prediction, base_price), base_price)
            reg_loss = tf.losses.mean_squared_error(
                ground_truth, return_ratio, weights=mask
            )
            pre_pw_dif = tf.subtract(
                tf.matmul(return_ratio, all_one, transpose_b=True),
                tf.matmul(all_one, return_ratio, transpose_b=True)
            )
            gt_pw_dif = tf.subtract(
                tf.matmul(all_one, ground_truth, transpose_b=True),
                tf.matmul(ground_truth, all_one, transpose_b=True)
            )
            mask_pw = tf.matmul(mask, mask, transpose_b=True)
            rank_loss = tf.reduce_mean(
                tf.nn.relu(
                    tf.multiply(
                        tf.multiply(pre_pw_dif, gt_pw_dif),
                        mask_pw
                    )
                )
            )
            loss = reg_loss + tf.cast(self.parameters['alpha'], tf.float32) * \
                              rank_loss

            optimizer = tf.train.AdamOptimizer(
                learning_rate=self.parameters['lr']
            ).minimize(loss)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        best_valid_pred = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_gt = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_valid_mask = np.zeros(
            [len(self.tickers), self.test_index - self.valid_index],
            dtype=float
        )
        best_test_pred = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_gt = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_test_mask = np.zeros(
            [len(self.tickers), self.trade_dates - self.parameters['seq'] -
             self.test_index - self.steps + 1], dtype=float
        )
        best_valid_perf = {
            'mse': np.inf, 'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrrt': 0.0,
            'btl': 0.0, 'abtl': 0.0, 'btl5': 0.0, 'abtl5': 0.0, 'btl10': 0.0,
            'abtl10': 0.0, 'rho': -1.0
        }
        best_test_perf = {
            'mse': np.inf, 'top1': 0.0, 'top5': 0.0, 'top10': 0.0, 'mrrt': 0.0,
            'btl': 0.0, 'abtl': 0.0, 'btl5': 0.0, 'abtl5': 0.0, 'btl10': 0.0,
            'abtl10': 0.0, 'rho': -1.0
        }
        best_valid_loss = np.inf

        batch_offsets = np.arange(start=0, stop=self.valid_index, dtype=int)
        for i in range(self.epochs):
            t1 = time()
            np.random.shuffle(batch_offsets)
            tra_loss = 0.0
            tra_reg_loss = 0.0
            tra_rank_loss = 0.0
            for j in range(self.valid_index - self.parameters['seq'] -
                           self.steps + 1):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j])
                feed_dict = {
                    feature: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, batch_out = \
                    sess.run((loss, reg_loss, rank_loss, optimizer),
                             feed_dict)
                tra_loss += cur_loss
                tra_reg_loss += cur_reg_loss
                tra_rank_loss += cur_rank_loss
            print('Train Loss:',
                  tra_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_reg_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1),
                  tra_rank_loss / (self.valid_index - self.parameters['seq'] - self.steps + 1))

            # test on validation set
            cur_valid_pred = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_gt = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            cur_valid_mask = np.zeros(
                [len(self.tickers), self.test_index - self.valid_index],
                dtype=float
            )
            val_loss = 0.0
            val_reg_loss = 0.0
            val_rank_loss = 0.0
            for cur_offset in range(
                self.valid_index - self.parameters['seq'] - self.steps + 1,
                self.test_index - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_semb, cur_rr, = \
                    sess.run((loss, reg_loss, rank_loss, seq_emb,
                              return_ratio), feed_dict)

                val_loss += cur_loss
                val_reg_loss += cur_reg_loss
                val_rank_loss += cur_rank_loss
                cur_valid_pred[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(cur_rr[:, 0])
                cur_valid_gt[:, cur_offset - (self.valid_index -
                                              self.parameters['seq'] -
                                              self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_valid_mask[:, cur_offset - (self.valid_index -
                                                self.parameters['seq'] -
                                                self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
            print('Valid MSE:',
                  val_loss / (self.test_index - self.valid_index),
                  val_reg_loss / (self.test_index - self.valid_index),
                  val_rank_loss / (self.test_index - self.valid_index))
            cur_valid_perf = evaluate(cur_valid_pred, cur_valid_gt,
                                      cur_valid_mask)
            print('\t Valid preformance:', cur_valid_perf)

            # test on testing set
            cur_test_pred = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_gt = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            cur_test_mask = np.zeros(
                [len(self.tickers), self.trade_dates - self.test_index],
                dtype=float
            )
            test_loss = 0.0
            test_reg_loss = 0.0
            test_rank_loss = 0.0
            for cur_offset in range(
                self.test_index - self.parameters['seq'] - self.steps + 1,
                self.trade_dates - self.parameters['seq'] - self.steps + 1
            ):
                eod_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: eod_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_semb, cur_rr = \
                    sess.run((loss, reg_loss, rank_loss, seq_emb,
                              return_ratio), feed_dict)

                test_loss += cur_loss
                test_reg_loss += cur_reg_loss
                test_rank_loss += cur_rank_loss

                cur_test_pred[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(cur_rr[:, 0])
                cur_test_gt[:, cur_offset - (self.test_index -
                                             self.parameters['seq'] -
                                             self.steps + 1)] = \
                    copy.copy(gt_batch[:, 0])
                cur_test_mask[:, cur_offset - (self.test_index -
                                               self.parameters['seq'] -
                                               self.steps + 1)] = \
                    copy.copy(mask_batch[:, 0])
            # print('----------')
            print('Test MSE:',
                  test_loss / (self.trade_dates - self.test_index),
                  test_reg_loss / (self.trade_dates - self.test_index),
                  test_rank_loss / (self.trade_dates - self.test_index))
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            print('\t Test performance:', cur_test_perf)
            # if cur_valid_perf['mse'] < best_valid_perf['mse']:
            if val_loss / (self.test_index - self.valid_index) < \
                    best_valid_loss:
                best_valid_loss = val_loss / (self.test_index -
                                              self.valid_index)
                best_valid_perf = copy.copy(cur_valid_perf)
                best_valid_gt = copy.copy(cur_valid_gt)
                best_valid_pred = copy.copy(cur_valid_pred)
                best_valid_mask = copy.copy(cur_valid_mask)
                best_test_perf = copy.copy(cur_test_perf)
                best_test_gt = copy.copy(cur_test_gt)
                best_test_pred = copy.copy(cur_test_pred)
                best_test_mask = copy.copy(cur_test_mask)

                print('Better valid loss:', best_valid_loss)
            t4 = time()
            print('epoch:', i, ('time: %.4f ' % (t4 - t1)))
        print('\nBest Valid performance:', best_valid_perf)
        print('\tBest Test performance:', best_test_perf)
        sess.close()
        tf.reset_default_graph()

        return best_valid_pred, best_valid_gt, best_valid_mask, \
               best_test_pred, best_test_gt, best_test_mask
```

Update the model parameters:
```
    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True

```

After completing the RankLSTM class, then add the driver code:
```
if __name__ == '__main__':
    desc = 'train a rank lstm model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-p', help='path of EOD data',
                        default='../data/2013-01-01')
    parser.add_argument('-m', help='market name', default='NASDAQ')
    parser.add_argument('-t', help='fname for selected tickers')
    parser.add_argument('-l', default=4,
                        help='length of historical sequence for feature')
    parser.add_argument('-u', default=64,
                        help='number of hidden units in lstm')
    parser.add_argument('-s', default=1,
                        help='steps to make prediction')
    parser.add_argument('-r', default=0.001,
                        help='learning rate')
    parser.add_argument('-a', default=1,
                        help='alpha, the weight of ranking loss')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='use gpu')
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('parameters:', parameters)

    rank_LSTM = RankLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        parameters=parameters,
        steps=1, epochs=50, batch_size=None, gpu=args.gpu
    )
    pred_all = rank_LSTM.train()
```

### Training the Relational Stock Ranking Model

## Evaluating the Model

## Summary
