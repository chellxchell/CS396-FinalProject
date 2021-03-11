# Temporal Relational Ranking for Stock Prediction
__Relational Stock Ranking (RSR)__ is a new deep learning solution for stock prediction developed by the authors of [this paper](https://arxiv.org/pdf/1809.09441.pdf). The paper contributes the following:
* A proposal for a novel neural network-based framework, RSR, to solve the stock prediction problem in a learning-to-rank fashion
* A proposal for a new component in the neural network modeling, called __Temporal Graph Convolution (TGC)__ that explicitly and quickly captures the domain knowledge of stock relations
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
* [Evaluating the Models](#evaluating-the-models)
* [Summary](#summary)

## Relevance
Stock prediction is important because predicting the future trends of a stock helps investor make more informed, better investment decisions. There are existing traditional solutions for stock predictions that are based on time-series analysis, but these methods are stochastic, and hard to optimize without special knowledge of finance. There are also existing neural-network solutions, but these treat stocks as independent of each other and ignore relationships between different stocks in the same industry. RSR outperforms all of these methods with an average return ratio of 98& and 71% on NYSE and NASDAQ data.

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
As discussed, one of the benefits to __RSR__ is that it takes into account the relations between stocks in the same industry. To observe the trends that stocks under the same industry are similar influenced by, the sector-industry relation between stocks was collected. In NASDAQ and NYSE, each stock in the dataset was classified into a sector and industry.  
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
1. Do stock relations enhance the neural network-based solution for stock prediction? How effective is the proposed __TGC__ component compared to conventional graph-based learning?
1. How does the __RSR__ solution perform under different back-testing strategies?  

A buy-hold-sell trading strategy was adopted to evaluate the performance of stock prediction methods regarding revenue. The target of this experiment was to accurately predict the return ratio of stocks and rank the relative order of stocks. Mean Square Error (MSE), Mean Reciprocal Rank (MRR), and cumulative investment return ratio (IRR) were used to report model performance.  


## Training the Model
Before describing how to train the models, let's go over the proposed __RSR__ framework. First, historical time series data of each stock is fed into the __Long Short-Term Memory (LSTM)__ network. The __LSTM__ network is a special type of  Recurrent Neural Networks (RNNs) used in the proposed __RSR__ model to capture the sequential dependencies and learn a stock-wise sequential embedding. Next, a __Temporal Graph Convolution (TGC)__ is devised to account for stock relations in a time-sensitive way. Finally, the concatenation of sequential embeddings and relational embeddings is fed into a fully connected layer to obtain the ranking score of stocks.
<img src="/blog_images/RSR.png" alt="RSR Framework" width="500">


### Training the Rank_LSRM Model
To answer research question one, the Rank_LSTM method is used. However for the purposes of this experiment, the relational embedding layer was removed for this model, i.e. this Rank_LSTM method ignores stock relations. This is done in order to see a basic solution and study primarily stock ranking formulation without the relation aspect.  

The following is the complete constructor for the `RankLSTM` class:
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
Rank_LSTM outperforms __State Frequency Memory (SFM)__, which is a state-of-the-art neural network-based solution that models the historical data in a recurrent fashion. Rank_LSM also outperforms vanilla __LSTM__ - this performance verifies the advantage of the stock ranking solutions and answers research question 1 that stock ranking is a promising formulation of stock prediction. However, its performance on NYSE is worse than SFM, perhaps because minimizing the combination of point-wise and pair-wise losses leads to a tradeoff between accurately predicting absolute value of return ratios.

<img src="/blog_images/rank_lstm_performance.png" alt="Performance comparison of Rank_LSTM, SFM, and LSTM regarding IRR" width="800">


### Training the Relational Stock Ranking Model
To answer research question 2, the authors studied the effect of industry relations between stocks.  
The code for this model is very similar to the previous Rank_LSRM model, but with the relation data integrated. The following is the constructor for the `ReRaLSTM` class:
```
    def __init__(self, data_path, market_name, tickers_fname, relation_name,
                 emb_fname, parameters, steps=1, epochs=50, batch_size=None, flat=False, gpu=False, in_pro=False):

        seed = 123456789
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

        self.data_path = data_path
        self.market_name = market_name
        self.tickers_fname = tickers_fname
        self.relation_name = relation_name
        # load data
        self.tickers = np.genfromtxt(os.path.join(data_path, '..', tickers_fname),
                                     dtype=str, delimiter='\t', skip_header=False)

        print('#tickers selected:', len(self.tickers))
        self.eod_data, self.mask_data, self.gt_data, self.price_data = \
            load_EOD_data(data_path, market_name, self.tickers, steps)

        # relation data
        rname_tail = {'sector_industry': '_industry_relation.npy',
                      'wikidata': '_wiki_relation.npy'}

        self.rel_encoding, self.rel_mask = load_relation_data(
            os.path.join(self.data_path, '..', 'relation', self.relation_name,
                         self.market_name + rname_tail[self.relation_name])
        )
        print('relation encoding shape:', self.rel_encoding.shape)
        print('relation mask shape:', self.rel_mask.shape)

        self.embedding = np.load(
            os.path.join(self.data_path, '..', 'pretrain', emb_fname))
        print('embedding shape:', self.embedding.shape)

        self.parameters = copy.copy(parameters)
        self.steps = steps
        self.epochs = epochs
        self.flat = flat
        self.inner_prod = in_pro
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

Then, add the `get_batch` function - this is very similar to the `get_batch` function in the Rank_LSTM class we saw earlier, but does have a one-line difference:
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

Then, add the `train` function to the class:
```
def train(self):
        if self.gpu == True:
            device_name = '/gpu:0'
        else:
            device_name = '/cpu:0'
        print('device name:', device_name)
        with tf.device(device_name):
            tf.reset_default_graph()

            seed = 123456789
            random.seed(seed)
            np.random.seed(seed)
            tf.set_random_seed(seed)

            ground_truth = tf.placeholder(tf.float32, [self.batch_size, 1])
            mask = tf.placeholder(tf.float32, [self.batch_size, 1])
            feature = tf.placeholder(tf.float32,
                                     [self.batch_size, self.parameters['unit']])
            base_price = tf.placeholder(tf.float32, [self.batch_size, 1])
            all_one = tf.ones([self.batch_size, 1], dtype=tf.float32)

            relation = tf.constant(self.rel_encoding, dtype=tf.float32)
            rel_mask = tf.constant(self.rel_mask, dtype=tf.float32)

            rel_weight = tf.layers.dense(relation, units=1,
                                         activation=leaky_relu)

            if self.inner_prod:
                print('inner product weight')
                inner_weight = tf.matmul(feature, feature, transpose_b=True)
                weight = tf.multiply(inner_weight, rel_weight[:, :, -1])
            else:
                print('sum weight')
                head_weight = tf.layers.dense(feature, units=1,
                                              activation=leaky_relu)
                tail_weight = tf.layers.dense(feature, units=1,
                                              activation=leaky_relu)
                weight = tf.add(
                    tf.add(
                        tf.matmul(head_weight, all_one, transpose_b=True),
                        tf.matmul(all_one, tail_weight, transpose_b=True)
                    ), rel_weight[:, :, -1]
                )
            weight_masked = tf.nn.softmax(tf.add(rel_mask, weight), dim=0)
            outputs_proped = tf.matmul(weight_masked, feature)
            if self.flat:
                print('one more hidden layer')
                outputs_concated = tf.layers.dense(
                    tf.concat([feature, outputs_proped], axis=1),
                    units=self.parameters['unit'], activation=leaky_relu,
                    kernel_initializer=tf.glorot_uniform_initializer()
                )
            else:
                outputs_concated = tf.concat([feature, outputs_proped], axis=1)

            # One hidden layer
            prediction = tf.layers.dense(
                outputs_concated, units=1, activation=leaky_relu, name='reg_fc',
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
        saver = tf.train.Saver()
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
            'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0
        }
        best_test_perf = {
            'mse': np.inf, 'mrrt': 0.0, 'btl': 0.0
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
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    batch_offsets[j])
                feed_dict = {
                    feature: emb_batch,
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
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: emb_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr, = \
                    sess.run((loss, reg_loss, rank_loss,
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
                emb_batch, mask_batch, price_batch, gt_batch = self.get_batch(
                    cur_offset)
                feed_dict = {
                    feature: emb_batch,
                    mask: mask_batch,
                    ground_truth: gt_batch,
                    base_price: price_batch
                }
                cur_loss, cur_reg_loss, cur_rank_loss, cur_rr = \
                    sess.run((loss, reg_loss, rank_loss,
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
            print('Test MSE:',
                  test_loss / (self.trade_dates - self.test_index),
                  test_reg_loss / (self.trade_dates - self.test_index),
                  test_rank_loss / (self.trade_dates - self.test_index))
            cur_test_perf = evaluate(cur_test_pred, cur_test_gt, cur_test_mask)
            print('\t Test performance:', cur_test_perf)
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
Then, update the model parameters (this is the same `update_model` function as the one in the Rank_LSTM class)
```
    def update_model(self, parameters):
        for name, value in parameters.items():
            self.parameters[name] = value
        return True
```

Finally, after finishing the `ReRaLSTM` class, add the driver code:
```
if __name__ == '__main__':
    desc = 'train a relational rank lstm model'
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

    parser.add_argument('-e', '--emb_file', type=str,
                        default='NASDAQ_rank_lstm_seq-16_unit-64_2.csv.npy',
                        help='fname for pretrained sequential embedding')
    parser.add_argument('-rn', '--rel_name', type=str,
                        default='sector_industry',
                        help='relation type: sector_industry or wikidata')
    parser.add_argument('-ip', '--inner_prod', type=int, default=0)
    args = parser.parse_args()

    if args.t is None:
        args.t = args.m + '_tickers_qualify_dr-0.98_min-5_smooth.csv'
    args.gpu = (args.gpu == 1)

    args.inner_prod = (args.inner_prod == 1)

    parameters = {'seq': int(args.l), 'unit': int(args.u), 'lr': float(args.r),
                  'alpha': float(args.a)}
    print('arguments:', args)
    print('parameters:', parameters)

    RR_LSTM = ReRaLSTM(
        data_path=args.p,
        market_name=args.m,
        tickers_fname=args.t,
        relation_name=args.rel_name,
        emb_fname=args.emb_file,
        parameters=parameters,
        steps=1, epochs=50, batch_size=None, gpu=args.gpu,
        in_pro=args.inner_prod
    )

    pred_all = RR_LSTM.train()
```
Taking industry relations into account was more beneficial to stock ranking for NYSE than it was or NASDAQ, since NASDAQ is much more volatile and dominated by short-term factors. This model was compared to the __Graph Convolutional Network (GCN)__ method, a state-of-the-art graph-based learning method - this replaced the __Temporal Graph Convolution (TGC)__ layer in the RSR model. Another comparison was made to __Graph-based Ranking (GBR)__ - the graph regularization term was added to the loss function of __Rank_LSTM__. The authors found that __RSR_E__ (RSR with explicit modeling) and __RSR_I__ (RSR with implicit modeling) achieve improvement over both GCN and GBR, verifying the effectiveness of the proposed __TGC__ component.

<img src="/blog_images/industry_relations_performance.png" alt="Back-testing procedure of relational ranking methods with industry relations regarding IRR" width="800">

When considering the Wiki relation of stocks, the __RSR_E__ and __RSR_I__ achieve the best performance, further
demonstrating the effectiveness of the __TGC__ component.

<img src="/blog_images/wiki_relations_performance.png" alt="Performance comparison of relational ranking methods with Wiki relations regarding IRR" width="800">

## Evaluating the Models
The performance of the proposed methods was investigated under three different back-testing strategies, __Top1__, __Top5__, and __Top10__, buying stocks with top-1, 5, 10 highest expected revenue, respectively.  

The following is the complete code for the evaluator:
```
def evaluate(prediction, ground_truth, mask, report=False):
    assert ground_truth.shape == prediction.shape, 'shape mis-match'
    performance = {}
    performance['mse'] = np.linalg.norm((prediction - ground_truth) * mask)**2\
        / np.sum(mask)
    mrr_top = 0.0
    all_miss_days_top = 0
    bt_long = 1.0
    bt_long5 = 1.0
    bt_long10 = 1.0

    for i in range(prediction.shape[1]):
        rank_gt = np.argsort(ground_truth[:, i])
        gt_top1 = set()
        gt_top5 = set()
        gt_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(gt_top1) < 1:
                gt_top1.add(cur_rank)
            if len(gt_top5) < 5:
                gt_top5.add(cur_rank)
            if len(gt_top10) < 10:
                gt_top10.add(cur_rank)

        rank_pre = np.argsort(prediction[:, i])

        pre_top1 = set()
        pre_top5 = set()
        pre_top10 = set()
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_pre[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            if len(pre_top1) < 1:
                pre_top1.add(cur_rank)
            if len(pre_top5) < 5:
                pre_top5.add(cur_rank)
            if len(pre_top10) < 10:
                pre_top10.add(cur_rank)

        # calculate mrr of top1
        top1_pos_in_gt = 0
        for j in range(1, prediction.shape[0] + 1):
            cur_rank = rank_gt[-1 * j]
            if mask[cur_rank][i] < 0.5:
                continue
            else:
                top1_pos_in_gt += 1
                if cur_rank in pre_top1:
                    break
        if top1_pos_in_gt == 0:
            all_miss_days_top += 1
        else:
            mrr_top += 1.0 / top1_pos_in_gt

        # back testing on top 1
        real_ret_rat_top = ground_truth[list(pre_top1)[0]][i]
        bt_long += real_ret_rat_top

        # back testing on top 5
        real_ret_rat_top5 = 0
        for pre in pre_top5:
            real_ret_rat_top5 += ground_truth[pre][i]
        real_ret_rat_top5 /= 5
        bt_long5 += real_ret_rat_top5

        # back testing on top 10
        real_ret_rat_top10 = 0
        for pre in pre_top10:
            real_ret_rat_top10 += ground_truth[pre][i]
        real_ret_rat_top10 /= 10
        bt_long10 += real_ret_rat_top10


    performance['mrrt'] = mrr_top / (prediction.shape[1] - all_miss_days_top)
    performance['btl'] = bt_long
    # performance['btl5'] = bt_long5
    # performance['btl10'] = bt_long10
    return performance
```
__RSR_I__ fails to achieve the expected performance when ranking stocks in NASDAq and modeling their industry relations (NASDAQ-Industry setting), indicating less effectiveness of industry relations on NASDAQ. In other cases, the performance of each strategy is __Top1__>__Top5__>__Top10__. This could be because the ranking algorithm could accurately rank the relative order of stocks regarding future return ratios. Once the order is accurate, buying and selling the stock with higher expected profit would achieve higher cumulative return ratio.

<img src="/blog_images/backtesting.png" alt=". Comparison on back-testing strategies (Top1, Top5, and Top10) w.r.t. IRR based on prediction of
RSR_I" width="800">

## Summary
In this blog post, you learned about the propose method __RSR__ for predicting stocks, and why it is superior to existing solutions. Specifically, you learned:
* How to train a model to study stock ranking formation
* How to train a model to examine the effect of industry relations
* How to use back-testing strategies to evaluate the performance of those models
