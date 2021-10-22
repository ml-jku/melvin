# melvin

## Data
Download the file `melvinFull.db.gz` from [here](https://ml.jku.at/research/melvin/downloads/) and decompress it. The file `data.py` implements data access. For learning, the most central class is 
`MelvinDataset`, which wraps an [SQLite](https://www.sqlite.org/) database file as `pytorch.Dataset`. The file 
`melvinFull.db` contains all the data. `data.py` has a main, which implements a few commands for 
generating the database from text files, subsetting the data, deleting duplicates, sanity checks, 
and visualizing some statistics.

## SRV Prediction
The Schmidt-rank vector (SRV) resulting from an experiment is a 3-tuple $`(n, m, k)`$ with 
$`n \geq m \geq k`$ and $`m \cdot k \geq n`$, where $`n, m, k \in \mathbb N`$. 

Since SRV entries represent counts, we model them as Poisson processes. The loss function is the 
negative log likelihood, i.e.
```math
L\left(y_i, \hat{y}_i\right) = -\sum_{i=1}^m \left( y_i \hat{y}_i - e^{\hat{y}_i} - \log(y_i!)\right),
```
which is implemented in `torch.nn.PoissonNLLLoss`. We want to compare this to 
least-squares and Huber-loss regression and rounding.

## Run
1. clone the repo by `git clone https://git.bioinf.jku.at/adler/melvin.git`
2. decompress the database by `cat melvinFull.db.gz | gunzip > melvinFull.db`
3. generate the G6 data sets by `python3 data.py create_g6_sqlite_databases --dbfile melvinFull.db`
4. train the model

```
tomte@grumz$ python3 main.py --help
usage: main.py [-h] [--batch-size N] [--epochs N] [--lr LR] [--momentum M]
               [--no-cuda] [--seed S] [--log-interval N] [--save-model]
               [--dataset {full,g6}]

melvin args

optional arguments:
  -h, --help           show this help message and exit
  --batch-size N       input batch size for training (default: 256)
  --epochs N           number of epochs to train (default: 10)
  --lr LR              learning rate (default: 0.1)
  --momentum M         SGD momentum (default: 0.5)
  --no-cuda            disables CUDA training
  --seed S             random seed (default: 1)
  --log-interval N     how many batches to wait before logging training status
  --save-model         whether to save the model at the end of training
  --dataset {full,g6}  which dataset to use
```

## Beam Search (Continual Prediction)
If we train the LSTM for continual prediction, we can use it as guide for 
generating new positives by beam search. The problem in this setting are the 
filters as they occur at the end of every sequence and have great impact on the 
label. We can eliminate the filters by training the LSTM to give answer to the 
question 

> Exists a filter for the sequence such that the label is positive?

The database can easily provide such data by something like

```SELECT n, m, k, com, MAX(maxent) FROM data GROUP BY n, m, k, com```

However, there might exist samples for which the right filter has just not been 
tried. The model will learn to classify such samples as negatives, although 
they are really positives. To estimate the rate at which these false negatives 
will occur in the database, we can draw a sample (e.g. of size 1000) for which 
we do an exhaustive search on the filters. 

## TODO
* throw out small-set samples
* continual prediction
* contribution analysis
* generalization to 4-partite setting possible?
* why not regress entanglement measure $`y \in [0,1] \subset \mathbb{R}`$ 
  instead of binary classification?
* can we compute the true SRV of the {0,0,0} (empty trigger) samples?
* stats about sequence lengths and component frequencies
* stats about SRV distribution to compute random classifier performance
* exact SRV computation and feedback loop
* what is a good criterion for interesting SRVs?

## Distribution over SRVs
n|m|k|neg|pos
---:| ---:| ---:| ---:| ---:
0 | 0 | 0 | 3831737 | 0
1 | 1 | 1 | 468531 | 0
2 | 2 | 1 | 443151 | 0
2 | 2 | 2 | 12565 | 9950
3 | 2 | 2 | 19193 | 13157
3 | 3 | 1 | 875376 | 0
3 | 3 | 2 | 227991 | 115665
3 | 3 | 3 | 34698 | 184978
4 | 2 | 2 | 440 | 34533
4 | 3 | 2 | 12669 | 2761
4 | 3 | 3 | 32519 | 14612
4 | 4 | 1 | 28907 | 0
4 | 4 | 2 | 15395 | 572
4 | 4 | 3 | 89931 | 4379
4 | 4 | 4 | 3627 | 0
5 | 3 | 2 | 769 | 30968
5 | 3 | 3 | 13886 | 40
5 | 4 | 2 | 1254 | 78590
5 | 4 | 3 | 5290 | 273303
5 | 4 | 4 | 1336 | 9
5 | 5 | 1 | 2301 | 0
5 | 5 | 2 | 1749 | 84
5 | 5 | 3 | 8934 | 12
5 | 5 | 4 | 4497 | 0
5 | 5 | 5 | 1514 | 0
6 | 3 | 3 | 1299 | 2106
6 | 4 | 2 | 24 | 1114
6 | 4 | 3 | 1680 | 252376
6 | 4 | 4 | 2892 | 259279
6 | 5 | 2 | 699 | 52510
6 | 5 | 3 | 4054 | 220098
6 | 5 | 4 | 617 | 0
6 | 5 | 5 | 948 | 0
6 | 6 | 1 | 27858 | 0
6 | 6 | 2 | 14755 | 327
6 | 6 | 3 | 4450 | 0
6 | 6 | 4 | 1564 | 0
6 | 6 | 5 | 2101 | 0
6 | 6 | 6 | 2759 | 0
7 | 4 | 3 | 55 | 11

n|m|k|neg|pos
---:| ---:| ---:| ---:| ---:
7 | 4 | 4 | 71 | 67
7 | 5 | 2 | 5 | 170
7 | 5 | 3 | 192 | 2312
7 | 5 | 4 | 169 | 5627
7 | 5 | 5 | 92 | 0
7 | 6 | 2 | 49 | 1584
7 | 6 | 3 | 2027 | 74294
7 | 6 | 4 | 220 | 1
7 | 6 | 5 | 217 | 0
7 | 6 | 6 | 262 | 0
7 | 7 | 1 | 53 | 0
7 | 7 | 2 | 161 | 243
7 | 7 | 3 | 767 | 0
7 | 7 | 4 | 1908 | 0
7 | 7 | 5 | 399 | 0
7 | 7 | 6 | 1467 | 0
7 | 7 | 7 | 327 | 0
8 | 4 | 4 | 1 | 0
8 | 5 | 3 | 102 | 0
8 | 5 | 4 | 35 | 61
8 | 5 | 5 | 39 | 1057
8 | 6 | 2 | 1 | 4
8 | 6 | 3 | 215 | 207
8 | 6 | 4 | 34 | 186
8 | 6 | 5 | 63 | 0
8 | 6 | 6 | 96 | 0
8 | 7 | 2 | 5 | 41
8 | 7 | 3 | 66 | 61
8 | 7 | 4 | 32 | 0
8 | 7 | 5 | 42 | 0
8 | 7 | 6 | 78 | 0
8 | 7 | 7 | 36 | 0
8 | 8 | 1 | 64 | 0
8 | 8 | 2 | 27 | 334
8 | 8 | 3 | 390 | 0
8 | 8 | 4 | 138 | 0
8 | 8 | 5 | 163 | 0
8 | 8 | 6 | 223 | 0
8 | 8 | 7 | 100 | 0
8 | 8 | 8 | 65 | 0

n|m|k|neg|pos
---:| ---:| ---:| ---:| ---:
9 | 5 | 4 | 2 | 0
9 | 5 | 5 | 3 | 0
9 | 6 | 3 | 14 | 1
9 | 6 | 4 | 8 | 27
9 | 6 | 5 | 8 | 6
9 | 6 | 6 | 67 | 0
9 | 7 | 3 | 8 | 4
9 | 7 | 4 | 6 | 1
9 | 7 | 5 | 11 | 0
9 | 7 | 6 | 50 | 0
9 | 7 | 7 | 12 | 0
9 | 8 | 2 | 0 | 2
9 | 8 | 3 | 4 | 0
9 | 8 | 4 | 1 | 0
9 | 8 | 5 | 16 | 0
9 | 8 | 6 | 39 | 0
9 | 8 | 7 | 9 | 0
9 | 8 | 8 | 15 | 0
9 | 9 | 1 | 9 | 0
9 | 9 | 2 | 8 | 2
9 | 9 | 3 | 59 | 0
9 | 9 | 4 | 58 | 0
9 | 9 | 5 | 67 | 0
9 | 9 | 6 | 202 | 0
9 | 9 | 7 | 69 | 0
9 | 9 | 8 | 40 | 0
9 | 9 | 9 | 43 | 0
10 | 5 | 5 | 1 | 0
10 | 6 | 6 | 30 | 487
10 | 7 | 4 | 1 | 0
10 | 7 | 5 | 2 | 5
10 | 7 | 6 | 5 | 0
10 | 7 | 7 | 4 | 0
10 | 8 | 4 | 1 | 0
10 | 8 | 5 | 6 | 0
10 | 8 | 6 | 3 | 0
10 | 8 | 7 | 2 | 0
10 | 8 | 8 | 5 | 0
10 | 9 | 3 | 1 | 0
10 | 9 | 5 | 3 | 0
10 | 9 | 6 | 16 | 0

n|m|k|neg|pos
---:| ---:| ---:| ---:| ---:
10 | 9 | 7 | 8 | 0
10 | 9 | 8 | 3 | 0
10 | 9 | 9 | 5 | 0
10 | 10 | 1 | 3 | 0
10 | 10 | 2 | 4 | 15
10 | 10 | 3 | 9 | 0
10 | 10 | 4 | 8 | 0
10 | 10 | 5 | 14 | 0
10 | 10 | 6 | 40 | 0
10 | 10 | 7 | 30 | 0
10 | 10 | 8 | 21 | 0
10 | 10 | 9 | 29 | 0
10 | 10 | 10 | 5 | 0
11 | 6 | 6 | 1 | 0
11 | 7 | 5 | 1 | 0
11 | 7 | 6 | 2 | 0
11 | 8 | 3 | 1 | 0
11 | 8 | 4 | 1 | 0
11 | 8 | 6 | 2 | 0
11 | 8 | 7 | 1 | 0
11 | 8 | 8 | 2 | 0
11 | 9 | 3 | 1 | 0
11 | 9 | 6 | 3 | 0
11 | 10 | 6 | 1 | 0
11 | 10 | 9 | 1 | 0
11 | 11 | 2 | 1 | 0
11 | 11 | 3 | 1 | 0
11 | 11 | 4 | 1 | 0
11 | 11 | 5 | 5 | 0
11 | 11 | 6 | 17 | 0
11 | 11 | 7 | 2 | 0
11 | 11 | 8 | 14 | 0
11 | 11 | 9 | 14 | 0
11 | 11 | 11 | 2 | 0
12 | 8 | 6 | 1 | 0
12 | 9 | 6 | 1 | 0
12 | 9 | 7 | 1 | 0
12 | 9 | 9 | 2 | 0
12 | 10 | 6 | 2 | 0
12 | 10 | 9 | 1 | 0

n|m|k|neg|pos
---:| ---:| ---:| ---:| ---:
12 | 11 | 2 | 1 | 0
12 | 11 | 9 | 1 | 0
12 | 12 | 3 | 3 | 0
12 | 12 | 5 | 1 | 0
12 | 12 | 6 | 18 | 0
12 | 12 | 7 | 2 | 0
12 | 12 | 8 | 1 | 0
12 | 12 | 9 | 6 | 0
12 | 12 | 12 | 2 | 0








