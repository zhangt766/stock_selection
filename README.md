Code for the Relational Stock Ranking (RSR) model and the Temporal Graph Convolution in our paper "Temporal Relational Ranking for Stock Prediction", [\[paper\]](https://arxiv.org/abs/1809.09441).

## Environment

Python 3.6 & Tensorflow > 1.3

## Data

All stock data, including Raw Data, Sequential Embedding, features from raw End-of-day data are under the [data] folder.

### Sequential Data

Raw data: files under the [yfinance] are the historical (30 months) End-of-day data (i.e., open, high, low, close prices and trading volume) of more than 8,000 stocks traded in US stock market collected from yfinance.

Processed data: [2024-05-01] is the dataset used to conducted experiments in our paper.

Relation data: files under [Relation] including Industry Relation and Wiki Relation
```
tar zxvf relation.tar.gz
```

### Industry Relation

Under the sector_industry folder, there are row relation file and binary encoding file (.npy) storing the industry relations between stocks in NASDAQ and NYSE.

### Wiki Relation

Under the wikidata folder, there are row relation file and binary encoding file (.npy) storing the Wiki relations between stocks in NASDAQ and NYSE.

## Code

### Pre-processing

|       Script       | Function |
|:------------------:| :-----------: |
|       eod.py       | To generate features from raw End-of-day data |
| sector_industry.py | Generate binary encoding of industry relation |
|        get_wiki_data.py         | Generate binary encoding of Wiki relation |

### Training
| Script | Function |
| :-----------: | :-----------: |
| rank_lstm.py | Train a model of Rank_LSTM |
| relation_rank_lstm.py | Train a model of Relational Stock Ranking |

## Reference:
https://github.com/fulifeng/Temporal_Relational_Stock_Ranking



