# Preprocessing datasets
Preprocessing dataset for link prediction.

### Data format
- ````edges.txt````
  - ````node1````[tab]````node2````
  #### If you want to use your own dataset, make a directory and put ````edges.txt```` inside the directory.

### How to preprocess datasets
- For directed graph (ex. filmtrust)
  ````
  python create_dataset.py --input filmtrust --directed
  ````

- For undirected graph (ex. ppi)
  ````
  python create_dataset.py --input ppi
  ````

In our code, we preprocess the following datasets in this way:
```bash
$ python create_dataset.py --input 4area --seed 0
# Output:
Namespace(directed=False, input='4area', remove_percent=0.5, seed=0)
Reading graph from 4area/edges.txt
Create 4area dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges: 100%|████████████████████████████████████████████████████████████| 57508/57508 [04:59<00:00, 192.14it/s]
removed edges percentage:  0.47064756207831954
Size of train_edges: 30440
Size of train_negatives: 30440
Size of test_edges: 27066
Size of test_edges_neg: 27066
Saved to 4area/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input BlogCatalog --seed 0
# Output:
Namespace(directed=False, input='BlogCatalog', remove_percent=0.5, seed=0)
Reading graph from BlogCatalog/edges.txt
Create BlogCatalog dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges:  50%|█████████████████████████████▎                            | 168561/333983 [09:44<09:38, 285.79it/s]breaking
Pruning edges:  50%|█████████████████████████████▎                            | 168580/333983 [09:44<09:33, 288.61it/s]
removed edges percentage:  0.49999850291781317
Size of train_edges: 166992
Size of train_negatives: 166992
Size of test_edges: 166991
Size of test_edges_neg: 166991
Saved to BlogCatalog/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input ca-astroph --seed 0
# Output:
Namespace(directed=False, input='ca-astroph', remove_percent=0.5, seed=0)
Reading graph from ca-astroph/edges.txt
Create ca-astroph dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges:  53%|███████████████████████████████                           | 105316/197031 [08:16<06:18, 242.16it/s]breaking
Pruning edges:  53%|███████████████████████████████                           | 105320/197031 [08:16<07:12, 212.10it/s]
removed edges percentage:  0.500002537671737
Size of train_edges: 98515
Size of train_negatives: 98515
Size of test_edges: 98516
Size of test_edges_neg: 98516
Saved to ca-astroph/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input ca-hepth --seed 0
# Output:
Namespace(directed=False, input='ca-hepth', remove_percent=0.5, seed=0)
Reading graph from ca-hepth/edges.txt
Create ca-hepth dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges: 100%|████████████████████████████████████████████████████████████| 24827/24827 [00:56<00:00, 439.42it/s]
removed edges percentage:  0.4539815523422081
Size of train_edges: 13555
Size of train_negatives: 13555
Size of test_edges: 11271
Size of test_edges_neg: 11271
Saved to ca-hepth/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input ciaodvd --directed --seed 0
# Output
Namespace(directed=True, input='ciaodvd', remove_percent=0.5, seed=0)
Reading graph from ciaodvd/edges.txt
Create ciaodvd dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges:  66%|███████████████████████████████████████▋                    | 26473/40073 [00:25<00:14, 962.15it/s]breaking
Pruning edges:  66%|███████████████████████████████████████                    | 26535/40073 [00:25<00:12, 1044.83it/s]
removed edges percentage:  0.49998752277094305
Size of train_edges: 20037
Size of train_negatives: 20037
Size of test_edges: 20036
Size of test_edges_neg: 20036
Size of test_edges_neg_directed: 46079
Saved to ciaodvd/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input cora --seed 0
# Output:
Namespace(directed=False, input='cora', remove_percent=0.5, seed=0)
Reading graph from cora/edges.txt
Create cora dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges: 100%|█████████████████████████████████████████████████████████████| 5069/5069 [00:03<00:00, 1553.18it/s]
removed edges percentage:  0.35036496350364965
Size of train_edges: 3290
Size of train_negatives: 3290
Size of test_edges: 1776
Size of test_edges_neg: 1776
Saved to cora/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input epinions --directed --seed 0
# Output:
Namespace(directed=True, input='epinions', remove_percent=0.5, seed=0)
Reading graph from epinions/edges.txt
Create epinions dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges:  62%|███████████████████████████████████▏                     | 300276/487182 [1:10:26<38:56, 79.98it/s]breaking
Pruning edges:  62%|███████████████████████████████████▏                     | 300280/487182 [1:10:26<43:50, 71.04it/s]
removed edges percentage:  0.5
Size of train_edges: 243591
Size of train_negatives: 243591
Size of test_edges: 243591
Size of test_edges_neg: 243591
Size of test_edges_neg_directed: 518660
Saved to epinions/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input filmtrust --directed --seed 0
# Output:
Namespace(directed=True, input='filmtrust', remove_percent=0.5, seed=0)
Reading graph from filmtrust/edges.txt
Create filmtrust dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges: 100%|█████████████████████████████████████████████████████████████| 1604/1604 [00:00<00:00, 6095.36it/s]
removed edges percentage:  0.3403990024937656
Size of train_edges: 1055
Size of train_negatives: 1055
Size of test_edges: 546
Size of test_edges_neg: 546
Size of test_edges_neg_directed: 1179
Saved to filmtrust/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input flickr --seed 0
# Output:
Namespace(directed=False, input='flickr', remove_percent=0.5, seed=0)
Reading graph from flickr/edges.txt
Create flickr dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges:  52%|███████████████████████████▏                        | 3086901/5899882 [2:35:17<2:14:06, 349.58it/s]breaking
Pruning edges:  52%|███████████████████████████▏                        | 3087000/5899882 [2:35:17<2:21:29, 331.32it/s]
removed edges percentage:  0.5000101696949193
Size of train_edges: 2949881
Size of train_negatives: 2949881
Size of test_edges: 2950001
Size of test_edges_neg: 2950001
Saved to flickr/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input ppi --seed 0
# Output:
Namespace(directed=False, input='ppi', remove_percent=0.5, seed=0)
Reading graph from ppi/edges.txt
Create ppi dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges:  55%|████████████████████████████████▏                          | 21141/38705 [00:16<00:14, 1226.06it/s]breaking
Pruning edges:  55%|████████████████████████████████▎                          | 21210/38705 [00:16<00:13, 1274.90it/s]
removed edges percentage:  0.5000904275933342
Size of train_edges: 19349
Size of train_negatives: 19349
Size of test_edges: 19356
Size of test_edges_neg: 19356
Saved to ppi/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input wiki-vote --directed --seed 0
# Output:
Namespace(directed=True, input='wiki-vote', remove_percent=0.5, seed=0)
Reading graph from wiki-vote/edges.txt
Create wiki-vote dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges:  57%|█████████████████████████████████▊                         | 59426/103663 [01:47<01:34, 466.87it/s]breaking
Pruning edges:  57%|█████████████████████████████████▊                         | 59465/103663 [01:47<01:20, 551.96it/s]
removed edges percentage:  0.4999951766782748
Size of train_edges: 51832
Size of train_negatives: 51832
Size of test_edges: 51831
Size of test_edges_neg: 51831
Size of test_edges_neg_directed: 149640
Saved to wiki-vote/data_remove_percent_0.5_seed_0.pkl

$ python create_dataset.py --input wikipedia --seed 0
# Output:
Namespace(directed=False, input='wikipedia', remove_percent=0.5, seed=0)
Reading graph from wikipedia/edges.txt
Create wikipedia dataset (Remove percent: 0.5)
Generate dataset for link prediction
Pruning edges:  50%|██████████████████████████████                              | 46261/92517 [00:50<00:49, 933.73it/s]breaking
Pruning edges:  50%|██████████████████████████████                              | 46295/92517 [00:50<00:50, 924.37it/s]
removed edges percentage:  0.5000270220608104
Size of train_edges: 46256
Size of train_negatives: 46256
Size of test_edges: 46261
Size of test_edges_neg: 46261
Saved to wikipedia/data_remove_percent_0.5_seed_0.pkl
```

Once the above commands generate ````data_remove_percent_0.5_seed_0.pkl````, we are done with data preprocessing.

### Preprocessed data split into train-test

- ````data_remove_percent_0.5.pkl````
  - A dictionary containing the following keys
    - ````isDirected````: whether the graph is directed, ````index````: mapped node index (dictionary)
    - ````num_nodes````: number of nodes, ````remove_percent````: the percentage of test data
    - ````train_edges````: positive edges in the training data, ````train_edges_neg````: negative edges in the training data
    - ````test_edges````: positive edges in the test data, ````test_edges_neg````: negative edges in the test data


Also available [here](https://purdue0-my.sharepoint.com/:f:/g/personal/gchoudha_purdue_edu/Euo87684H0xFoy6cL3nGeYAB4P5uGNzUxyrVZfNuGkWC-A?e=wcE16j) to download and use.