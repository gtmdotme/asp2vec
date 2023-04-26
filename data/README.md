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
python create_dataset.py --input 4area --seed 0
python create_dataset.py --input BlogCatalog --seed 0
python create_dataset.py --input ca-astroph --seed 0
python create_dataset.py --input ca-hepth --seed 0
python create_dataset.py --input ciaodvd --directed --seed 0
python create_dataset.py --input cora --seed 0
python create_dataset.py --input epinions --directed --seed 0
python create_dataset.py --input filmtrust --directed --seed 0
python create_dataset.py --input flickr --seed 0
python create_dataset.py --input ppi --seed 0
python create_dataset.py --input wiki-vote --directed --seed 0
python create_dataset.py --input wikipedia --seed 0
```

Once the above commands generate ````data_remove_percent_0.5_seed_0.pkl````, we are done with data preprocessing.

### Preprocessed data split into train-test

- ````data_remove_percent_0.5.pkl````
  - A dictionary containing the following keys
    - ````isDirected````: whether the graph is directed, ````index````: mapped node index (dictionary)
    - ````num_nodes````: number of nodes, ````remove_percent````: the percentage of test data
    - ````train_edges````: positive edges in the training data, ````train_edges_neg````: negative edges in the training data
    - ````test_edges````: positive edges in the test data, ````test_edges_neg````: negative edges in the test data

