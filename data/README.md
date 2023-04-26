# Preprocessing datasets

### How to preprocess datasets
- Goto ````src/```` directory
  - For directed graph (ex. filmtrust)
    ````
    python create_dataset.py --input filmtrust --directed
    ````

  - For undirected graph (ex. ppi)
    ````
    python create_dataset.py --input ppi
    ````
- Once the above commands generate ````data_remove_percent_0.5.pkl````, we are done with data preprocessing.

### Data format
- ````edges.txt````
  - ````node1````[tab]````node2````
- ````data_remove_percent_0.5.pkl````
  - A dictionary containing the following keys
    - ````isDirected````: whether the graph is directed, ````index````: mapped node index (dictionary)
    - ````num_nodes````: number of nodes, ````remove_percent````: the percentage of test data
    - ````train_edges````: positive edges in the training data, ````train_edges_neg````: negative edges in the training data
    - ````test_edges````: positive edges in the test data, ````test_edges_neg````: negative edges in the test data
#### If you want to use your own dataset, make a directory and put ````edges.txt```` inside the directory.

### Preprocessing dataset with different seeds
```bash
$ python create_dataset.py --input filmtrust --directed --seed 0

python create_dataset.py --input 4area --directed --seed 0
python create_dataset.py --input BlogCatalog --seed 0
python create_dataset.py --input ca-astroph --directed --seed 0
python create_dataset.py --input ca-hepth --directed --seed 0
python create_dataset.py --input ciaodvd --directed --seed 0
python create_dataset.py --input cora --directed --seed 0
python create_dataset.py --input epinions --directed --seed 0
python create_dataset.py --input filmtrust --directed --seed 0
python create_dataset.py --input flickr --seed 0
python create_dataset.py --input ppi --seed 0
python create_dataset.py --input wiki-vote --directed --seed 0
python create_dataset.py --input wikipedia --seed 0
```
