### Exploring Node Polysemy in Networks

### Project Report 
 - [PDF](./ProjectReport.pdf)

### Papers
- [ **Unsupervised Differentiable Multi-aspect Network Embedding (*KDD 2020*)** ](https://arxiv.org/abs/2006.04239)
- [ **node2vec: Scalable Feature Learning for Networks (*SIGKDD 2016*)**](https://arxiv.org/pdf/1607.00653.pdf)
- [ **DeepWalk: Online Learning of Social Representations (*KDD 2014*)**](https://arxiv.org/pdf/1403.6652.pdf)

### Requirements
- Python version: 3.9
- Pytorch version: 2.0
- [fastrand](https://github.com/lemire/fastrand) (Fast random number generation in Python)
- scikit-learn
- [Node2vec](https://github.com/eliorc/node2vec)

### How to Run
````
git clone https://github.com/gtmdotme/node_polysemy
cd node_polysemy/
conda create -n myenv python=3.9
source activate myenv
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -c anaconda pandas numpy scikit-learn networkx seaborn tqdm
pip install fastrand node2vec
cd src/
````

### Dataset
For instructions regarding data, please check [````data````](./data) directory. Most of the preprocessed datasets are already present and ready to use.

## Execute `asp2vec` on Filmtrust dataset
````
python main.py --embedder asp2vec --dataset filmtrust --isSoftmax --isGumbelSoftmax --dim 20 --num_aspects 5 --isReg --isInit
````

## Execute `node2vec` on Filmtrust dataset
````
python main.py --embedder node2vec --dataset filmtrust --dim 100
````

## Execute `deepwalk` on Filmtrust dataset
````
python main.py --embedder deepwalk --dataset filmtrust --dim 100
````


### Arguments
````--embedder````: name of the embedding method

````--dataset````: name of the dataset

````--isInit````: If ````True````, warm-up step is performed

````--iter_max````: maximum iteration

````--dim````: dimension size of a node

````--window_size````: window_size to determine the context

````--path_length````: lentgh of each random walk

````--num_neg````: number of negative samples

````--num_walks_per_node````: number of random walks starting from each node

````--lr````: learning rate

````--patience````: when to stop (early stop criterion)

````--isReg````: enable aspect regularization framework

````--reg_coef````: lambda in aspect regularization framework

````--threshold````: threshold for aspect regularization

````--isSoftmax````: enable softmax

````--isGumbelSoftmax````: enable gumbel-softmax

````--isNormalSoftmax````: enable conventional softmax

````--num_aspects````: number of predefined aspects (K)
