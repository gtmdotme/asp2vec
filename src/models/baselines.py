import networkx as nx
import random
import numpy as np
from typing import List
from tqdm import tqdm

from embedder import embedder
from node2vec import Node2Vec
from gensim.models.word2vec import Word2Vec

class node2vec_model(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        # Create a graph
        if self.isDirected:
            self.graph = nx.DiGraph()
        else:
            self.graph = nx.Graph()
        self.graph.add_edges_from(self.train_edges)

        # pdb.set_trace()
        self.init_model = Node2Vec(
            self.graph, 
            dimensions=self.dim, 
            walk_length=self.path_length, 
            num_walks=self.num_walks_per_node, 
            p=1,
            q=1,
            workers=4,
            seed=self.seed,
        )

    def training(self):
        # Embed nodes
        self.model = Word2Vec(sentences=self.init_model.walks, window=self.window_size, 
                              vector_size=self.dim, seed=self.seed)
        # self.model = self.init_model.fit(window=self.window_size, min_count=1, batch_words=4)

        emb = np.zeros((self.num_nodes, self.dim))

        for i in self.graph.nodes():
            emb[i] = self.model.wv[str(i)]
            
        self.eval_link_prediction(emb)
        self.print_result(isFinal='Final')
        return
    
# not used
class deepwalk_model(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)

    def training(self):
        # train skip-gram model
        self.model = Word2Vec(sentences=self.walks, window=self.window_size, 
                              vector_size=self.dim, seed=self.seed)

        # fetch embeddings
        emb = np.zeros((self.num_nodes, self.dim))
        for i in self.G.nodes():
            emb[i] = self.model.wv[i]
        
        self.eval_link_prediction(emb)
        self.print_result(isFinal='Final')
        return
    
class DeepWalk:
    def __init__(self, window_size: int, embedding_size: int, walk_length: int, walks_per_node: int):
        """
        # adapted from https://towardsdatascience.com/exploring-graph-embeddings-deepwalk-and-node2vec-ee12c4c0d26d
        :param window_size: window size for the Word2Vec model
        :param embedding_size: size of the final embedding
        :param walk_length: length of the walk
        :param walks_per_node: number of walks per node
        """
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.walk_length = walk_length
        self.walk_per_node = walks_per_node

    def random_walk(self, g: nx.Graph, start: str, randGen: random.Random,
                    use_probabilities: bool = False) -> List[str]:
        """
        Generate a random walk starting on start
        :param g: Graph
        :param start: starting node for the random walk
        :param use_probabilities: if True take into account the weights assigned to each edge to select the next candidate
        :return:
        """
        walk = [start]
        for i in range(self.walk_length):
            neighbours = g.neighbors(walk[i])
            neighs = list(neighbours)
            if len(neighs) == 0:
                break
            if use_probabilities:
                probabilities = [g.get_edge_data(walk[i], neig)["weight"] for neig in neighs]
                sum_probabilities = sum(probabilities)
                probabilities = list(map(lambda t: t / sum_probabilities, probabilities))
                p = randGen.choice(neighs, p=probabilities)
            else:
                p = random.choice(neighs)
            walk.append(p)
        return walk

    def get_walks(self, g: nx.Graph, use_probabilities: bool = False, random_state: int = 0) -> List[List[str]]:
        """
        Generate all the random walks
        :param g: Graph
        :param use_probabilities:
        :return:
        """
        randGen = random.Random(random_state)
        random_walks = []
        for _ in range(self.walk_per_node):
            random_nodes = list(g.nodes)
            random.shuffle(random_nodes)
            for node in tqdm(random_nodes):
                random_walks.append(self.random_walk(g=g, start=node, randGen=randGen,
                                                     use_probabilities=use_probabilities))
        return random_walks