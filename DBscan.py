import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import hamming
from sklearn.metrics.pairwise import manhattan_distances

class DBscan():
    def __init__(self, eps, min_samples, similarity='hamming'):
        self.eps = eps
        self.min_samples = min_samples
        
        if similarity == 'manhattan':
            self.distance  = manhattan_distances
            #self.distance = lambda x, y: sum(abs(a - b) for a, b in zip(x, y))
        elif similarity == 'hamming':
            self.distance  = hamming
            #self.distance = lambda x, y: sum(a != b for a, b in zip(x, y))

    def calculate_matrix(self):
        self.dist_matrix = np.full((self.X.shape[0], self.X.shape[0]), float('inf'))
        for i in tqdm(range(self.X.shape[0])):
            for j in range(i+1, self.X.shape[0]):
                self.dist_matrix[i, j] = self.distance(self.X[i].reshape(1, -1), self.X[j].reshape(1, -1))
        

    def cluster(self, X, stop=False, dist_matrix=None):
        self.X = X
        if dist_matrix is not None:
            self.dist_matrix = dist_matrix
        else:
            print('Calculating distance matrix...')
            self.calculate_matrix()
        if stop:
            return
            
        self.cluster_labels = np.full(self.X.shape[0], -1)   
        cluster_label = 0
        for i in range(self.X.shape[0]):
            if self.cluster_labels[i] == -1:
                self.cluster_labels[i] = -2
                neighbors = self.get_neighbors(i)
                if len(neighbors) >= self.min_samples:
                    self.cluster_labels[i] = cluster_label 
                    self.expand_cluster(neighbors, cluster_label)
                    cluster_label += 1
        return self.cluster_labels 

    def get_neighbors(self, i):
        neighbors = [i]
        for j in range(i + 1, self.X.shape[0]):
            if self.dist_matrix[i, j] <= self.eps and self.cluster_labels[j] == -1:
                neighbors.append(j)
        return neighbors

    def expand_cluster(self, neighbors, cluster_label):
        shit = 0
        for j in neighbors:
            if self.cluster_labels[j] == -1:
                shit += 1
                self.cluster_labels[j] = cluster_label
                new_neighbors = self.get_neighbors(j)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors 