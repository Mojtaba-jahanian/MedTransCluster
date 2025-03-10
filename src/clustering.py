from sklearn.cluster import KMeans, DBSCAN
import numpy as np

class ImageClustering:
    def __init__(self, method='kmeans', n_clusters=5):
        self.method = method
        self.n_clusters = n_clusters
        self.model = self._initialize_model()
        
    def _initialize_model(self):
        if self.method == 'kmeans':
            return KMeans(n_clusters=self.n_clusters)
        elif self.method == 'dbscan':
            return DBSCAN(eps=0.5, min_samples=5)
            
    def fit_predict(self, features):
        return self.model.fit_predict(features)