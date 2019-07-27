from sklearn.cluster import KMeans
from . import BaseClusterWithN

class KMeansHelper(BaseClusterWithN):
    def __init__(self, X, n_clusters, evaluate_by = None, init = "k-means++", n_jobs = None, scaled = True, random_state = 101):
        """
        Class to train and evaluate a KMeans Cluster Model
        """
        super().__init__(X=X, n_clusters=n_clusters, evaluate_by=evaluate_by, random_state=random_state)
        self.init = init
        self.n_jobs = n_jobs
        self.cluster_obj = KMeans(n_clusters=self.n_clusters, init=self.init, random_state=self.random_state, n_jobs=self.n_jobs)
        
                    
    
        
    
        
            
