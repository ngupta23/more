from sklearn.cluster import AgglomerativeClustering
from . import BaseClusterWithN

class AgglomerativeHelper(BaseClusterWithN):
    def __init__(self, X, n_clusters, linkage = "ward", scaled = True, random_state = 101):
        """
        Class to train and evaluate a Agglomerative (Hierarchical) Cluster Model
        """
        super().__init__(X=X, n_clusters=n_clusters, random_state = random_state)
        self.linkage = linkage
        self.cluster_obj = AgglomerativeClustering(n_clusters=self.n_clusters, linkage=self.linkage)
        
        
                    
    
        
    
        
            
