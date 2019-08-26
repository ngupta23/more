from sklearn.cluster import AgglomerativeClustering
from . import BaseClusterWithN


class AgglomerativeHelper(BaseClusterWithN):
    def __init__(self,
                 X,
                 n_clusters,
                 evaluate_by=None,
                 linkage="ward",
                 scaled=True,
                 random_state=101,
                 space=None,
                 const_params=None,
                 loss_fun=None  # Hyperparameter Optimization
                 ):
        """
        Class to train and evaluate a Agglomerative (Hierarchical)
        Cluster Model. X must be a dataframe (not numpy array)
        TODO: Fix later to check and take care of automatically
        """
        super().__init__(X=X,
                         n_clusters=n_clusters,
                         evaluate_by=evaluate_by,
                         scaled=scaled,
                         random_state=random_state,
                         space=space,
                         const_params=const_params,
                         loss_fun=loss_fun)
        self.linkage = linkage
        self.cluster_obj = AgglomerativeClustering(n_clusters=self.n_clusters,
                                                   linkage=self.linkage)
