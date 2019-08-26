from sklearn.cluster import KMeans
from . import BaseClusterWithN


class KMeansHelper(BaseClusterWithN):
    def __init__(self,
                 X,
                 n_clusters,
                 evaluate_by=None,
                 init="k-means++",
                 n_jobs=None,
                 scaled=True,
                 random_state=101,
                 space=None,
                 const_params=None,
                 loss_fun=None  # Hyperparameter Optimization
                 ):
        """
        Class to train and evaluate a KMeans Cluster Model
        X must be a dataframe (not numpy array)
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
        self.init = init
        self.n_jobs = n_jobs

        self.cluster_obj = KMeans(n_clusters=self.n_clusters,
                                  init=self.init,
                                  random_state=self.random_state,
                                  n_jobs=self.n_jobs)
