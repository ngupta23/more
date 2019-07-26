import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from more import viz_helper as vh

class BaseClusterWithN:
    def __init__(self, X, n_clusters, scaled = True, random_state = 101):
        """
        Class to train and evaluate a Base Cluster Class with Number of Clusters Specified
        """
        self.X = X.reset_index(drop=True)
        self.n_clusters = n_clusters
        self.scaled = scaled
        self.random_state = random_state
        self.cluster_obj = None # Define in child class
        self.labels = None
        self.silhoutte_score = None;
        self.merged_data = None
        self.merged_scaled_data = None
        self.columns = self.X.columns
        
        std_scl = StandardScaler()
        self.X_scaled = pd.DataFrame(std_scl.fit_transform(self.X), columns=self.columns)
                    
    def train(self, merge = True):
        """
        Train the clustering method
        """
        if (self.scaled):
            self.cluster_obj.fit(self.X_scaled)
        else:
            self.cluster_obj.fit(self.X)
            
        self.labels = self.cluster_obj.labels_
        
        if (merge):
            self.merge_data_labels()
            
        return(self) # Allows to cascade methods
            
    def evaluate(self, metric = "silhoutte"):
        """
        Provides the Goodness of Fit Statistics for the clustering algorithm
        """
        if (metric == "silhoutte"):
            if (self.scaled):
                self.silhoutte_score = metrics.silhouette_score(self.X_scaled, self.labels, random_state= self.random_state)
            else:
                self.silhoutte_score = metrics.silhouette_score(self.X, self.labels, random_state= self.random_state)
        else:
            warnings.warn("Metrix {} is not supported".format(metric))

        print("Silhouette Coefficient: {}".format(self.silhoutte_score))
        
    def merge_data_labels(self):
        self.merged_data = pd.concat([self.X,pd.Series(self.labels,name='labels')], axis = 1)
        self.merged_scaled_data = pd.concat([self.X_scaled,pd.Series(self.labels,name='labels')], axis = 1)
        
    def cluster_obs_count(self):
        """
        Gives the number of observations in each cluster
        """
        return(self.merged_data.groupby('labels').count().transpose().iloc[0,:])
        
    def cluster_means(self):
        """
        Provides the means of the cluster features for each cluster
        """
        return(self.merged_data.groupby('labels').mean().transpose())
        
    def cluster_means_scaled(self):
        """
        Provides the means (scaled) of the cluster features for each cluster
        """
        return(self.merged_data.groupby('labels').mean().transpose())
        
    def plot_parallel_coordinates(self, scaled=True, frac=0.05, figsize=(12,6), xrot=0):
        """
        Plot the parallel coordinate plots for the features in each cluster
        """
        if (scaled):
            vh.plot_parallel_coordinates(data = self.merged_scaled_data, by = 'labels', normalize=False, frac=frac, figsize=figsize, xrot=xrot)
        else:
            vh.plot_parallel_coordinates(data = self.merged_data, by = 'labels', normalize=False, frac=frac, figsize=figsize, xrot=xrot)
            
        
    
        
        
            
        
        
    
        
            
