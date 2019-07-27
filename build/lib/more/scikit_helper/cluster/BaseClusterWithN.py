import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from more import viz_helper as vh
from more import pandas_helper

class BaseClusterWithN:
    def __init__(self, X, n_clusters, evaluate_by=None, scaled=True, random_state=101):
        """
        Class to train and evaluate a Base Cluster Class with Number of Clusters Specified
        evaluate_by = column name to use to compare across the clusters eventually
        """
        self.evaluate_by = evaluate_by
        
        if (self.evaluate_by != None):
            self.evaluate_by_values = X[self.evaluate_by]
            self.X = X.helper.drop_columns([self.evaluate_by])
        else:
            self.X = X
                    
        #self.X = X.reset_index(drop=True)
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
            
    def evaluate_fit(self, metric = "silhoutte"):
        """
        Provides the Goodness of Fit Statistics for the clustering algorithm
        """
        if (metric == "silhoutte"):
            if (self.scaled):
                self.silhoutte_score = metrics.silhouette_score(self.X_scaled, self.labels, random_state= self.random_state)
            else:
                self.silhoutte_score = metrics.silhouette_score(self.X, self.labels, random_state= self.random_state)
        else:
            warnings.warn("Metric {} is not supported".format(metric))

        print("Silhouette Coefficient: {}".format(self.silhoutte_score))
        
    def merge_data_labels(self):
        if (self.evaluate_by == None):
            self.merged_data = pd.concat([self.X,pd.Series(self.labels,name='labels')], axis = 1)
            self.merged_scaled_data = pd.concat([self.X_scaled,pd.Series(self.labels,name='labels')], axis = 1)
        else:
            self.merged_data = pd.concat([self.X,pd.Series(self.labels,name='labels'), self.evaluate_by_values], axis = 1)
            self.merged_scaled_data = pd.concat([self.X_scaled,pd.Series(self.labels,name='labels'), self.evaluate_by_values], axis = 1)
        
    def cluster_obs_count(self):
        """
        Gives the number of observations in each cluster
        """
        return(self.merged_data.groupby('labels').count().transpose().iloc[0,:])
        
    def cluster_means(self):
        """
        Provides the means of the cluster features for each cluster
        If evaluate_by is set, then clusters will be sorted by the mean value of the "evaluate_by" column
        """
        if self.evaluate_by is not None:
            return(self.merged_data.groupby('labels').mean().sort_values(self.evaluate_by).transpose())
        else:
            return(self.merged_data.groupby('labels').mean().transpose())    
            
    def cluster_means_scaled(self):
        """
        Provides the means (scaled) of the cluster features for each cluster
        If evaluate_by is set, then clusters will be sorted by the mean value of the "evaluate_by" column
        """
        if self.evaluate_by is not None:
            return(self.merged_scaled_data.groupby('labels').mean().sort_values(self.evaluate_by).transpose())
        else:
            return(self.merged_scaled_data.groupby('labels').mean().transpose())
        
    def plot_parallel_coordinates(self, scaled=True, frac=0.05, figsize=(12,6), xrot=0):
        """
        Plot the parallel coordinate plots for the features in each cluster
        """
        if (scaled):
            vh.plot_parallel_coordinates(data = self.merged_scaled_data, by = 'labels', normalize=False, frac=frac, figsize=figsize, xrot=xrot)
        else:
            vh.plot_parallel_coordinates(data = self.merged_data, by = 'labels', normalize=False, frac=frac, figsize=figsize, xrot=xrot)
            
            
    def plot_headmap(self, scale_rows=True, cmap='viridis', figsize=(6,6)
                     , annot=False, valfmt="{x:.1f}", fontsize=12, fontweight="bold",textcolors=["white", "black"] ):
        """
        valfmt example: "{x:.1f}"
        """
        clmeans_df = self.cluster_means()
        clmeans_np = clmeans_df.to_numpy()
        if (scale_rows):
            cbarlabel = "Normalized Values"
        else:
            cbarlabel = "Values"
            
        fig, ax = plt.subplots(figsize=figsize)
        im, cbar = vh.heatmap(clmeans_np, row_labels=clmeans_df.index, col_labels=clmeans_df.columns
                              , ax=ax, scale_rows=scale_rows, cmap=cmap, cbarlabel=cbarlabel)
        
        if annot:
            vh.annotate_heatmap(im, valfmt=valfmt, size=fontsize, fontweight=fontweight,textcolors=textcolors)
            
        
    
        
        
            
        
        
    
        
            
