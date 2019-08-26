import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from more import viz_helper as vh
from more import pandas_helper
from more import hyperopt_helper as hh
from .plot_elbow_curve import plot_elbow_curve


class BaseClusterWithN:
    def __init__(self,
                 X,
                 n_clusters=2,
                 evaluate_by=None,
                 scaled=True,
                 random_state=101,
                 space=None,
                 const_params=None,
                 loss_fun=None):
        """
        Class to train and evaluate a Base Cluster Class with
            Number of Clusters Specified
        evaluate_by = column name to use to compare across the
                      clusters eventually
        """
        self.evaluate_by = evaluate_by

        if (self.evaluate_by is not None):
            self.evaluate_by_values = X[self.evaluate_by]
            self.X = X.helper.drop_columns([self.evaluate_by])
        else:
            self.X = X

        self.n_clusters = n_clusters

        self.scaled = scaled
        self.random_state = random_state

        self.space = space
        self.const_params = const_params
        self.loss_fun = loss_fun
        self.objective = None
        self.best_params = None

        self.cluster_obj = None  # Define in child class
        self.labels = None
        self.silhoutte_score = None
        self.merged_data = None
        self.merged_scaled_data = None
        self.columns = self.X.columns

        std_scl = StandardScaler()
        self.X_scaled = pd.DataFrame(std_scl.fit_transform(self.X),
                                     columns=self.columns)

    # Getters and Setters ----
    def set_model(self, model):
        self.cluster_obj = model

    # Training ----

    def train(self, n_clusters=None, merge=True):
        """
        Train the clustering method
        n_clusters:
            If specified, this will override the existing value.
            Useful when the value is determined after plotting elbow curve
        merge (Default = True)
            Should the data be merged with the labels.
            Recommended not to change to False right now since
            that functionality has not been tested.
        """
        if (n_clusters is not None):
            self.n_clusters = n_clusters
            setattr(self.cluster_obj, 'n_clusters', self.n_clusters)

        if (self.scaled):
            self.cluster_obj.fit(self.X_scaled)
        else:
            self.cluster_obj.fit(self.X)

        self._post_train_process(merge=merge)

        return(self)  # Allows to cascade methods

    # Basic Evaluation ----

    def evaluate_fit(self,
                     metric="silhoutte"):
        """
        Provides the Goodness of Fit Statistics for the clustering algorithm
        """
        if (metric == "silhoutte"):
            if (self.scaled):
                self.silhoutte_score = (
                    metrics.silhouette_score(
                        self.X_scaled,
                        self.labels,
                        random_state=self.random_state
                    )
                )
            else:
                self.silhoutte_score = (
                    metrics.silhouette_score(
                        self.X,
                        self.labels,
                        random_state=self.random_state
                    )
                )
        else:
            warnings.warn("Metric {} is not supported".format(metric))

        print("Silhouette Coefficient: {}".format(self.silhoutte_score))

    def cluster_obs_count(self):
        """
        Gives the number of observations in each cluster
        """
        return(self.merged_data.groupby(
            'labels').count().transpose().iloc[0, :])

    def cluster_means(self):
        """
        Provides the means of the cluster features for each cluster
        If evaluate_by is set, then clusters will be sorted by the mean value
        of the "evaluate_by" column
        """
        if self.evaluate_by is not None:
            return(self.merged_data.groupby(
                'labels').mean().sort_values(self.evaluate_by).transpose())
        else:
            return(self.merged_data.groupby('labels').mean().transpose())

    def cluster_means_scaled(self):
        """
        Provides the means (scaled) of the cluster features for each cluster
        If evaluate_by is set, then clusters will be sorted by the mean value
        of the "evaluate_by" column
        """
        if self.evaluate_by is not None:
            return(self.merged_scaled_data.groupby(
                'labels').mean().sort_values(self.evaluate_by).transpose())
        else:
            return(self.merged_scaled_data.groupby(
                'labels').mean().transpose())

    # Plotting ----

    def plot_parallel_coordinates(self,
                                  scaled=True,
                                  frac=0.05,
                                  figsize=(12, 6),
                                  xrot=0):
        """
        Plot the parallel coordinate plots for the features in each cluster
        """
        if (scaled):
            vh.plot_parallel_coordinates(data=self.merged_scaled_data,
                                         by='labels',
                                         normalize=False,
                                         frac=frac,
                                         figsize=figsize,
                                         xrot=xrot)
        else:
            vh.plot_parallel_coordinates(data=self.merged_data,
                                         by='labels',
                                         normalize=False,
                                         frac=frac,
                                         figsize=figsize,
                                         xrot=xrot)

    def plot_means_heatmap(self,
                           scale_rows=True,
                           cmap='viridis',
                           figsize=(6, 6),
                           annot=False,
                           valfmt="{x:.1f}",
                           fontsize=12,
                           fontweight="bold",
                           textcolors=["white", "black"]):
        """
        Always plots the with unscaled data irrespecive of what was
        used for training. This ensures, we maintain the original context.
        valfmt example: "{x:.1f}"
        """
        clmeans_df = self.cluster_means()
        clmeans_np = clmeans_df.to_numpy()
        if (scale_rows):
            cbarlabel = "Normalized Values"
        else:
            cbarlabel = "Values"

        fig, ax = plt.subplots(figsize=figsize)
        im, cbar = vh.heatmap(clmeans_np,
                              row_labels=clmeans_df.index,
                              col_labels=clmeans_df.columns,
                              ax=ax,
                              scale_rows=scale_rows,
                              cmap=cmap,
                              cbarlabel=cbarlabel)

        if annot:
            vh.annotate_heatmap(im,
                                valfmt=valfmt,
                                size=fontsize,
                                fontweight=fontweight,
                                textcolors=textcolors)

    def plot_elbow_curve(self,
                         cluster_ranges,
                         second_metric='time',
                         n_jobs=1,
                         figsize=(6, 6)):
        """
        n_jobs:
                Different from the one in the object that is used for training.
                This is because when calculating silhoute score can take up a
                lot of memory so it may be advisable to run it without
                parallelism. But training can still occur in parallel, hence
                this option to set n_jobs is provided.
        """

        if (self.scaled):
            # This plot_elbow_curve is not the same as self.plot_elbow_curve.
            # It is coming from the plot_elbow_curve.py file
            plot_elbow_curve(self.cluster_obj,
                             X=self.X_scaled,
                             cluster_ranges=cluster_ranges,
                             second_metric=second_metric,
                             n_jobs=n_jobs,
                             figsize=figsize)
        else:
            plot_elbow_curve(self.cluster_obj,
                             X=self.X,
                             cluster_ranges=cluster_ranges,
                             second_metric=second_metric,
                             n_jobs=n_jobs,
                             figsize=figsize)

        plt.show()

    # Hyperparameter Optimization using Bayesian Optimization ----

    def set_loss_function(self, loss_fun):
        self.loss_fun = loss_fun

    def set_const_params(self, const_params):
        self.const_params = const_params

    def set_space(self, space):
        self.space = space

    def train_best_model(self, max_evals=20):
        self._set_objective()
        hyperopt_helper = hh.HyperoptHelper(space=self.space,
                                            objective=self.objective,
                                            max_evals=max_evals,
                                            random_state=self.random_state)
        model, self.best_params = hyperopt_helper.train_best_model()

        self.set_model(model=model)
        self._post_train_process(merge=True)

        return(self)

    # Private Methods ----

    def _post_train_process(self, merge=True):
        self.labels = self.cluster_obj.labels_  # Set Labels

        if (merge):  # Merge data is neecessary
            self._merge_data_labels()

    def _merge_data_labels(self):
        if (self.evaluate_by is None):
            self.merged_data = pd.concat([self.X,
                                          pd.Series(self.labels,
                                                    name='labels')],
                                         axis=1)
            self.merged_scaled_data = pd.concat([self.X_scaled,
                                                 pd.Series(self.labels,
                                                           name='labels')],
                                                axis=1)
        else:
            self.merged_data = pd.concat([self.X,
                                          pd.Series(self.labels,
                                                    name='labels'),
                                          self.evaluate_by_values], axis=1)
            self.merged_scaled_data = pd.concat([self.X_scaled,
                                                 pd.Series(self.labels,
                                                           name='labels'),
                                                 self.evaluate_by_values],
                                                axis=1)

    def _set_objective(self):
        if (self.scaled):
            self.objective = hh.MyObjective(model=self.cluster_obj,
                                            X=self.X_scaled,
                                            y=None,
                                            const_params=self.const_params,
                                            loss_fun=self.loss_fun)
        else:
            self.objective = hh.MyObjective(model=self.cluster_obj,
                                            X=self.X,
                                            y=None,
                                            const_params=self.const_params,
                                            loss_fun=self.loss_fun)
