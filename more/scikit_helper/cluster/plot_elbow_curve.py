"""
Code modified from: https://github.com/reiinakano/scikit-plot
Modifications made to support running for Hierarchical Clustering
as well as support for plotting Silhoutte Score

The :mod:`scikitplot.cluster` module includes plots built specifically for
scikit-learn clusterer instances e.g. KMeans. You can use your own clusterers,
but these plots assume specific properties shared by scikit-learn estimators.
The specific requirements are documented per function.
"""

from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import clone
from joblib import Parallel, delayed

import warnings
import math
from scipy.cluster.vq import vq
from sklearn import metrics as mt


def plot_elbow_curve(clf,
                     X,
                     title='Elbow Plot',
                     cluster_ranges=None,
                     n_jobs=1,
                     show_second_metric=True,
                     second_metric="time",
                     ax=None, figsize=None,
                     title_fontsize="large",
                     text_fontsize="medium"):
    """Plots elbow curve of different values of K for KMeans clustering.
    Args:
        clf: Clusterer instance that implements ``fit``,``fit_predict``, and
            ``score`` methods, and an ``n_clusters`` hyperparameter.
            e.g. :class:`sklearn.cluster.KMeans` instance
        X (array-like, shape (n_samples, n_features)):
            Data to cluster, where n_samples is the number of samples and
            n_features is the number of features.
        title (string, optional): Title of the generated plot. Defaults to
            "Elbow Plot"
        cluster_ranges (None or :obj:`list` of int, optional): List of
            n_clusters for which to plot the explained variances. Defaults to
            ``range(1, 12, 2)``.
        n_jobs (int, optional): Number of jobs to run in parallel. Defaults to
            1.
        show_second_metric [Previously: show_cluster_time] (bool, optional):
            Should plot of second metric be included
        second_metric (string, optional)= Metric to ploy on second axis.
            Defaults to "time" for time it took to cluster for a particular K.
            Other options are 'silhoutte' for Silhoutte Score
            for a particular K
        ax (:class:`matplotlib.axes.Axes`, optional): The axes upon which to
            plot the curve. If None, the plot is drawn on a new set of axes.
        figsize (2-tuple, optional): Tuple denoting figure size of the plot
            e.g. (6, 6). Defaults to ``None``.
        title_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "large".
        text_fontsize (string or int, optional): Matplotlib-style fontsizes.
            Use e.g. "small", "medium", "large" or integer-values. Defaults to
            "medium".
    Returns:
        ax (:class:`matplotlib.axes.Axes`): The axes on which the plot was
            drawn.
    Example:
        >>> import scikitplot as skplt
        >>> kmeans = KMeans(random_state=1)
        >>> skplt.cluster.plot_elbow_curve(kmeans, cluster_ranges=range(1, 30))
        <matplotlib.axes._subplots.AxesSubplot object at 0x7fe967d64490>
        >>> plt.show()
        .. image:: _static/examples/plot_elbow_curve.png
           :align: center
           :alt: Elbow Curve
    """
    if (second_metric != 'time' and second_metric != 'silhoutte'):
        warnings.warn("\nSecond Metric is not allowed. Must be one of \
                      ['time','silhoutte'].\nYou entered '{}'. \
                      This will be reset to 'time'".format(second_metric))
        second_metric = 'time'

    if cluster_ranges is None:
        cluster_ranges = range(1, 12, 2)
    else:
        cluster_ranges = sorted(cluster_ranges)

    if not hasattr(clf, 'n_clusters'):
        raise TypeError('"n_clusters" attribute not in classifier. '
                        'Cannot plot elbow method.')

    tuples = Parallel(n_jobs=n_jobs)(delayed(_clone_and_score_clusterer)
                                     (clf, X, i, second_metric) for i in cluster_ranges)
    clfs, second_metric_score = zip(*tuples)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    ax.set_title(title, fontsize=title_fontsize)
    ax.plot(cluster_ranges, np.absolute(clfs), 'b*-')
    ax.grid(True)
    ax.set_xlabel('Number of clusters', fontsize=text_fontsize)
    ax.set_ylabel('Sum of Squared Errors', fontsize=text_fontsize)
    ax.tick_params(labelsize=text_fontsize)

    if show_second_metric:
        y_label = 'Clustering duration (seconds)'  # Default Value

        # Overwrite if needed
        # technically checking for time is not required but kept
        # in there for consistency (in case default changes later)
        if (second_metric == 'time'):
            y_label = 'Clustering duration (seconds)'
        elif (second_metric == 'silhoutte'):
            y_label = 'Silhoutte Score'

        ax2_color = 'green'
        ax2 = ax.twinx()
        ax2.plot(cluster_ranges, second_metric_score,
                 ':', alpha=0.75, color=ax2_color)
        ax2.set_ylabel(y_label,
                       color=ax2_color, alpha=0.75,
                       fontsize=text_fontsize)
        ax2.tick_params(colors=ax2_color, labelsize=text_fontsize)

    return ax


def _clone_and_score_clusterer(clf, X, n_clusters, second_metric):
    """Clones and scores clusterer instance.
    Args:
        # NOTE:
            In this modified implementation, the score method is not
            needed anymore since the SSE is calculated manually
        clf: Clusterer instance that implements ``fit``,``fit_predict``, and
            ``score`` methods, and an ``n_clusters`` hyperparameter.
            e.g. :class:`sklearn.cluster.KMeans` instance
        X (array-like, shape (n_samples, n_features)):
            Data to cluster, where n_samples is the number of samples and
            n_features is the number of features.
        n_clusters (int): Number of clusters
        second_metric (string): Second metric to return. First is always SSE
    Returns:
        score: Score of clusters
        second_metric: Number of seconds it took to fit cluster
    """
    start = time.time()
    clf = clone(clf)
    setattr(clf, 'n_clusters', n_clusters)
    clf.fit(X)

    labels = clf.labels_

    # Not every clustering algorithm returns the centers
    # (hence calculating manually)

    # centers = clf.cluster_centers_
    if (True):
        num_features = X.shape[1]
        centers = np.empty((num_features, 0))

        for i in range(len(set(labels))):
            single_cluster_means = X[labels == i].mean(
            ).to_numpy().reshape(num_features, 1)
            centers = np.concatenate((centers, single_cluster_means),
                                     axis=1, out=None)

        centers = centers.T

    # Calculating SSE
    # https://|stats.stackexchange.com/questions/81954/ssb-sum-of-squares-between-clusters
    partition, euc_distance_to_centroids = vq(obs=X, code_book=centers)

    TSS = np.sum((X-X.mean(0))**2)
    SSW = np.sum(euc_distance_to_centroids**2)
    #SSB = TSS - SSW

    # # The 'direct' way
    # B = []
    # c = scaled_data.mean(0)
    # for i in range(partition.max()+1):
    #     ci = X[partition == i].mean(0)
    #     B.append(np.bincount(partition)[i]*np.sum((ci - c)**2))
    # SSB_ = np.sum(B)

    second_metric_score = math.nan
    if (second_metric == 'time'):
        second_metric_score = time.time() - start
    elif (second_metric == 'silhoutte'):
        second_metric_score = mt.silhouette_score(X, labels, random_state=101)

    return SSW, second_metric_score
