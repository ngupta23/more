# "More" Package

This is a helper package for a variety of functions as described in the Overview section below. 

# Installation

* For standard installation > pip install more 
* For installing a particular version >  pip install more==0.0.1b8

# Overview

This is a helper package for a variety of functions
1. Extension for Pandas Dataframe (Beta version released)
2. Extension for Visualization (Beta version released)
3. Extension for Scikit-learn (Beta version released)

# Examples
Check out the  [examples](https://github.com/ngupta23/more/tree/master/examples) folder for details on usage

# Version History

## 0.0.1b8

* Added Cluster Helpers for KMeans and Agglomerative Clustering

## 0.0.1b7

* Added Cluster Helper for Gaussian Clusters
* Fixed Bug for plot_parallel_coordinates where it was not working correctly for a multi-level categorical label
* Fixed bug for pandas helper for describing categorical and numeric fields - Now it gives a warning if the dataframe does not have any categorical or numeric field when those respective describe functions are called.

