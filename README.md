# "More" Package

This is a helper package for a variety of functions as described in the Overview section below. 

# Installation

* For standard installation 
```
pip install more 
```
* For installing a particular version
```
pip install more==0.0.1b14
```

# Overview

This is a helper package for a variety of functions
1. Extension for Pandas Dataframe (Beta version released)
2. Extension for Visualization (Beta version released)
3. Extension for Scikit-learn (Beta version released)

# Examples
Check out the  [examples](https://github.com/ngupta23/more/tree/master/examples) folder for details on usage

# Version History

## 0.0.1b14

* Added more helper functions for pandas
* Made files PEP8 compliant

## 0.0.1b13

* Added Hyperoptimization helper class (using hyperopt library)
* Added ability to do hyperparameter optimization in Cluster Helper Class
* Updated heatmap method name in cluster helper from plot_heatmap to plot_means_heatmap


## 0.0.1b12

* Add functions for plotting elbow curves. 
    - Code modified from: https://github.com/reiinakano/scikit-plot
    - Modifications made to support running for Hierarchical Clustering as well as support for plotting Silhoutte Score 
* typo fixed in function name

## 0.0.1b10 & 0.0.1b11

* Updated Visualization Helper to add function to plot Heatmap
* Updated BaseClusterWithN to allow plotting of heatmap showing how "cluster feature means" vary between clusters
* 0.0.1b11 included a small bug fix in 0.0.1b10

## 0.0.1b9

* Updated KMeans and Agglomerative Cluster Helpers to include evaluate_by argument


## 0.0.1b8

* Added Cluster Helpers for KMeans and Agglomerative Clustering

## 0.0.1b7

* Added Cluster Helper for Gaussian Clusters
* Fixed Bug for plot_parallel_coordinates where it was not working correctly for a multi-level categorical label
* Fixed bug for pandas helper for describing categorical and numeric fields - Now it gives a warning if the dataframe does not have any categorical or numeric field when those respective describe functions are called.

