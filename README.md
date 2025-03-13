# ClusterSpacingAnalysis
Python script that analyse the distances between clusters from an in-situ heating TEM movie


# Introduction
This script aims to provides useful quantitative information regarding the clusters' distances during an in situ heating TEM experiments.
The approaches used are: Euclidean Distance (ED) and Nearest Nieghbour (NN)

## Euclidean Distance
Frame is binarised in a way that the substrate is signal and clusters are backgound. The script calculates the ED map, then the ridge map and finally plot the distribution of distances thus obtained.

## Nearest Neighbour
Frame is binarised (cluster are signal now). The script find the contours of all the clusters and calculates the minimum distance between a cluster and its first (or first three) neighbours. 

