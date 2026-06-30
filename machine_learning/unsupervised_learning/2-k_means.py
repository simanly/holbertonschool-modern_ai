#!/usr/bin/env python3
"""
This module provides a function to perform K-Means clustering
on tabular data using Scikit-learn.
"""
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """
    Creates and fits a K-Means clustering model using Scikit-learn.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters to find.
        random_state (int): Random seed for reproducibility.

    Returns:
        sklearn.cluster.KMeans: K-Means instance fitted on the input data.
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)

    return kmeans
