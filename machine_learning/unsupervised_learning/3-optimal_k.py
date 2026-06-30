#!/usr/bin/env python3
"""
Evaluates the optimal number of clusters for K-Means using
inertia (Elbow method) and Silhouette scores.
"""
from sklearn import metrics
K_Means = __import__('2-k_means').K_Means


def optimal_k(X, max_clusters, random_state):
    """
    Evaluates K-Means clustering quality.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        max_clusters (int): Maximum number of clusters to evaluate (>=2).
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (ks, inertias, silhouette_scores)
            - ks (list[int]): Evaluated cluster numbers.
            - inertias (list[float]): Inertia values for each cluster number.
            - silhouette_scores (list[float]): Silhouette scores for each k.
    """
    ks = []
    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = K_Means(X, k, random_state)

        ks.append(k)
        inertias.append(kmeans.inertia_)

        score = metrics.silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)

    return ks, inertias, silhouette_scores
