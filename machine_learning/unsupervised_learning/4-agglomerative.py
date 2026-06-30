#!/usr/bin/env python3
"""
Provides a function to perform Agglomerative Hierarchical
clustering on tabular data using Scikit-learn, with optional PCA.
"""
from sklearn import cluster
from sklearn import metrics
Apply_PCA = __import__('1-pca').Apply_PCA


def Agglomerative_Clustering(X, n_clusters, random_state,
                             n_components, use_pca_data=True):
    """
    Performs Agglomerative hierarchical clustering on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_clusters (int): Number of clusters to form.
        random_state (int): Random seed for reproducibility (used in PCA).
        n_components (int): Number of PCA components to reduce the data to.
        use_pca_data (bool): Whether to apply PCA before clustering.

    Returns:
        tuple: (model, X_used, score)
            - model: Fitted Agglomerative Clustering instance.
            - X_used: Data used for fitting (PCA-reduced or original).
            - score: Silhouette score of the clustering (None if <= 1).
    """
    # 1. Dimensionality Reduction (Optional)
    if use_pca_data:
        X_used, _ = Apply_PCA(X, n_components, random_state)
    else:
        X_used = X

    # 2. Clustering (using Ward linkage)
    model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    model.fit(X_used)

    # 3. Evaluation
    if n_clusters > 1:
        score = metrics.silhouette_score(X_used, model.labels_)
    else:
        score = None

    return model, X_used, score
