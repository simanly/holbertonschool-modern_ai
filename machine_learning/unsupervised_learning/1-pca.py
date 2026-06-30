#!/usr/bin/env python3
"""
This module provides a function to perform Principal Component Analysis (PCA)
on tabular data using Scikit-learn.
"""
from sklearn import decomposition


def Apply_PCA(X, n_components, random_state):
    """
    Performs Principal Component Analysis (PCA) on tabular data.

    Args:
        X (numpy.ndarray): Tabular data of shape (n_samples, n_features).
        n_components (int | float | None): Number of principal components
            to keep. If a float between 0 and 1, it represents the minimum
            fraction of variance to preserve. If None, all are kept.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple: (X_pca, pca)
            - X_pca (numpy.ndarray): Data transformed into the principal
              component space.
            - pca (sklearn.decomposition.PCA): The fitted PCA instance.
    """
    pca = decomposition.PCA(n_components=n_components,
                            random_state=random_state)
    X_pca = pca.fit_transform(X)

    return X_pca, pca
