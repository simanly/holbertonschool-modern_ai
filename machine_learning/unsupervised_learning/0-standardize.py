#!/usr/bin/env python3
'''
Module for feature standardization using Scikit-learn
'''
from sklearn import preprocessing


def Standardize(X):
    '''
    Standardizes tabular data using
    Scikit-learn's StandardScaler
    '''
    scaler = preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled
