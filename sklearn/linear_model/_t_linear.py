# Authors: Daniel Iong daniong@umich.edu

import numpy as np

from scipy import optimize

from ..base import BaseEstimator, RegressorMixin
from ._base import LinearModel


def _t_loss_and_gradient(beta, df, X, y, sample_weight=None):
    pass


class tLinearRegression(LinearModel, RegressorMixin, BaseEstimator):
    def __init__(self, warm_start=False, fit_intercept=True):
        self.warm_start = warm_start
        self.fit_intercept = fit_intercept

    def fit(self, X, y=None):
        
        X, y = self._validate_data(
            X, y, copy=False, y_numeric=True,
            dtype=[np.float64, np.float32])
        
        if self.warm_start and hasattr(self, 'coef_'):
            parameters = self.coef_
        else:
            if self.fit_intercept:
                parameters = np.zeros(X.shape[1] + 1)
            else:
                parameters = np.zeros(X.shape[1])
            parameters[0] = 1
        

    def predict(self, X, y=None):
        pass


# Test (remove from file later.)
from sklearn.datasets import make_regression
X, y, coef = make_regression(n_samples=500, n_features=5, noise=10.0, coef=True, random_state=0)

