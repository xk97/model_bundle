#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `model_bundle` package."""


import unittest

from model_bundle.model_bundle import ModelBundle


class TestModelBundle(unittest.TestCase):
    """Tests for `model_bundle` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.datasets import make_blobs
        from model_bundle.model_params import model_list_pca, model_params_pca
        self.model_list, self.model_params = model_list_pca, model_params_pca
        X, y = make_blobs(n_samples=200, n_features=10, centers=3, random_state=42)
        X_rand = np.random.random(X[:4].shape)
        self.X_test = X_rand
        self.X, self.y = X, y
        # plt.scatter(X[:, 0], X[:, 1], marker='o')
        # plt.scatter(X_rand[:, 0], X_rand[:, 1], marker='*')
        # plt.show()

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_init(self):
        """Test workflow."""
        ModelBundle()

    def test_001_init(self):
        """Test workflow."""
        ModelBundle(model_list=self.model_list)

    def test_002_fit(self):
        """Test workflow."""
        models = ModelBundle(model_list=self.model_list)
        models.fit(self.X)
        # models.transform(self.X)

    def test_002_fit_cluster(self):
        """Test workflow."""
        from model_bundle.model_params import model_list_cluster, model_params_cluster
        models = ModelBundle(model_list=model_list_cluster)
        models.fit(self.X)
        # models.transform(self.X)

    def test_003_fit(self):
        """Test workflow."""
        models = ModelBundle(model_list=self.model_list)
        models.fit(self.X)
        models.transform(self.X)

    def test_000_model_params(self):
        # from model_bundle.model_params import *
        pass
    
    def test_do_nonthing(self):
        pass
    
if __name__ == "__main__":
    unittest.TestCase()