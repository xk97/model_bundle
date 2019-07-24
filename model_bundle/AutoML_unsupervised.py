#https://github.com/PacktPublishing/Hands-On-Automated-Machine-Learning/blob/master/Chapter%2004/Automated_Machine_Learning.ipynb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MeanShift, estimate_bandwidth, SpectralClustering
from sklearn.decomposition import PCA
from hdbscan import HDBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
class Unsupervised_AutoML:

    def __init__(self, estimators=None, transformers=None):
        self.estimators = estimators
        self.transformers = transformers
        pass

    def fit_predict(self, X, y=None, scaler=True, decomposer={'name': PCA, 'args':[], 'kwargs': {'n_components': 2}}):
        """
        fit_predict will train given estimator(s) and predict cluster membership for each sample
        """

        shape = X.shape
        df_type = isinstance(X, pd.core.frame.DataFrame)

        if df_type:
            column_names = X.columns
            index = X.index

        if scaler == True:
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            if df_type:
                X = pd.DataFrame(X, index=index, columns=column_names)

        if decomposer is not None:
            X = decomposer['name'](*decomposer['args'], **decomposer['kwargs']).fit_transform(X)

            if df_type:
                if decomposer['name'].__name__ == 'PCA':
                    X = pd.DataFrame(X, index=index, columns=['component_' + str(i + 1) for i in
                                                              range(decomposer['kwargs']['n_components'])])
                else:
                    X = pd.DataFrame(X, index=index, columns=['component_1', 'component_2'])

            # if dimensionality reduction is applied, then n_components will be set accordingly in hyperparameter configuration
            for estimator in self.estimators:
                if 'n_clusters' in estimator['kwargs'].keys():
                    if decomposer['name'].__name__ == 'PCA':
                        estimator['kwargs']['n_clusters'] = decomposer['kwargs']['n_components']
                    else:
                        estimator['kwargs']['n_clusters'] = 2

        # This dictionary will hold predictions for each estimator
        predictions = []
        performance_metrics = {}

        for estimator in self.estimators:
            labels = estimator['estimator'](*estimator['args'], **estimator['kwargs']).fit_predict(X)
            estimator['estimator'].n_clusters_ = len(np.unique(labels))
            metrics = self._get_cluster_metrics(estimator['estimator'].__name__, estimator['estimator'].n_clusters_, X, labels, y)
            predictions.append({estimator['estimator'].__name__: labels})
            performance_metrics[estimator['estimator'].__name__] = metrics

        self.predictions = predictions
        self.performance_metrics = performance_metrics

        return predictions, performance_metrics

    # Printing cluster metrics for given arguments
    def _get_cluster_metrics(self, name, n_clusters_, X, pred_labels, true_labels=None):
        from sklearn.metrics import homogeneity_score, \
            completeness_score, \
            v_measure_score, \
            adjusted_rand_score, \
            adjusted_mutual_info_score, \
            silhouette_score

        print("""################## %s metrics #####################""" % name)
        if len(np.unique(pred_labels)) >= 2:

            silh_co = silhouette_score(X, pred_labels)

            if true_labels is not None:

                h_score = homogeneity_score(true_labels, pred_labels)
                c_score = completeness_score(true_labels, pred_labels)
                vm_score = v_measure_score(true_labels, pred_labels)
                adj_r_score = adjusted_rand_score(true_labels, pred_labels)
                adj_mut_info_score = adjusted_mutual_info_score(true_labels, pred_labels)

                metrics = {"Silhouette Coefficient": silh_co,
                           "Estimated number of clusters": n_clusters_,
                           "Homogeneity": h_score,
                           "Completeness": c_score,
                           "V-measure": vm_score,
                           "Adjusted Rand Index": adj_r_score,
                           "Adjusted Mutual Information": adj_mut_info_score}

                for k, v in metrics.items():
                    print("\t%s: %0.3f" % (k, v))

                return metrics

            metrics = {"Silhouette Coefficient": silh_co,
                       "Estimated number of clusters": n_clusters_}

            for k, v in metrics.items():
                print("\t%s: %0.3f" % (k, v))

            return metrics

        else:
            print("\t# of predicted labels is {}, can not produce metrics. \n".format(np.unique(pred_labels)))

    # plot_clusters will visualize the clusters given predicted labels
    def plot_clusters(self, estimator, X, labels, plot_kwargs):

        palette = sns.color_palette('deep', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

        plt.scatter(X[:, 0], X[:, 1], c=colors, **plot_kwargs)
        plt.title('{} Clusters'.format(str(estimator.__name__)), fontsize=14)
        plt.show()

    def plot_all_clusters(self, estimators, labels, X, plot_kwargs):

        fig = plt.figure()

        for i, algorithm in enumerate(labels):

            quotinent = np.divide(len(estimators), 2)

            # Simple logic to decide row and column size of the figure
            if isinstance(quotinent, int):
                dim_1 = 2
                dim_2 = quotinent
            else:
                dim_1 = np.ceil(quotinent)
                dim_2 = 3

            palette = sns.color_palette('deep',
                                        np.unique(algorithm[estimators[i]['estimator'].__name__]).max() + 1)
            colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in
                      algorithm[estimators[i]['estimator'].__name__]]

            plt.subplot(dim_1, dim_2, i + 1)
            plt.scatter(X[:, 0], X[:, 1], c=colors, **plot_kwargs)
            plt.title('{} Clusters'.format(str(estimators[i]['estimator'].__name__)), fontsize=8)

        plt.show()

if __name__ == '__main__':
    # Necessary for bandwidth
    # Make blobs will generate isotropic Gaussian blobs
    # You can play with arguments like center of blobs, cluster standard deviation
    from sklearn.datasets import make_blobs
    centers = [[2, 1], [-1.5, -1], [1, -1], [-2, 2]]
    cluster_std = [0.1, 0.1, 0.1, 0.1]
    cluster_std = [0.4, 0.5, 0.6, 0.5]
    X, y = make_blobs(n_samples=1000,
                        centers=centers,
                        cluster_std=cluster_std,
                        random_state=53)

    plot_kwargs = {'s': 12, 'linewidths': 0.1}
    plt.scatter(X[:, 0], X[:, 1], **plot_kwargs)
    plt.show()
    bandwidth = estimate_bandwidth(X, quantile=0.1, n_samples=100)

    estimators = [{'estimator': KMeans, 'args': (), 'kwargs': {'n_clusters': 5}},
                            {'estimator': DBSCAN, 'args': (), 'kwargs': {'eps': 0.3}},
                            {'estimator': AgglomerativeClustering, 'args': (), 'kwargs': {'n_clusters': 5, 'linkage': 'ward'}},
                            {'estimator': MeanShift, 'args': (), 'kwargs': {'cluster_all': False, "bandwidth": bandwidth, "bin_seeding": True}},
                            {'estimator': SpectralClustering, 'args': (), 'kwargs': {'n_clusters':5}},
                            {'estimator': HDBSCAN, 'args': (), 'kwargs': {'min_cluster_size':15}}]

    unsupervised_learner = Unsupervised_AutoML(estimators)

    predictions, performance_metrics = unsupervised_learner.fit_predict(X, y, decomposer=None)