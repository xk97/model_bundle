from sklearn import cluster 
from sklearn.decomposition import PCA, KernelPCA, SparsePCA
from sklearn.preprocessing import StandardScaler

model_list_pca = [(PCA, {}), 
                (KernelPCA, {}),
                (SparsePCA, {}),
]
model_params_pca = {
    'PCA': {'n_components': 1, }, 
    'KernelPCA': {'n_components': 1, }, 
    'SparsePCA': {'n_components': 1, },
}

model_params_grid_pca = {
    'PCA': {'n_components': [1, ], }, 
    'KernelPCA': {'n_components': [1,], }, 
    'SparsePCA': {'n_components': [1, ], },
}

# model_list_cluster = [
#     (cluster.AgglomerativeClustering, {}),
#     (cluster.KMeans, {}),
#     (cluster.DBSCAN, {})
# ]

# model_params_cluster = {
#     ('AgglomerativeClustering', {'n_clusters': 2}),
#     ('KMeans', {'n_clusters': 2}),
#     ('DBSCAN', {'eps': 10, })
# }
