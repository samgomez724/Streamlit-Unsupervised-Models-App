import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, pairwise_distances

class CustomKMedoids:
    def __init__(self, n_clusters=3, metric='euclidean', max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.metric = metric
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self, X):
        np.random.seed(self.random_state)
        D = pairwise_distances(X, metric=self.metric)
        n_samples = X.shape[0]

        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)

        for _ in range(self.max_iter):
            labels = np.argmin(D[:, medoid_indices], axis=1)
            new_medoid_indices = np.zeros(self.n_clusters, dtype=int)

            for i in range(self.n_clusters):
                cluster_indices = np.where(labels == i)[0]
                if len(cluster_indices) == 0:
                    new_medoid_indices[i] = medoid_indices[i]
                    continue
                sub_D = D[np.ix_(cluster_indices, cluster_indices)]
                best_idx = cluster_indices[np.argmin(sub_D.sum(axis=1))]
                new_medoid_indices[i] = best_idx

            if np.array_equal(medoid_indices, new_medoid_indices):
                break
            medoid_indices = new_medoid_indices

        return labels

def preprocessing(df, features, feature_mode='Manual Selection', n_components=2):
    X = df[features].copy()

    numerical = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns
    categorical = X.select_dtypes(include=["object", "category"]).columns

    if len(categorical) > 0:
        encoder = OrdinalEncoder()
        X[categorical] = encoder.fit_transform(X[categorical])

    if len(numerical) > 0:
        scaler = StandardScaler()
        X[numerical] = scaler.fit_transform(X[numerical])

    if feature_mode == 'PCA Reduction':
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        cols = [f'PCA_{i+1}' for i in range(n_components)]
        X = pd.DataFrame(X_reduced, columns=cols, index=df.index)

    return X

def run_clustering(X, algorithm, **kwargs):
    if algorithm == "K-Means":
        model = KMeans(n_clusters=kwargs.get("n_clusters", 3), random_state=42)
        labels = model.fit_predict(X)
    elif algorithm == "K-Medoids":
        model = CustomKMedoids(
            n_clusters=kwargs.get("n_clusters", 3),
            metric=kwargs.get("metric", "euclidean"),
            random_state=42
        )
        labels = model.fit_predict(X)
    elif algorithm == "DBSCAN":
        model = DBSCAN(eps=kwargs.get("eps", 0.5), min_samples=kwargs.get("min_samples", 5))
        labels = model.fit_predict(X)
    elif algorithm == "Hierarchical Clustering":
        model = AgglomerativeClustering(n_clusters=kwargs.get("n_clusters", 3), linkage=kwargs.get("linkage", "ward"))
        labels = model.fit_predict(X)
    else:
        raise ValueError("Algoritmo no soportado")

    return labels

def evaluate(X, labels):
    unique_labels = set(labels)
    if len(unique_labels - {-1}) < 2:
        return None, None

    try:
        silhouette = silhouette_score(X, labels)
        davies_bouldin = davies_bouldin_score(X, labels)
        return silhouette, davies_bouldin
    except Exception:
        return None, None

def get_visualization_data(X):
    if X.shape[1] == 2:
        return X.copy()
    else:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        return pd.DataFrame(X_vis, columns=['PC1', 'PC2'], index=X.index)