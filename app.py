import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from model import preprocessing, run_clustering, evaluate, get_visualization_data

st.title('Clustering App')

file = st.file_uploader('Upload your dataset (CSV)', type=['csv'])

if file is not None:
    df = pd.read_csv(file)
else:
    st.info('Por favor, sube un archivo CSV para comenzar.')
    st.stop()

st.subheader('Dataset Preview')
st.dataframe(df.head())

st.subheader('Feature Selection')
feature_mode = st.radio(
    'Select feature mode:',
    ['Manual Selection', 'PCA Reduction']
)

all_columns = df.columns.to_list()
selected_features = st.multiselect('Select features:', all_columns, default=all_columns)

if len(selected_features) < 2:
    st.warning('Select at least 2 features.')
    st.stop()

n_components = 2
if feature_mode == 'PCA Reduction':
    max_comp = min(len(selected_features), len(df))
    n_components = st.number_input('Number of PCA components (at least 2):', min_value=2, max_value=max_comp, value=2, step=1)

X = preprocessing(df, selected_features, feature_mode=feature_mode, n_components=n_components)

st.subheader('Model Selection & Hyperparameters')
algorithm = st.selectbox(
    'Select a clustering algorithm:',
    ['K-Means', 'K-Medoids', 'DBSCAN', 'Hierarchical Clustering']
)

kwargs = {}
if algorithm == 'K-Means':
    kwargs['n_clusters'] = st.slider('Number of clusters (n_clusters)', 2, 10, 3)

elif algorithm == 'K-Medoids':
    kwargs['n_clusters'] = st.slider('Number of clusters (n_clusters)', 2, 10, 3)
    kwargs['metric'] = st.selectbox('Metric', ['euclidean', 'manhattan', 'cosine'])

elif algorithm == 'DBSCAN':
    kwargs['eps'] = st.number_input('Epsilon (eps)', min_value=0.01, max_value=10.0, value=0.5, step=0.1)
    kwargs['min_samples'] = st.slider('Min samples', 1, 20, 5)

elif algorithm == 'Hierarchical Clustering':
    kwargs['n_clusters'] = st.slider('Number of clusters (n_clusters)', 2, 10, 3)
    kwargs['linkage'] = st.selectbox('Select linkage', ['ward', 'complete', 'average', 'single'])

labels = run_clustering(X, algorithm, **kwargs)
df['cluster'] = labels

st.subheader('Clustered Dataset')
st.dataframe(df)

st.subheader('Visualization')
X_vis = get_visualization_data(X)

fig, ax = plt.subplots(figsize=(8, 5))
scatter = ax.scatter(X_vis.iloc[:, 0], X_vis.iloc[:, 1], c=labels, cmap='viridis', alpha=0.7, edgecolors='k')
ax.set_xlabel(X_vis.columns[0])
ax.set_ylabel(X_vis.columns[1])
ax.set_title(f'Cluster Plot ({algorithm})')
plt.colorbar(scatter, ax=ax, label='Cluster Label')
st.pyplot(fig)

st.subheader('Evaluation Metrics')
silhouette, davies_bouldin = evaluate(X, labels)

if silhouette is not None and davies_bouldin is not None:
    st.write(f'Silhouette Score: {silhouette}')
    st.write(f'Davies-Bouldin Index: {davies_bouldin}')
else:
    st.warning('No fue posible calcular las métricas (se necesitan al menos 2 clusters distintos para la evaluación).')