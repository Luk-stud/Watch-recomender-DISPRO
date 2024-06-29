import streamlit as st
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import random
import os
from collections import Counter

# Load embeddings and watch data
def load_embeddings(embedding_file='embeddings_classifier_family_v2.json'):
    with open(embedding_file, 'r') as f:
        embeddings_dict = json.load(f)
    paths = list(embeddings_dict.keys())
    embeddings = np.array([v['embedding'] for v in embeddings_dict.values()])
    brands = [v['brand'] for v in embeddings_dict.values()]
    families = [v['family'] for v in embeddings_dict.values()]
    return embeddings, paths, brands, families

embeddings, paths, brands, families = load_embeddings()

# Prepare unique list of brands and models
unique_brands = sorted(set(brands))
brand_model_dict = {}
for path, brand, family in zip(paths, brands, families):
    if brand not in brand_model_dict:
        brand_model_dict[brand] = []
    if family not in brand_model_dict[brand]:
        brand_model_dict[brand].append(family)

# Find k-NN
def find_knn(embeddings, n_neighbors=6):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(embeddings)
    return knn

knn = find_knn(embeddings, n_neighbors=6)

def knn_query(knn, query_embedding, n_neighbors=6, min_distance=0.0001):
    distances, indices = knn.kneighbors([query_embedding], n_neighbors=n_neighbors)
    filtered_indices = []
    filtered_distances = []
    brand_count = Counter()

    for distance, index in zip(distances[0], indices[0]):
        if distance > min_distance:
            brand = brands[index]
            if brand_count[brand] < 2:
                filtered_indices.append(index)
                filtered_distances.append(distance)
                brand_count[brand] += 1

    return np.array(filtered_distances), np.array(filtered_indices)

# Streamlit layout
st.set_page_config(layout="wide")
st.title('Watch Recommender System')

# Layout columns
left_col, right_col = st.columns([1, 2])

with left_col:
    st.header("Choose a Watch")
    
    selection_mode = st.radio("Selection Mode", ("Manual Selection", "Random"))
    
    selected_watch_path = None
    watch_dict = {}

    if selection_mode == "Manual Selection":
        selected_brand = st.selectbox('Select a Brand', unique_brands)
        if selected_brand:
            selected_model = st.selectbox('Select a Model', sorted(brand_model_dict[selected_brand]))
        else:
            selected_model = None

        if selected_model:
            watches = [(path, brand, family) for path, brand, family in zip(paths, brands, families) if family == selected_model]
            watch_paths = [path for path, _, _ in watches]
            watch_dict = {path: (brand, family) for path, brand, family in watches}
            selected_watch_path = st.selectbox('Select a Watch', watch_paths)

            if selected_watch_path:
                st.image(selected_watch_path, caption=f'{watch_dict[selected_watch_path][0]} - {watch_dict[selected_watch_path][1]}')
    else:
        if 'random_watch' not in st.session_state or st.button('Find Another One'):
            selected_watch_path = random.choice(paths)
            st.session_state['random_watch'] = selected_watch_path
        else:
            selected_watch_path = st.session_state['random_watch']

        watch_dict = {path: (brand, family) for path, brand, family in zip(paths, brands, families)}
        st.image(selected_watch_path, caption=f"{watch_dict[selected_watch_path][0]} - {watch_dict[selected_watch_path][1]}")

    if selected_watch_path:
        if st.button('Find Similar Watches'):
            query_index = paths.index(selected_watch_path)
            query_embedding = embeddings[query_index]

            distances, indices = knn_query(knn, query_embedding, n_neighbors=6)
            neighbor_paths = [paths[i] for i in indices]
            neighbor_distances = distances

            st.session_state['recommendations'] = list(zip(neighbor_paths, neighbor_distances))

with right_col:
    st.header("Recommended Watches")
    if 'recommendations' in st.session_state:
        recommendations = st.session_state['recommendations']
        watch_dict = {path: (brand, family) for path, brand, family in zip(paths, brands, families)}
        
        rows = len(recommendations) // 3 + int(len(recommendations) % 3 > 0)
        for row in range(rows):
            cols = st.columns(3)
            for col, (neighbor_path, distance) in zip(cols, recommendations[row*3:(row+1)*3]):
                col.image(neighbor_path, caption=f'{watch_dict[neighbor_path][0]} - {watch_dict[neighbor_path][1]}')
                col.write(f"Distance: {distance:.4f}")
