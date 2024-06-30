# Import necessary libraries
import streamlit as st
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from PIL import Image
import random
import os
from collections import Counter
import urllib.parse
import requests

# Set page configuration
st.set_page_config(layout="wide")

# Define the function to load embeddings
def load_embeddings():
    # URL of the embeddings file in Google Cloud Storage
    embeddings_url = 'https://storage.googleapis.com/watch_images_recommender/embeddings_classifier_family_v2.json'
    
    # Make a request to get the embeddings file
    response = requests.get(embeddings_url)
    
    # Check if the request was successful
    if response.status_code != 200:
        st.error(f"Failed to fetch embeddings file. Status code: {response.status_code}")
        st.stop()
    
    # Try to parse the JSON response
    try:
        embeddings_dict = response.json()
    except json.JSONDecodeError as e:
        st.error(f"Failed to decode JSON. Error: {str(e)}")
        st.stop()
    
    paths = list(embeddings_dict.keys())
    # Replace local paths with cloud paths
    bucket_name = "watch_images_recommender"  # Replace with your actual bucket name
    cloud_path_prefix = f"https://storage.googleapis.com/{bucket_name}/images/"
    local_path_prefix = "scraping_output/images/"  # This should match the local folder structure in your JSON
    # Update paths to be full URLs to the cloud storage and URL-encode the path part only
    paths = [cloud_path_prefix + urllib.parse.quote(path.replace(local_path_prefix, "")) for path in paths]
    
    embeddings = np.array([v['embedding'] for v in embeddings_dict.values()])
    brands = [v['brand'] for v in embeddings_dict.values()]
    families = [v['family'] for v in embeddings_dict.values()]

    return embeddings, paths, brands, families

# Call the function to load embeddings and initialize variables
embeddings, paths, brands, families = load_embeddings()

# Check if embeddings were loaded successfully
if embeddings is None or paths is None or brands is None or families is None:
    st.stop()

# Debugging output to verify paths
st.write("Sample paths:", paths[:5])

# Process data to get unique brands and create a brand-model dictionary
unique_brands = sorted(set(brands))
brand_model_dict = {}
for path, brand, family in zip(paths, brands, families):
    if brand not in brand_model_dict:
        brand_model_dict[brand] = []
    if family not in brand_model_dict[brand]:
        brand_model_dict[brand].append(family)

# Define and find k-NN
def find_knn(embeddings, n_neighbors=6):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(embeddings)
    return knn

knn = find_knn(embeddings)

# Define a query function for k-NN
def knn_query(knn, query_embedding, n_neighbors=6, min_distance=0.0001):
    distances, indices = knn.kneighbors([query_embedding], n_neighbors=n_neighbors)
    filtered_indices = []
    filtered_distances = []
    brand_count = Counter()
    for distance, index in zip(distances[0], indices[0]):
        if distance > min_distance and brand_count[brands[index]] < 2:
            filtered_indices.append(index)
            filtered_distances.append(distance)
            brand_count[brands[index]] += 1
    return np.array(filtered_distances), np.array(filtered_indices)

# Streamlit layout
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
                try:
                    st.image(selected_watch_path, caption=f'{watch_dict[selected_watch_path][0]} - {watch_dict[selected_watch_path][1]}')
                except Exception as e:
                    st.error(f"Failed to load image from URL: {selected_watch_path}. Error: {str(e)}")
    else:
        if 'random_watch' not in st.session_state or st.button('Find Another One'):
            selected_watch_path = random.choice(paths)
            st.session_state['random_watch'] = selected_watch_path
        else:
            selected_watch_path = st.session_state['random_watch']

        watch_dict = {path: (brand, family) for path, brand, family in zip(paths, brands, families)}
        try:
            st.image(selected_watch_path, caption=f"{watch_dict[selected_watch_path][0]} - {watch_dict[selected_watch_path][1]}")
        except Exception as e:
            st.error(f"Failed to load image from URL: {selected_watch_path}. Error: {str(e)}")

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
                try:
                    col.image(neighbor_path, caption=f'{watch_dict[neighbor_path][0]} - {watch_dict[neighbor_path][1]}')
                    col.write(f"Distance: {distance:.4f}")
                except Exception as e:
                    col.error(f"Failed to load image from URL: {neighbor_path}. Error: {str(e)}")
