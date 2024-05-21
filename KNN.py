import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from util_div import *
from util_model import *
from util_visualise_embeddings import *

def load_embeddings(embedding_file='embeddings.json'):
    with open(embedding_file, 'r') as f:
        embeddings_dict = json.load(f)
    paths = list(embeddings_dict.keys())
    embeddings = np.array([v['embedding'] for v in embeddings_dict.values()])
    brands = [v['brand'] for v in embeddings_dict.values()]
    families = [v['family'] for v in embeddings_dict.values()]
    return embeddings, paths, brands, families

def find_knn(embeddings, n_neighbors=5):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='euclidean')
    knn.fit(embeddings)
    return knn

def knn_query(knn, embeddings, query_embedding, n_neighbors=5):
    distances, indices = knn.kneighbors([query_embedding], n_neighbors=n_neighbors)
    return distances, indices

def visualize_results(query_path, neighbor_paths, distances):
    fig, axes = plt.subplots(1, len(neighbor_paths) + 1, figsize=(15, 5))

    # Display the query image
    query_img = Image.open(query_path)
    axes[0].imshow(query_img)
    axes[0].set_title("Query Image")
    axes[0].axis('off')

    # Display the nearest neighbor images
    for i, (neighbor_path, distance) in enumerate(zip(neighbor_paths, distances)):
        neighbor_img = Image.open(neighbor_path)
        axes[i + 1].imshow(neighbor_img)
        axes[i + 1].set_title(f"Neighbor {i + 1}\nDist: {distance:.2f}")
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()

def main_knn(query_path, weights_path='model_epoch_2.pth', exclude_family=False, exclude_brand=False):
    # Load embeddings from JSON file
    embeddings, paths, brands, families = load_embeddings()

    # Check if the query_path exists in paths
    if query_path not in paths:
        print(f"Query path '{query_path}' not found in embeddings.")
        return

    # Get the index of the query image
    query_index = paths.index(query_path)
    query_embedding = embeddings[query_index]
    query_brand = brands[query_index]
    query_family = families[query_index]

    # Filter out the embeddings based on exclude options
    valid_indices = list(range(len(paths)))

    if exclude_family:
        valid_indices = [i for i in valid_indices if families[i] != query_family]

    if exclude_brand:
        valid_indices = [i for i in valid_indices if brands[i] != query_brand]

    filtered_embeddings = embeddings[valid_indices]
    filtered_paths = [paths[i] for i in valid_indices]

    # Initialize k-NN
    knn = find_knn(filtered_embeddings, n_neighbors=5)

    # Query for the nearest neighbors of a specific watch
    distances, indices = knn_query(knn, filtered_embeddings, query_embedding, n_neighbors=5)
    
    # Display the results
    print(f"Query Watch: {query_path}")
    neighbor_paths = [filtered_paths[i] for i in indices[0]]
    neighbor_distances = distances[0]
    
    for i, (neighbor_path, distance) in enumerate(zip(neighbor_paths, neighbor_distances)):
        print(f"Neighbor {i + 1}: {neighbor_path} with distance {distance}")

    # Visualize the results
    visualize_results(query_path, neighbor_paths, neighbor_distances)

# Example of calling main_knn with a specific query image path
if __name__ == "__main__":
    query_path = "scraping_output/images/Norqain/adventure/Norqain_-_NN1001SC1CA_EB101-BGF_Adventure_Neverest_40_Stainless_Steel___Green___Black_-_Green_Flex_Case.jpg"  # Change to your actual image path
    main_knn(query_path, exclude_family=True, exclude_brand=False)  # Change exclude options as needed
