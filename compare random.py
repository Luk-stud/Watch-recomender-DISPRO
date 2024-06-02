
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import random


def load_embeddings(embedding_file):
    with open(embedding_file, 'r') as f:
        embeddings_dict = json.load(f)
    paths = list(embeddings_dict.keys())
    embeddings = np.array([v['embedding'] for v in embeddings_dict.values()])
    brands = [v['brand'] for v in embeddings_dict.values()]
    families = [v['family'] for v in embeddings_dict.values()]
    return embeddings, paths, brands, families

def find_knn(embeddings, n_neighbors=5):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='l2')
    knn.fit(embeddings)
    return knn

def knn_query(knn, query_embedding, n_neighbors=5):
    distances, indices = knn.kneighbors([query_embedding], n_neighbors=n_neighbors)
    return distances, indices

def visualize_results(query_path, neighbor_paths_before, neighbor_paths_after, distances_before, distances_after):
    fig, axes = plt.subplots(2, len(neighbor_paths_before) + 1, figsize=(15, 10))

    # Display the query image
    query_img = Image.open(query_path)
    axes[0, 0].imshow(query_img)
    axes[0, 0].set_title("Query Image (Before)")
    axes[0, 0].axis('off')

    axes[1, 0].imshow(query_img)
    axes[1, 0].set_title("Query Image (After)")
    axes[1, 0].axis('off')

    # Display the nearest neighbor images before training
    for i, (neighbor_path, distance) in enumerate(zip(neighbor_paths_before, distances_before)):
        neighbor_img = Image.open(neighbor_path)
        axes[0, i + 1].imshow(neighbor_img)
        axes[0, i + 1].set_title(f"Neighbor {i + 1}\nDist: {distance:.2f}")
        axes[0, i + 1].axis('off')

    # Display the nearest neighbor images after training
    for i, (neighbor_path, distance) in enumerate(zip(neighbor_paths_after, distances_after)):
        neighbor_img = Image.open(neighbor_path)
        axes[1, i + 1].imshow(neighbor_img)
        axes[1, i + 1].set_title(f"Neighbor {i + 1}\nDist: {distance:.2f}")
        axes[1, i + 1].axis('off')

    plt.tight_layout()
    plt.show()

def main_knn(embedding_file_before, embedding_file_after, exclude_family=False, exclude_brand=False):
    # Load embeddings from both JSON files
    embeddings_before, paths_before, brands_before, families_before = load_embeddings(embedding_file_before)
    embeddings_after, paths_after, brands_after, families_after = load_embeddings(embedding_file_after)

    # Get 10 random watches from different brands
    unique_brands = list(set(brands_before))
    selected_brands = random.sample(unique_brands, 10)
    selected_indices = [brands_before.index(brand) for brand in selected_brands]

    for query_index in selected_indices:
        query_path = paths_before[query_index]
        query_embedding_before = embeddings_before[query_index]
        query_brand_before = brands_before[query_index]
        query_family_before = families_before[query_index]

        query_embedding_after = embeddings_after[query_index]
        query_brand_after = brands_after[query_index]
        query_family_after = families_after[query_index]

        # Filter out the embeddings based on exclude options
        valid_indices_before = list(range(len(paths_before)))
        valid_indices_after = list(range(len(paths_after)))

        if exclude_family:
            valid_indices_before = [i for i in valid_indices_before if families_before[i] != query_family_before]
            valid_indices_after = [i for i in valid_indices_after if families_after[i] != query_family_after]

        if exclude_brand:
            valid_indices_before = [i for i in valid_indices_before if brands_before[i] != query_brand_before]
            valid_indices_after = [i for i in valid_indices_after if brands_after[i] != query_brand_after]

        filtered_embeddings_before = embeddings_before[valid_indices_before]
        filtered_paths_before = [paths_before[i] for i in valid_indices_before]

        filtered_embeddings_after = embeddings_after[valid_indices_after]
        filtered_paths_after = [paths_after[i] for i in valid_indices_after]

        # Initialize k-NN
        knn_before = find_knn(filtered_embeddings_before, n_neighbors=5)
        knn_after = find_knn(filtered_embeddings_after, n_neighbors=5)

        # Query for the nearest neighbors of a specific watch
        distances_before, indices_before = knn_query(knn_before, query_embedding_before, n_neighbors=5)
        distances_after, indices_after = knn_query(knn_after, query_embedding_after, n_neighbors=5)

        # Display the results
        print(f"Query Watch: {query_path}")

        neighbor_paths_before = [filtered_paths_before[i] for i in indices_before[0]]
        neighbor_distances_before = distances_before[0]

        neighbor_paths_after = [filtered_paths_after[i] for i in indices_after[0]]
        neighbor_distances_after = distances_after[0]

        for i, (neighbor_path, distance) in enumerate(zip(neighbor_paths_before, neighbor_distances_before)):
            print(f"Before Training - Neighbor {i + 1}: {neighbor_path} with distance {distance}")

        for i, (neighbor_path, distance) in enumerate(zip(neighbor_paths_after, neighbor_distances_after)):
            print(f"After Training - Neighbor {i + 1}: {neighbor_path} with distance {distance}")

        # Visualize the results
        visualize_results(query_path, neighbor_paths_before, neighbor_paths_after, neighbor_distances_before, neighbor_distances_after)

# Example of calling main_knn with specific embedding files
if __name__ == "__main__":
    embedding_file_before = "embeddings_stock.json"  # Change to your actual embedding file path
    embedding_file_after = "embeddings_clasifier_vibrant_blaze_33.json"    # Change to your actual embedding file path
    main_knn(embedding_file_before, embedding_file_after, exclude_family=True, exclude_brand=True)  # Change exclude options as needed
