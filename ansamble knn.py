from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
import random
import os

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

def concatenate_embeddings(embedding_files, output_file='concatenated_embeddings.json'):
    combined_embeddings_dict = {}
    
    for embedding_file in embedding_files:
        with open(embedding_file, 'r') as f:
            embeddings_dict = json.load(f)
        
        for img_path, data in embeddings_dict.items():
            if img_path not in combined_embeddings_dict:
                combined_embeddings_dict[img_path] = {
                    'embedding': [],
                    'brand': data['brand'],
                    'family': data['family']
                }
            combined_embeddings_dict[img_path]['embedding'].extend(data['embedding'])
    
    with open(output_file, 'w') as f:
        json.dump(combined_embeddings_dict, f)
    
    print(f"Concatenated embeddings saved to {output_file}")

def main_knn(embedding_files, exclude_family=False, exclude_brand=False):
    # Concatenate embeddings
    concatenate_embeddings(embedding_files)

    # Load concatenated embeddings
    concatenated_embeddings_file = 'concatenated_embeddings.json'
    embeddings, paths, brands, families = load_embeddings(concatenated_embeddings_file)

    # Get 10 random watches from different brands
    unique_brands = list(set(brands))
    selected_brands = random.sample(unique_brands, 10)
    selected_indices = [brands.index(brand) for brand in selected_brands]

    for query_index in selected_indices:
        query_path = paths[query_index]
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
        distances, indices = knn_query(knn, query_embedding, n_neighbors=5)

        # Display the results
        print(f"Query Watch: {query_path}")

        neighbor_paths = [filtered_paths[i] for i in indices[0]]
        neighbor_distances = distances[0]

        for i, (neighbor_path, distance) in enumerate(zip(neighbor_paths, neighbor_distances)):
            print(f"Neighbor {i + 1}: {neighbor_path} with distance {distance}")

        # Visualize the results
        visualize_results(query_path, neighbor_paths, neighbor_paths, neighbor_distances, neighbor_distances)

# Example of calling main_knn with specific embedding files
if __name__ == "__main__":
    embedding_files = ["embeddings_clasifier_vague_leaf_7_family.json", "embeddings_clasifier_prime_voertex_34.json"]  # List of embedding files
    main_knn(embedding_files, exclude_family=True, exclude_brand=False)  # Change exclude options as needed
