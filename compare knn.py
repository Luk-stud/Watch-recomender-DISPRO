import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class WatchClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(WatchClassificationModel, self).__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def load_model(model_path, num_classes):
    model = WatchClassificationModel(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Remove the classification head
    model.base_model.classifier[1] = nn.Identity()
    return model

def generate_embedding(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        embedding = model(image).squeeze().numpy()  # Generate and convert to numpy
    return embedding

def load_embeddings(embedding_file):
    with open(embedding_file, 'r') as f:
        embeddings_dict = json.load(f)
    paths = list(embeddings_dict.keys())
    embeddings = np.array([v['embedding'] for v in embeddings_dict.values()])
    brands = [v['brand'] for v in embeddings_dict.values()]
    families = [v['family'] for v in embeddings_dict.values()]
    return embeddings, paths, brands, families

def find_knn(embeddings, n_neighbors=20):
    knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
    knn.fit(embeddings)
    return knn

def knn_query(knn, query_embedding, n_neighbors=20):
    distances, indices = knn.kneighbors([query_embedding], n_neighbors=n_neighbors)
    return distances, indices

def visualize_results(query_path, neighbor_paths, distances):
    fig, axes = plt.subplots(1, len(neighbor_paths) + 1, figsize=(15, 10))

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

def filter_by_family(paths, distances, families):
    unique_families = set()
    filtered_paths = []
    filtered_distances = []
    for path, distance, family in zip(paths, distances, families):
        if family not in unique_families:
            unique_families.add(family)
            filtered_paths.append(path)
            filtered_distances.append(distance)
        if len(filtered_paths) == 5:  # Ensure we only keep up to 5 results
            break
    return filtered_paths, filtered_distances

def filter_by_brand(paths, distances, brands):
    unique_brands = set()
    filtered_paths = []
    filtered_distances = []
    for path, distance, brand in zip(paths, distances, brands):
        if brand not in unique_brands:
            unique_brands.add(brand)
            filtered_paths.append(path)
            filtered_distances.append(distance)
        if len(filtered_paths) == 5:  # Ensure we only keep up to 5 results
            break
    return filtered_paths, filtered_distances

def main_knn(embedding_file, input_image_path, model_path, num_classes, exclude_family=False, exclude_brand=False, filter_by_brand_family='family'):
    # Load embeddings from the JSON file
    embeddings, paths, brands, families = load_embeddings(embedding_file)

    # Load the model and define the transform
    model = load_model(model_path, num_classes)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Generate embedding for the input image
    query_embedding = generate_embedding(model, input_image_path, transform)

    # Initialize k-NN with more neighbors than needed to allow for filtering
    knn = find_knn(embeddings, n_neighbors=20)

    # Query for the nearest neighbors of a specific watch
    distances, indices = knn_query(knn, query_embedding, n_neighbors=20)

    # Retrieve paths and distances for nearest neighbors
    neighbor_paths = [paths[i] for i in indices[0]]
    neighbor_distances = distances[0]
    neighbor_families = [families[i] for i in indices[0]]
    neighbor_brands = [brands[i] for i in indices[0]]

    # Filter the results to show only one neighbor per family or brand
    if filter_by_brand_family == 'family':
        neighbor_paths, neighbor_distances = filter_by_family(neighbor_paths, neighbor_distances, neighbor_families)
    elif filter_by_brand_family == 'brand':
        neighbor_paths, neighbor_distances = filter_by_brand(neighbor_paths, neighbor_distances, neighbor_brands)

    # Ensure we only keep up to 5 results
    neighbor_paths = neighbor_paths[:5]
    neighbor_distances = neighbor_distances[:5]

    # Display the results
    print(f"Query Watch: {input_image_path}")

    for i, (neighbor_path, distance) in enumerate(zip(neighbor_paths, neighbor_distances)):
        print(f"Neighbor {i + 1}: {neighbor_path} with distance {distance}")

    # Visualize the results
    visualize_results(input_image_path, neighbor_paths, neighbor_distances)

# Example of calling main_knn with specific embedding files
if __name__ == "__main__":
    embedding_file = "embeddings_classifier_family.json"  # Change to your actual embedding file path
    input_image_path = "C:/Users/bozov/OneDrive/Desktop/lorier.jpg"  # Change to your actual input image path
    model_path = "best_family_model.pth"  # Change to your actual model path
    num_classes = 295  # Change to the actual number of classes in your model
    main_knn(embedding_file, input_image_path, model_path, num_classes, exclude_family=False, exclude_brand=False, filter_by_brand_family='family')  # Change exclude options as needed
