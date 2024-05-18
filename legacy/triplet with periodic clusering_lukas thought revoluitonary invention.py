import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import random
import json




# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# Prepare data from JSON
def prepare_dataframe(data):
    corrected_data = []
    for item in data:
        corrected_item = {key.replace(':', '').strip(): value for key, value in item.items()}
        corrected_data.append(corrected_item)
    return corrected_data

# Group images by brand
def group_images_by_brand(data):
    grouped = {}
    for item in data:
        brand = item['Brand']
        image_path = item['Image Path']
        if brand in grouped:
            grouped[brand].append(image_path)
        else:
            grouped[brand] = [image_path]
    return grouped

def compute_average_vectors(model, grouped_images, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    brand_vectors = {}
    model.eval()

    with torch.no_grad():
        for brand, images in grouped_images.items():
            vectors = []
            for image_path in images:
                try:
                    image = Image.open(image_path).convert('RGB')
                    image = transform(image).unsqueeze(0).to(device)
                    output = model(image)
                    output = output.view(output.size(0), -1)  # Flatten the output
                    vector = output.cpu().numpy()
                    vectors.append(vector)
                except FileNotFoundError:
                    print(f"File not found: {image_path}")
                    continue
                except Exception as e:
                    print(f"Error processing image {image_path}: {str(e)}")
                    continue

            if vectors:
                # Calculate the mean vector for the brand
                mean_vector = np.mean(np.vstack(vectors), axis=0)  # Stack and find mean
                brand_vectors[brand] = mean_vector

    return brand_vectors






def cluster_brands(brand_vectors, n_clusters=20):
    if not brand_vectors:
        return {}
    vectors = np.array(list(brand_vectors.values()))
    brands = list(brand_vectors.keys())
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(vectors)
    cluster_map = {brand: label for brand, label in zip(brands, labels)}
    return cluster_map



# Split Data
def split_data(grouped_images, train_frac=0.8, val_frac=0.1, test_frac=0.1):
    train_data = {}
    val_data = {}
    test_data = {}

    for brand, images in grouped_images.items():
        random.shuffle(images)
        train_end = int(len(images) * train_frac)
        val_end = train_end + int(len(images) * val_frac)
        train_data[brand] = images[:train_end]
        val_data[brand] = images[train_end:val_end]
        test_data[brand] = images[val_end:]

    print("data spli complited")
    return train_data, val_data, test_data

def generate_triplets_with_cluster_constraints(grouped_images, cluster_map, n_samples=10):
    triplets = []
    for brand in grouped_images.keys():
        if brand not in cluster_map:
            continue
        images = grouped_images[brand]
        current_cluster = cluster_map[brand]
        other_brands = [b for b in grouped_images.keys() if b in cluster_map and cluster_map[b] != current_cluster]

        for _ in range(n_samples):
            if len(images) < 2:
                continue
            anchor, positive = random.sample(images, 2)
            if not other_brands:
                continue
            negative_brand = random.choice(other_brands)
            negative = random.choice(grouped_images[negative_brand])
            triplets.append((anchor, positive, negative))

    return triplets



class WatchEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=600):
        super(WatchEmbeddingModel, self).__init__()
        # Load the pre-trained VGG16 model
        base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Remove the last classifier layer
        self.features = base_model.features
        self.classifier = nn.Sequential(
            *list(base_model.classifier.children())[:-1],  # Remove last layer
            nn.Linear(4096, embedding_size)  # Add new layer with output size `embedding_size`
        )

    def forward(self, x):
        x = self.features(x)  # Apply feature extractor
        x = x.view(x.size(0), -1)  # Flatten the features
        x = self.classifier(x)  # Classifier to embedding
        return x


class TripletDataset(Dataset):
    def __init__(self, triplets, transform=None):
        self.triplets = triplets
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, idx):
        path_anchor, path_pos, path_neg = self.triplets[idx]
        anchor = self.load_image(path_anchor)
        positive = self.load_image(path_pos)
        negative = self.load_image(path_neg)
        return anchor, positive, negative

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.triplets)



# Periodic Reclustering Function
def periodic_reclustering(model, data, current_epoch, cluster_freq, device):
    if current_epoch % cluster_freq == 0:
        print(f"Reclustering at epoch {current_epoch}")
        brand_vectors = compute_average_vectors(model, data, device)
        cluster_map = cluster_brands(brand_vectors)
        return cluster_map
    return None

def train(model, epochs, cluster_freq, device, grouped_images, subset_size=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.TripletMarginLoss(margin=1.0)
    
    # Reduce dataset for initial tests
    for brand, images in grouped_images.items():
        grouped_images[brand] = random.sample(images, min(len(images), subset_size))

    for epoch in range(epochs):
        # Recluster and regenerate triplets if necessary
        if epoch % cluster_freq == 0 or epoch == 0:
            print(f"Reclustering at epoch {epoch}")
            brand_vectors = compute_average_vectors(model, grouped_images, device)
            cluster_map = cluster_brands(brand_vectors)
            train_triplets = generate_triplets_with_cluster_constraints(grouped_images, cluster_map)

            # Create a DataLoader for triplets
            train_dataset = TripletDataset(train_triplets)
            train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)

        # Train over generated triplets
        model.train()
        total_loss = 0
        total_batches = 0

        for anchors, positives, negatives in train_loader:
            optimizer.zero_grad()
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchor_out, positive_out, negative_out = model(anchors), model(positives), model(negatives)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_batches += 1
            print(f"Batch Loss: {loss.item():.4f}")  # Print loss for each batch

        avg_loss = total_loss / total_batches if total_batches > 0 else 0
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")


# Supporting classes and functions remain the same

if __name__ == "__main__":
    print("Script started")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the custom embedding model
    model = WatchEmbeddingModel(embedding_size=600).to(device)
    print("Model loaded and transferred to device")

    # Proceed with data loading and preparation
    data = load_json_data('scraping_output/all_watches.json')
    data = prepare_dataframe(data)
    grouped_images = group_images_by_brand(data)

    # Initial clustering and training
    print("Starting to compute average vectors")
    brand_vectors = compute_average_vectors(model, grouped_images, device)
    print("Average vectors computed")

    print("Starting clustering")
    cluster_map = cluster_brands(brand_vectors)
    print("Clustering completed")


    # Using a small subset for quick testing
    train(model, epochs=10, cluster_freq=5, device=device, grouped_images=grouped_images, subset_size=10)
