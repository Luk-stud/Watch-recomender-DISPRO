import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import os
import json
import numpy as np
from util_div import *
from util_model import *
from util_visualise_embeddings import *

class WatchDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_path

def create_embeddings(model, dataset, device, image_info):
    model.eval()
    embeddings_dict = {}
    
    with torch.no_grad():
        for idx in range(len(dataset)):
            image, img_path = dataset[idx]
            print(f"Processing image {img_path}...")
            image = image.to(device).unsqueeze(0)  # Add batch dimension

            # Debug: Check the image tensor being fed into the model
            print(f"Image tensor shape: {image.shape}")
            print(f"Image tensor stats - min: {image.min().item()}, max: {image.max().item()}, mean: {image.mean().item()}")

            output = model(image)
            embedding = output.cpu().numpy().flatten().tolist()  # Convert to list
            
            # Add brand and family information
            brand = image_info[img_path]['Brand']
            family = image_info[img_path]['Family']
            
            embeddings_dict[img_path] = {
                'embedding': embedding,
                'brand': brand,
                'family': family
            }

            # Debug: Print the embedding to check if it is changing
            print(f"Embedding for {img_path}: {embedding[:5]}...")  # Print first 5 elements of the embedding for debugging
    
    return embeddings_dict

def save_embeddings(embeddings_dict, output_file='embeddings.json'):
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f)
    print(f"Embeddings saved to {output_file}")

def main_knn(weights_path):
    seed_everything()
    data_path = 'data/watches_database_main.json'
    raw_data = load_json_data(data_path)
    processed_data = prepare_dataframe(raw_data)
    grouped_images = group_images_by_brand(processed_data)

    # Create a mapping from image paths to brand and family information
    image_info = {}
    for entry in raw_data:
        img_path = entry['Image Path']
        image_info[img_path] = {
            'Brand': entry.get('Brand:', entry.get('Brand', '')),
            'Family': entry.get('Family:', entry.get('Family', ''))
        }

    # Flatten the grouped images into a list of image paths
    image_paths = [img for brand_images in grouped_images.values() for img in brand_images]

    transform = transforms.Compose([
        transforms.Resize((256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = WatchDataset(image_paths, transform=transform)

    embedding_size = 30
    model = WatchEmbeddingModel(embedding_size=embedding_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Load weights if path provided
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded model weights from {weights_path}")
    else:
        print(f"Specified model weights not found at {weights_path}, exiting.")
        return

    # Create embeddings for all watches
    embeddings_dict = create_embeddings(model, dataset, device, image_info)
    
    # Save embeddings to a JSON file
    save_embeddings(embeddings_dict)

# Example of calling main_knn with a specific model path and query index
if __name__ == "__main__":
    model_weights_path = "model_epoch_1.pth"  # Change to your actual model path
    main_knn(model_weights_path)
