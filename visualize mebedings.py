
import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import pickle
from util_div import *
from util_model import *
from util_visualise_embeddings import *
import torch.nn.functional as F

def initialize_model(model, device, load_weights=False, model_path=None):
    """ Initialize the model, optionally loading weights. """
    if load_weights and model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Main execution logic to load the model and visualize embeddings
if __name__ == "__main__":
    # Initialize a wandb run
    wandb.init(project='visualization_project', entity='DISPRO2', config={"task": "Embedding Visualization"})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_size = 200
    model = WatchEmbeddingModel(embedding_size=embedding_size)

    # Initialize model without loading pre-trained weights
    loaded_model = initialize_model(model, device, load_weights=False)

    # Assuming you already have the other necessary data and functions imported
    data_path = 'data/watches_database_main.json'
    raw_data = load_json_data(data_path)
    processed_data = prepare_dataframe(raw_data)
    grouped_images = group_images_by_brand(processed_data)
    val_brands = ['Tissot', 'Omega', 'Blancpain', 'Hamilton']

    transform = transforms.Compose([
        transforms.Resize((256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Visualize embeddings for validation brands
    visualize_brand_embeddings(loaded_model, grouped_images, device, val_brands, transform)

    # Finish the wandb run
    wandb.finish()

