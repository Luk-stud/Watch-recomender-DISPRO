import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from PIL import Image
import os
import random
import pickle

from torchvision.models import vgg16, VGG16_Weights
from util_div import *
from util_model import *
from util_visualise_embeddings import *


def stratified_split_data(grouped_images, val_brands=None, train_size=0.8, test_size=0.1, min_samples=2, random_state=None):
    print("spliting data...")
    # Filter out classes with fewer than min_samples
    filtered_grouped_images = {brand: images for brand, images in grouped_images.items() if len(images) >= min_samples}
    brands = list(filtered_grouped_images.keys())
    
    # Populate train and validation data dictionaries
    train_data = {brand: images for brand, images in filtered_grouped_images.items() if brand not in val_brands}
    val_data = {brand: images for brand, images in filtered_grouped_images.items() if brand in val_brands}
    return train_data, val_data
    
class CollectionDataset(Dataset):
    def __init__(self, collection_data, transform=None):
        self.images = []
        self.labels = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((400)),
            transforms.CenterCrop(size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        for collection in collection_data:
            collection_name = collection["Collection"]
            for watch in collection["Watches"]:
                image_path = watch["Extra Details"]["Image Path"]
                self.images.append(image_path)
                self.labels.append(collection_name)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, label


class PairDataset(Dataset):
    def __init__(self, grouped_images, transform=None, sim_ratio=0.5, dataset_type="train", save_dir="saved_pairs", max_pairs=None):
        self.grouped_images = grouped_images
        self.transform = transform or transforms.Compose([
            transforms.Resize((400)),
            transforms.CenterCrop(size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.sim_ratio = sim_ratio
        self.max_pairs = max_pairs
        self.save_dir = os.path.join(save_dir, dataset_type)
        self.pairs_filename = os.path.join(self.save_dir, "pairs.pkl")
        
        if not os.path.exists(self.pairs_filename):
            self.generate_pairs()
            self.save_pairs()
        else:
            self.load_pairs()
        self.apply_max_pairs()

    def generate_pairs(self):
        self.image_pairs = []
        self.labels = []
        self.weights = []
        brands = list(self.grouped_images.keys())
        
        for brand, images in self.grouped_images.items():
            num_similar = int(len(images) * (len(images) - 1) / 2 * self.sim_ratio)
            num_dissimilar = num_similar * (1 / self.sim_ratio - 1)

            for _ in range(num_similar):
                i, j = np.random.choice(len(images), 2, replace=False)
                self.image_pairs.append((images[i], images[j]))
                self.labels.append(1)
                self.weights.append(1.0)

            other_brands = [b for b in brands if b != brand]
            for _ in range(int(num_dissimilar)):
                other_brand = np.random.choice(other_brands)
                other_images = self.grouped_images[other_brand]
                img1 = np.random.choice(images)
                img2 = np.random.choice(other_images)
                self.image_pairs.append((img1, img2))
                self.labels.append(0)
                self.weights.append(0.5)

    def load_pairs(self):
        with open(self.pairs_filename, 'rb') as f:
            self.image_pairs, self.labels, self.weights = pickle.load(f)

    def apply_max_pairs(self):
        if self.max_pairs is not None and len(self.image_pairs) > self.max_pairs:
            self.image_pairs = self.image_pairs[:self.max_pairs]
            self.labels = self.labels[:self.max_pairs]
            self.weights = self.weights[:self.max_pairs]

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1, img2 = Image.open(img1_path).convert('RGB'), Image.open(img2_path).convert('RGB')
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        return img1, img2, label, weight

    def __len__(self):
        return len(self.image_pairs)
    
    def save_pairs(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.pairs_filename, 'wb') as f:
            pickle.dump((self.image_pairs, self.labels, self.weights), f)

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, loss_scale=3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_scale = loss_scale  # A scaling factor for the loss

    def forward(self, output1, output2, labels, weights):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_similar = (1 - labels) * weights * (euclidean_distance ** 2)
        loss_dissimilar = labels * weights * (self.margin - euclidean_distance).clamp(min=0) ** 2
        contrastive_loss = torch.mean(loss_similar + loss_dissimilar)
        return contrastive_loss * self.loss_scale  # Scale the computed loss


def train(model, train_loader, train_dataset, optimizer, criterion, device, epochs, val_loader, log_interval=50):
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1} starts...")

        for i, (img1, img2, labels, weights) in enumerate(train_loader):
            img1, img2 = img1.to(device), img2.to(device)
            labels, weights = labels.to(device), weights.to(device)
            optimizer.zero_grad()
            output1 = model(img1)
            output2 = model(img2)
            loss = criterion(output1, output2, labels, weights)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % log_interval == 0:
                print(f"At batch {i+1}, loss is {loss.item()}")
                wandb.log({"Epoch": epoch + 1, "Batch Training Loss": loss.item(), "Step": i})

        val_loss = validate(model, val_loader, criterion, device)
        wandb.log({"Epoch": epoch + 1, "Validation Loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)
            wandb.save(model_path)  # Save the model checkpoint to wandb

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

    return model

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img1, img2, labels, weights in val_loader:
            img1, img2 = img1.to(device), img2.to(device)
            labels, weights = labels.to(device), weights.to(device)
            output1 = model(img1)
            output2 = model(img2)
            loss = criterion(output1, output2, labels, weights)
            total_loss += loss.item()
    return total_loss / len(val_loader)

def check_for_saved_pairs(directory, dataset_type):
    """ Check if the saved pairs data for a specific dataset type exists and log the status. """
    save_dir = os.path.join(directory, dataset_type)
    pairs_filename = os.path.join(save_dir, "pairs.pkl")
    if os.path.exists(pairs_filename):
        print(f"Loading from saved pairs for {dataset_type}.")
        return True
    else:
        print(f"No saved pairs found for {dataset_type}. Generating new pairs.")
        return False

def main():
    seed_everything()
    data_path = 'scraping_output/wathec.json'
    raw_data = load_json_data(data_path)
    processed_data = prepare_dataframe(raw_data)
    grouped_images = group_images_by_brand(processed_data)
    val_brands = ['Tissot', 'Omega', 'Blancpain', 'Hamilton']

    # Assume transform is defined
    transform = transforms.Compose([
        transforms.Resize((300)),
        transforms.CenterCrop(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    seed_everything()
    data_path = 'scraping_output/watches_collections.json'  # Path to your JSON file with collections data
    collection_data = load_json_data(data_path)

    val_dataset = CollectionDataset(collection_data=collection_data, transform=transforms.Compose([
        transforms.Resize((400)),
        transforms.CenterCrop(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    train_data, _ = stratified_split_data(grouped_images, val_brands=val_brands)
    
    saved_pairs_dir = "saved_pairs"
    train_pairs_exist = check_for_saved_pairs(saved_pairs_dir, "train")
    val_pairs_exist = check_for_saved_pairs(saved_pairs_dir, "val")

    wandb.init(project='contrastive_loss', entity='DISPRO2', config={"learning_rate": 0.001, "epochs": 1, "batch_size": 32, "max_pairs": 64})
    train_dataset = PairDataset(train_data, save_dir=saved_pairs_dir, dataset_type="train",max_pairs= 10000)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)

    embedding_size = 128
    model = WatchEmbeddingModel(embedding_size=embedding_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    criterion = ContrastiveLoss(margin=2.0, loss_scale=3)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    trained_model = train(model, train_loader, train_dataset, optimizer, criterion, device, wandb.config.epochs, val_loader)

    
    # Example of visualization, you can also include training and validation here
    visualize_collection_embeddings(model, val_dataset, device)


    wandb.finish()

if __name__ == "__main__":
    main()

