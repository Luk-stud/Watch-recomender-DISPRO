import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import random
import torch.nn.functional as F
from util_div import *
from util_model import *
from util_visualise_embeddings import *

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Prepare data from JSON
def prepare_dataframe(data):
    return [{key.replace(':', '').strip(): value for key, value in item.items()} for item in data]

# Check if image paths exist
def check_image_paths(df, image_column):
    return df[df[image_column].apply(os.path.exists)]

# Ensure there are enough families and pictures for each brand
def filter_valid_families(df, min_families=5, min_images_per_family=5):
    valid_rows = []
    grouped = df.groupby('Brand')
    for brand, group in grouped:
        family_counts = group['Family'].value_counts()
        valid_families = family_counts[family_counts >= min_images_per_family].index
        if len(valid_families) >= min_families:
            valid_rows.append(group[group['Family'].isin(valid_families)])
    return pd.concat(valid_rows)

# Split the data by family ensuring different families in each split
def split_by_family(df):
    train_rows = []
    val_rows = []
    test_rows = []
    grouped = df.groupby('Brand')
    
    for brand, group in grouped:
        families = group['Family'].unique()
        if len(families) >= 3:  # Ensure there are at least 3 families to split into train, val, test
            train_families, test_families = train_test_split(families, test_size=0.2, random_state=13)
            train_families, val_families = train_test_split(train_families, test_size=0.25, random_state=13)  # 0.25 * 0.8 = 0.2
        elif len(families) == 2:  # Split into train and test only
            train_families, test_families = train_test_split(families, test_size=0.5, random_state=13)
            val_families = train_families  # Use train families for validation as well
        else:  # All data goes to train if there is only one family
            train_families = families
            val_families = families
            test_families = families
        
        train_rows.append(group[group['Family'].isin(train_families)])
        val_rows.append(group[group['Family'].isin(val_families)])
        test_rows.append(group[group['Family'].isin(test_families)])
    
    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows)
    test_df = pd.concat(test_rows)
    
    return train_df, val_df, test_df

# Convert DataFrame to dictionary grouped by brand
def df_to_grouped_dict(df):
    grouped = df.groupby('Brand')
    return {brand: group['Image Path'].tolist() for brand, group in grouped}

# PairDataset Class
class PairDataset(Dataset):
    def __init__(self, grouped_images, transform=None, sim_ratio=0.8, dataset_type="train", max_pairs=30000, image_cap=100):
        self.grouped_images = self.stratified_sample(grouped_images, image_cap)
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.RandomCrop(size=(112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.sim_ratio = sim_ratio
        self.max_pairs = max_pairs
        self.dataset_type = dataset_type
        self.image_pairs, self.labels, self.weights = self.generate_pairs()

    def stratified_sample(self, grouped_images, image_cap):
        stratified_images = {}
        min_images_per_brand = min(len(images) for images in grouped_images.values())
        max_samples_per_brand = min(min_images_per_brand, image_cap)

        for brand, images in grouped_images.items():
            sampled_images = random.sample(images, min(max_samples_per_brand, len(images)))
            stratified_images[brand] = sampled_images
        return stratified_images

    def generate_pairs(self):
        image_pairs = []
        labels = []
        weights = []
        brands = list(self.grouped_images.keys())
        temp_pairs = []

        for brand, images in self.grouped_images.items():
            valid_images = [img for img in images if os.path.exists(img)]
            num_similar = int(len(valid_images) * (len(valid_images) - 1) / 2 * self.sim_ratio)
            num_dissimilar = num_similar * (1 / self.sim_ratio - 1)

            for _ in range(num_similar):
                i, j = np.random.choice(len(valid_images), 2, replace=False)
                temp_pairs.append(((valid_images[i], valid_images[j]), 1, 1.0))

            other_brands = [b for b in brands if b != brand]
            for _ in range(int(num_dissimilar)):
                other_brand = np.random.choice(other_brands)
                other_images = [img for img in self.grouped_images[other_brand] if os.path.exists(img)]
                img1 = np.random.choice(valid_images)
                img2 = np.random.choice(other_images)
                temp_pairs.append(((img1, img2), 0, 1.0))

        # Shuffle all pairs to remove order bias
        random.shuffle(temp_pairs)

        # Limit the number of pairs
        limited_pairs = temp_pairs[:self.max_pairs]

        for pair, label, weight in limited_pairs:
            image_pairs.append(pair)
            labels.append(label)
            weights.append(weight)

        return image_pairs, labels, weights

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


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, loss_scale=3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_scale = loss_scale

    def forward(self, output1, output2, labels, weights):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_similar = (1 - labels) * weights * (euclidean_distance ** 2)
        loss_dissimilar = labels * weights * (self.margin - euclidean_distance).clamp(min=0) ** 2
        contrastive_loss = torch.mean(loss_similar + loss_dissimilar)
        return contrastive_loss * self.loss_scale
    


def train(model, train_loader, optimizer, criterion, device, epochs, val_loader, log_interval=1, patience=5):
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0

    scaler = torch.cuda.amp.GradScaler()  # Initialize scaler for mixed precision training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1} starts...")

        for i, (img1, img2, labels, weights) in enumerate(train_loader):
            img1, img2 = img1.to(device), img2.to(device)
            labels, weights = labels.to(device), weights.to(device)
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():  # Mixed precision context
                output1 = model(img1)
                output2 = model(img2)
                print(output1,output2)

                # Normalize embeddings
                output1 = F.normalize(output1, p=2, dim=1)
                output2 = F.normalize(output2, p=2, dim=1)

                loss = criterion(output1, output2, labels, weights)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected at batch {i+1}. Skipping this batch.")
                continue

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

            if i % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"At batch {i+1}, loss is {loss.item()}, learning rate is {current_lr}")
                wandb.log({"Epoch": epoch + 1, "Batch Training Loss": loss.item(), "Step": i, "Learning Rate": current_lr})

        val_loss = validate(model, val_loader, criterion, device)
        wandb.log({"Epoch": epoch + 1, "Validation Loss": val_loss})
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)

            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")

    return model

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for img1, img2, labels, weights in val_loader:
            img1, img2 = img1.to(device), img2.to(device)
            labels, weights = labels.to(device), weights.to(device)
            
            with torch.cuda.amp.autocast():  # Mixed precision context
                output1 = model(img1)
                output2 = model(img2)

                # Normalize embeddings
                output1 = F.normalize(output1, p=2, dim=1)
                output2 = F.normalize(output2, p=2, dim=1)

                loss = criterion(output1, output2, labels, weights)

            if torch.isnan(loss) or torch.isinf(loss):
                print("Invalid loss detected during validation. Skipping this batch.")
                continue

            total_loss += loss.item()
    return total_loss / len(val_loader)

def main():
    seed_everything()
    file_path = 'data/watches_database_main.json'  # Replace with your file path
    data = load_json_data(file_path)
    data = prepare_dataframe(data)
    df = pd.DataFrame(data)
    df = check_image_paths(df, 'Image Path')

    brand_encoder = LabelEncoder()
    df['Brand'] = brand_encoder.fit_transform(df['Brand'])
    df['Brand'] = df['Brand'].apply(str)  # Convert the 'Brand' column to string
    df['Family'] = df['Family'].apply(lambda x: x.strip() if pd.notnull(x) else x)

    # Filter out classes with fewer entries
    min_entries = 5  # Define the minimum number of entries required per class
    brand_counts = df['Brand'].value_counts()
    valid_brands = brand_counts[brand_counts >= min_entries].index
    df = df[df['Brand'].isin(valid_brands)]

    df = filter_valid_families(df)

    train_df, val_df, test_df = split_by_family(df)

    train_data = df_to_grouped_dict(train_df)
    val_data = df_to_grouped_dict(val_df)
    test_data = df_to_grouped_dict(test_df)

    grouped_images = df_to_grouped_dict(df)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(size=(224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomCrop(size=(112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    wandb.init(project='contrastive_loss', entity='DISPRO2', config={
        "learning_rate": 0.0001,  # Initial learning rate for SGD
        "epochs": 50,
        "batch_size": 512,  # Reduced batch size
        "margin": 1.0,
        "loss_type": "ContrastiveLoss",
        "sim_ratio": 0.6,
        "max_pairs": 20000,
        "image_cap": 100,
        "patience": 10,
        "embedding_size": 5,
        "log_interval": 1,
    })

    train_dataset = PairDataset(train_data, transform=transform, dataset_type="train", sim_ratio=wandb.config.sim_ratio, max_pairs=wandb.config.max_pairs, image_cap=wandb.config.image_cap)
    val_dataset = PairDataset(val_data, transform=transform, dataset_type="val", sim_ratio=wandb.config.sim_ratio, max_pairs=wandb.config.max_pairs, image_cap=wandb.config.image_cap)
    test_dataset = PairDataset(test_data, transform=transform, dataset_type="test", sim_ratio=wandb.config.sim_ratio, max_pairs=wandb.config.max_pairs, image_cap=wandb.config.image_cap)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)

    model = WatchEmbeddingModel(embedding_size=wandb.config.embedding_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = ContrastiveLoss(margin=wandb.config.margin, loss_scale=3)
    
    optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=0.9, weight_decay=1e-4)

    # Initial validation run
    initial_val_loss = validate(model, val_loader, criterion, device)
    print(f"Initial Validation Loss: {initial_val_loss}")
    wandb.log({"Validation Loss": initial_val_loss})

    trained_model = train(model, train_loader, optimizer, criterion, device, wandb.config.epochs, val_loader)

    print("Evaluating on test data...")
    test_loss = validate(trained_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss}")
    wandb.log({"Test Loss": test_loss})

    # Visualization without cropping
    visualize_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    visualize_brand_embeddings(model, grouped_images, device, list(test_data.keys()), visualize_transform)

    wandb.finish()

if __name__ == "__main__":
    main()
