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

# TripletDataset Class with Online Triplet Mining
class TripletDataset(Dataset):
    def __init__(self, grouped_images, anchors, label_encoder, transform=None):
        self.grouped_images = grouped_images
        self.anchors = anchors
        self.label_encoder = label_encoder
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.image_paths, self.labels = self.get_image_paths_and_labels()

    def get_image_paths_and_labels(self):
        image_paths = []
        labels = []
        for label, images in self.grouped_images.items():
            image_paths.extend(images)
            encoded_label = self.label_encoder.transform([label])[0]
            labels.extend([encoded_label] * len(images))
        return image_paths, labels

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        anchor = torch.tensor(self.anchors[label], dtype=torch.float32)
        return img, label, anchor

    def __len__(self):
        return len(self.image_paths)


def triplet_mining(model, images, labels, anchors, device, margin=1.0):
    embeddings = model(images.to(device))
    embeddings = F.normalize(embeddings, p=2, dim=1)
    
    triplets = []
    anchor_tensors = []
    positives = []
    negatives = []
    
    labels = labels.to(device)  # Ensure labels are tensors
    
    for i in range(len(embeddings)):
        anchor = anchors[i].to(device)
        label = labels[i]
        positive_mask = (labels == label)
        negative_mask = (labels != label)
        
        if positive_mask.sum() < 2:
            continue  # Skip if not enough positive samples
        
        positive_distances = torch.cdist(anchor.unsqueeze(0), embeddings[positive_mask])
        negative_distances = torch.cdist(anchor.unsqueeze(0), embeddings[negative_mask])
        
        positive_distances = positive_distances[0]
        negative_distances = negative_distances[0]
        
        hardest_positive_distance, hardest_positive_idx = positive_distances.min(dim=0)
        hardest_negative_distance, hardest_negative_idx = negative_distances.min(dim=0)
        
        positive_idx = torch.nonzero(positive_mask)[hardest_positive_idx.item()].item()
        negative_idx = torch.nonzero(negative_mask)[hardest_negative_idx.item()].item()
        
        triplets.append((i, positive_idx, negative_idx))
        anchor_tensors.append(anchor)
        positives.append(embeddings[positive_idx])
        negatives.append(embeddings[negative_idx])
    
    if not triplets:
        return None, None, None  # No valid triplets found
    
    return torch.stack(anchor_tensors), torch.stack(positives), torch.stack(negatives)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_distance = torch.nn.functional.pairwise_distance(anchor, positive)
        neg_distance = torch.nn.functional.pairwise_distance(anchor, negative)
        losses = F.relu(pos_distance - neg_distance + self.margin)
        return losses.mean()

def compute_anchors(model, grouped_images, device, transform, label_encoder):
    anchors = {}
    model.eval()
    with torch.no_grad():
        for brand, images in grouped_images.items():
            embeddings = []
            for img_path in images:
                img = Image.open(img_path).convert('RGB')
                img = transform(img).unsqueeze(0).to(device)
                embedding = model(img)
                embeddings.append(embedding.cpu().numpy())
            mean_embedding = np.mean(embeddings, axis=0)
            encoded_label = label_encoder.transform([brand])[0]
            anchors[encoded_label] = mean_embedding
    return anchors


def train_triplet(model, train_loader, optimizer, criterion, device, epochs, val_loader, log_interval=1, patience=5, grouped_images=None, transform=None):
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0

    scaler = torch.cuda.amp.GradScaler()  # Initialize scaler for mixed precision training
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1} starts...")

        for i, (images, labels, anchors) in enumerate(train_loader):
            optimizer.zero_grad()
            anchor_tensors, positives, negatives = triplet_mining(model, images, labels, anchors, device)
            if anchor_tensors is None:
                continue  # Skip if no valid triplets

            loss = criterion(anchor_tensors, positives, negatives)

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

        val_loss = validate_triplet(model, val_loader, criterion, device)
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

        # Update anchors
        if grouped_images and transform and (epoch + 1) % 3 == 0:
            anchors = compute_anchors(model, grouped_images, device, transform, brand_encoder)
            train_loader.dataset.anchors = anchors

    return model

def validate_triplet(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels, anchors in val_loader:
            anchor_tensors, positives, negatives = triplet_mining(model, images, labels, anchors, device)
            if anchor_tensors is None:
                continue  # Skip if no valid triplets

            loss = criterion(anchor_tensors, positives, negatives)

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

    # Combine all labels for fitting the LabelEncoder
    all_labels = list(train_data.keys()) + list(val_data.keys()) + list(test_data.keys())
    brand_encoder.fit(all_labels)

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

    wandb.init(project='triplet_loss', entity='DISPRO2', config={
        "learning_rate": 0.001,  # Initial learning rate for SGD
        "epochs": 50,
        "batch_size": 256,  # Reduced batch size
        "margin": 1.0,
        "loss_type": "TripletLoss",
        "max_pairs": 20000,
        "image_cap": 1000,
        "patience": 20,
        "embedding_size": 256,
        "log_interval": 1,
        "mode": "offline"  # Add this line to enable offline mode
    })

    model = WatchEmbeddingModel(embedding_size=wandb.config.embedding_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initial computation of anchors
    anchors = compute_anchors(model, grouped_images, device, transform, brand_encoder)

    train_dataset = TripletDataset(train_data, anchors, brand_encoder, transform=transform)
    val_dataset = TripletDataset(val_data, anchors, brand_encoder, transform=transform)
    test_dataset = TripletDataset(test_data, anchors, brand_encoder, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)

    criterion = TripletLoss(margin=wandb.config.margin)
    optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=0.9, weight_decay=1e-4)

    # Initial validation run
    initial_val_loss = validate_triplet(model, val_loader, criterion, device)
    print(f"Initial Validation Loss: {initial_val_loss}")
    wandb.log({"Validation Loss": initial_val_loss})

    trained_model = train_triplet(model, train_loader, optimizer, criterion, device, wandb.config.epochs, val_loader, grouped_images=grouped_images, transform=transform)

    print("Evaluating on test data...")
    test_loss = validate_triplet(trained_model, test_loader, criterion, device)
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
