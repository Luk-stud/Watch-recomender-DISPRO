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
from util_visualise_embeddings import *
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class WatchEmbeddingModel(nn.Module):
    def __init__(self, embedding_size, train_deep_layers=True):
        super(WatchEmbeddingModel, self).__init__()
        # Load pre-trained EfficientNet
        base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)  # Use pre-trained weights

        # Modify the first layer to accept single-channel input
        base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        # Optionally freeze the initial layers
        if not train_deep_layers:
            for param in base_model.parameters():
                param.requires_grad = False

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Replace the classification head with a custom embedding layer
        self.fc = nn.Linear(1280, embedding_size)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the output to (batch_size, 1280)
        x = self.fc(x)
        return x

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
def filter_valid_families(df, min_families=8, min_images_per_family=5, max_families=100):
    valid_rows = []
    grouped = df.groupby('Brand')
    
    for brand, group in grouped:
        family_counts = group['Family'].value_counts()
        valid_families = family_counts[family_counts >= min_images_per_family].index
        if len(valid_families) >= min_families:
            valid_rows.append(group[group['Family'].isin(valid_families)])
    
    df_filtered = pd.concat(valid_rows)
    
    # Limit to max_families (100) from different brands
    limited_families = df_filtered.groupby('Brand').head(1).groupby('Family').head(1)
    
    if len(limited_families) > max_families:
        limited_families = limited_families.sample(max_families, random_state=13)
        
    return df[df['Family'].isin(limited_families['Family'])]

def split_by_family(df):
    train_rows = []
    val_rows = []
    test_rows = []
    grouped = df.groupby('Family')
    
    for family, group in grouped:
        brands = group['Brand'].unique()
        if len(brands) >= 5:  # Ensure there are at least 3 brands to split into train, val, test
            train_brands, test_brands = train_test_split(brands, test_size=0.2, random_state=13)
            train_brands, val_brands = train_test_split(train_brands, test_size=0.25, random_state=13)  # 0.25 * 0.8 = 0.2
        elif len(brands) == 3:  # Split into train and test only
            train_brands, test_brands = train_test_split(brands, test_size=0.5, random_state=13)
            val_brands = train_brands  # Use train brands for validation as well
        else:  # All data goes to train if there is only one brand
            train_brands = brands
            val_brands = brands
            test_brands = brands
        
        train_rows.append(group[group['Brand'].isin(train_brands)])
        val_rows.append(group[group['Brand'].isin(val_brands)])
        test_rows.append(group[group['Brand'].isin(test_brands)])
    
    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows)
    test_df = pd.concat(test_rows)
    
    return train_df, val_df, test_df

# Convert DataFrame to dictionary grouped by family
def df_to_grouped_dict(df):
    grouped = df.groupby('Family')
    return {family: group['Image Path'].tolist() for family, group in grouped}

# PairDataset Class
from concurrent.futures import ThreadPoolExecutor, as_completed

class PairDataset(Dataset):
    def __init__(self, grouped_images, transform=None, sim_ratio=0.8, dataset_type="train", max_pairs=3000, max_val_pairs=500, max_test_pairs=500, image_cap=100):
        print(f"Initializing PairDataset for {dataset_type} data...")
        self.grouped_images = self.oversample_families(grouped_images, image_cap)
        self.transform = transform or transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
            transforms.Resize((512, 512)),
            transforms.CenterCrop(size=(256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomCrop(size=(224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std for single channel
        ])
        self.sim_ratio = sim_ratio
        self.dataset_type = dataset_type
        self.image_cap = image_cap

        if dataset_type == "train":
            self.max_pairs = max_pairs
        elif dataset_type == "val":
            self.max_pairs = max_val_pairs
        else:
            self.max_pairs = max_test_pairs

        print("Grouped images after oversampling", len(self.grouped_images))
        self.max_pairs_per_family = max(1, self.max_pairs // len(self.grouped_images))  # Adding limit per family
        self.image_pairs, self.labels, self.weights = self.generate_pairs()
        print(f"PairDataset for {dataset_type} data initialized with {len(self.image_pairs)} pairs.")

    def oversample_families(self, grouped_images, image_cap):
        oversampled_images = {}
        max_samples_per_family = min(image_cap, max(len(images) for images in grouped_images.values()))
        print(f"Max samples per family: {max_samples_per_family}")

        for family, images in grouped_images.items():
            if len(images) < max_samples_per_family:
                extra_samples = random.choices(images, k=max_samples_per_family - len(images))
                images.extend(extra_samples)
            oversampled_images[family] = images[:max_samples_per_family]

        print(f"Oversampling completed for {len(oversampled_images)} families.")
        return oversampled_images

    def generate_pairs(self):
        image_pairs = []
        labels = []
        weights = []
        families = list(self.grouped_images.keys())
        temp_pairs = []

        def generate_family_pairs(family, images):
            valid_images = [img for img in images if os.path.exists(img)]
            if len(valid_images) < 2:
                return []

            num_similar = min(int(len(valid_images) * (len(valid_images) - 1) / 2 * self.sim_ratio), self.max_pairs_per_family)
            num_dissimilar = min(int(num_similar * (1 / self.sim_ratio - 1)), self.max_pairs_per_family)
            print(f"Generating {num_similar} similar and {num_dissimilar} dissimilar pairs for family {family}...")

            family_pairs = []

            for _ in range(num_similar):
                i, j = np.random.choice(len(valid_images), 2, replace=False)
                family_pairs.append(((valid_images[i], valid_images[j]), 1, 1.0))
                if _ % 100 == 0:
                    print(f"Generated {_} similar pairs for family {family}")

            other_families = [f for f in families if f != family]
            for _ in range(num_dissimilar):
                other_family = np.random.choice(other_families)
                other_images = [img for img in self.grouped_images[other_family] if os.path.exists(img)]
                if other_images:
                    img1 = np.random.choice(valid_images)
                    img2 = np.random.choice(other_images)
                    family_pairs.append(((img1, img2), 0, 1.0))
                if _ % 100 == 0:
                    print(f"Generated {_} dissimilar pairs for family {family}")

            return family_pairs

        # Parallelize pair generation
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = [executor.submit(generate_family_pairs, family, images) for family, images in self.grouped_images.items()]
            for future in as_completed(futures):
                temp_pairs.extend(future.result())

        print("Shuffling pairs...")
        random.shuffle(temp_pairs)
        print("Shuffling completed.")

        limited_pairs = temp_pairs[:self.max_pairs]
        print(f"Limited pairs to {self.max_pairs}")

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

class NTXentLoss(nn.Module):
    def __init__(self, temperature):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim = torch.mm(z, z.t().contiguous())
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0)
        nominator = torch.exp(positives / self.temperature)
        negatives_mask = ~torch.eye(2 * batch_size, 2 * batch_size, dtype=torch.bool).to(z.device)
        denominator = torch.sum(negatives_mask * torch.exp(sim / self.temperature), dim=1)
        loss = -torch.log(nominator / denominator).mean()
        return loss

def train(model, train_loader, optimizer, criterion, device, epochs, val_loader, scheduler, log_interval=1, patience=20):
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    val_interval = 3

    scaler = torch.cuda.amp.GradScaler()  # Initialize scaler for mixed precision training

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

                # Normalize embeddings
                output1 = F.normalize(output1, p=2, dim=1)
                output2 = F.normalize(output2, p=2, dim=1)

                if torch.isnan(output1).any() or torch.isnan(output2).any():
                    print(f"NaN detected in model outputs at batch {i+1}. Skipping this batch.")
                    continue

                loss = criterion(output1, output2)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Invalid loss detected at batch {i+1}. Skipping this batch.")
                continue

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Step the scheduler after each batch
            
            running_loss += loss.item()

            if i % log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"At batch {i+1}, loss is {loss.item()}, learning rate is {current_lr}")
                wandb.log({"Epoch": epoch + 1, "Batch Training Loss": loss.item(), "Step": i, "Learning Rate": current_lr})

        if epoch % val_interval == 0:
            val_loss = validate(model, val_loader, criterion, device)
            wandb.log({"Epoch": epoch + 1, "Validation Loss": val_loss})

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

                if torch.isnan(output1).any() or torch.isnan(output2).any():
                    print("NaN detected in model outputs during validation. Skipping this batch.")
                    continue

                loss = criterion(output1, output2)

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

    family_encoder = LabelEncoder()
    df['Family'] = family_encoder.fit_transform(df['Family'])
    df['Family'] = df['Family'].apply(str)  # Convert the 'Family' column to string
    df['Brand'] = df['Brand'].apply(lambda x: x.strip() if pd.notnull(x) else x)

    # Filter out families with fewer entries and limit to 100 families from different brands
    min_entries = 5  # Define the minimum number of entries required per class
    family_counts = df['Family'].value_counts()
    valid_families = family_counts[family_counts >= min_entries].index
    df = df[df['Family'].isin(valid_families)]

    df = filter_valid_families(df, min_families=8, min_images_per_family=5, max_families=100)

    train_df, val_df, test_df = split_by_family(df)

    train_data = df_to_grouped_dict(train_df)
    val_data = df_to_grouped_dict(val_df)
    test_data = df_to_grouped_dict(test_df)

    grouped_images = df_to_grouped_dict(df)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Convert images to grayscale
        transforms.Resize((224, 224)),
        transforms.CenterCrop(size=(160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomCrop(size=(130, 130)),
        transforms.Resize((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std for single channel
    ])

    wandb.init(project='contrastive_loss', entity='DISPRO2', config={
        "learning_rate": 1,  # Initial learning rate for SGD
        "epochs": 200,
        "batch_size": 512,  # Increased batch size for grayscale images
        "temp": 1.0,
        "loss_type": "NTXentLoss",
        "sim_ratio": 0.5,
        "max_pairs": 25600,
        "max_val_pairs": 512,  # Limit for validation pairs
        "max_test_pairs": 512,  # Limit for test pairs
        "image_cap": 1000,
        "patience": 20,
        "embedding_size": 1280,
        "log_interval": 1,
    })

    config = wandb.config

    print("Creating train dataset...")
    train_dataset = PairDataset(train_data, transform=transform, dataset_type="train", sim_ratio=config.sim_ratio, max_pairs=config.max_pairs, max_val_pairs=config.max_val_pairs, max_test_pairs=config.max_test_pairs, image_cap=config.image_cap)
    print("Creating val dataset...")
    val_dataset = PairDataset(val_data, transform=transform, dataset_type="val", sim_ratio=config.sim_ratio, max_pairs=config.max_val_pairs, max_val_pairs=config.max_val_pairs, max_test_pairs=config.max_test_pairs, image_cap=config.image_cap)
    print("Creating test dataset...")
    test_dataset = PairDataset(test_data, transform=transform, dataset_type="test", sim_ratio=config.sim_ratio, max_pairs=config.max_test_pairs, max_val_pairs=config.max_val_pairs, max_test_pairs=config.max_test_pairs, image_cap=config.image_cap)

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True, num_workers=8)

    print("Initializing model...")
    model = WatchEmbeddingModel(embedding_size=config.embedding_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = NTXentLoss(temperature=config.temp)
    
    # Switching to LAMB optimizer
    from torch_optimizer import Lamb
    optimizer = Lamb(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    # Initialize OneCycleLR scheduler
    total_steps = config.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config.learning_rate, total_steps=total_steps
    )

    # Initial validation run
    initial_val_loss = validate(model, val_loader, criterion, device)
    print(f"Initial Validation Loss: {initial_val_loss}")
    wandb.log({"Validation Loss": initial_val_loss})

    trained_model = train(model, train_loader, optimizer, criterion, device, config.epochs, val_loader, scheduler)

    print("Evaluating on test data...")
    test_loss = validate(trained_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss}")
    wandb.log({"Test Loss": test_loss})

    # Visualization without cropping
    visualize_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Adjust mean and std for single channel
    ])

    #visualize_embeddings(trained_model, test_data, visualize_transform, device)

if __name__ == "__main__":
    main()
