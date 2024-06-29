import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pickle
import random
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
import wandb
from util_div import seed_everything

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
def filter_valid_families(df, min_families=3, min_images_per_family=5):
    valid_rows = []
    grouped = df.groupby('Brand')
    for brand, group in grouped:
        family_counts = group['Family'].value_counts()
        valid_families = family_counts[family_counts >= min_images_per_family].index
        if len(valid_families) >= min_families:
            valid_rows.append(group[group['Family'].isin(valid_families)])
    return pd.concat(valid_rows)

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

# Dataset class
class WatchDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['Image Path']
        image = Image.open(img_path).convert('RGB')
        label = int(self.df.iloc[idx]['Family'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class WatchClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(WatchClassificationModel, self).__init__()
        self.base_model = efficientnet_b0(pretrained=True)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

def train(model, train_loader, optimizer, criterion, device, epochs, val_loader, log_interval=1, patience=5):
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_labels = []
        train_preds = []

        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_labels.extend(labels.cpu().numpy())
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())

            if i % log_interval == 0:
                print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item()}")

        train_accuracy = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='weighted')
        val_loss, val_accuracy, val_f1 = validate(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {running_loss/len(train_loader)}, Train Accuracy: {train_accuracy}, Train F1: {train_f1}')
        print(f'Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}, Validation F1: {val_f1}')

        wandb.log({
            "Epoch": epoch+1,
            "Train Loss": running_loss/len(train_loader),
            "Train Accuracy": train_accuracy,
            "Train F1": train_f1,
            "Validation Loss": val_loss,
            "Validation Accuracy": val_accuracy,
            "Validation F1": val_f1
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_family_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping")
            break

    # Ensure that the model is returned
    return model

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    val_labels = []
    val_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    
    val_accuracy = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='weighted')
    
    return val_loss / len(val_loader), val_accuracy, val_f1

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

    # Filter out families with fewer entries
    min_entries = 5  # Define the minimum number of entries required per class
    family_counts = df['Family'].value_counts()
    valid_families = family_counts[family_counts >= min_entries].index
    df = df[df['Family'].isin(valid_families)]

    df = filter_valid_families(df)

    train_df, val_df, test_df = split_by_family(df)

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.CenterCrop(size=(256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomCrop(size=(224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Creating train dataset...")
    train_dataset = WatchDataset(train_df, transform=transform)
    print("Creating val dataset...")
    val_dataset = WatchDataset(val_df, transform=transform)
    print("Creating test dataset...")
    test_dataset = WatchDataset(test_df, transform=transform)

    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=8, pin_memory=True)

    print("Initializing model...")
    model = WatchClassificationModel(num_classes=len(family_encoder.classes_))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=wandb.config.learning_rate, momentum=0.5, weight_decay=1e-4)

    # Initial validation run
    initial_val_loss, initial_val_accuracy, initial_val_f1 = validate(model, val_loader, criterion, device)
    print(f"Initial Validation Loss: {initial_val_loss}, Validation Accuracy: {initial_val_accuracy}, Validation F1: {initial_val_f1}")
    wandb.log({"Validation Loss": initial_val_loss, "Validation Accuracy": initial_val_accuracy, "Validation F1": initial_val_f1})

    trained_model = train(model, train_loader, optimizer, criterion, device, wandb.config.epochs, val_loader)

    print("Evaluating on test data...")
    test_loss, test_accuracy, test_f1 = validate(trained_model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}, Test F1: {test_f1}")
    wandb.log({"Test Loss": test_loss, "Test Accuracy": test_accuracy, "Test F1": test_f1})

    # Save the label encoder
    label_encoder_path = 'label_encoder.pkl'
    with open(label_encoder_path, 'wb') as f:
        pickle.dump(family_encoder, f)
    
    # Save the label encoder to wandb
    artifact = wandb.Artifact('label_encoder', type='model')
    artifact.add_file(label_encoder_path)
    wandb.log_artifact(artifact)

if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project='cross_entropy_classification', entity='DISPRO2', config={
        "learning_rate": 0.003,  # Initial learning rate for SGD
        "epochs": 20,
        "batch_size": 32,  # Adjust batch size
        "patience": 5
    })
    main()