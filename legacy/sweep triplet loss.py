import torch
import torch.nn as nn
import wandb
import pandas as pd
from tqdm import tqdm
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import numpy as np
import json
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import GroupKFold
import matplotlib.pyplot as plt
import random
from torch.utils.data import Dataset, DataLoader
import time
from util_div import *
from util_model import *
from util_visualise_embeddings import *

# Define transforms
train_transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.RandomCrop(size=(224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def safe_split_data(grouped_images, train_size=0.8, val_size=0.2, test_size=0):
    train_data, val_data, test_data = {}, {}, {}
    for brand, images in grouped_images.items():
        num_images = len(images)
        if num_images < 5:
            print(f"Warning: Only {num_images} images for brand {brand}. Using leave-one-out strategy.")
            if num_images == 1:
                train_data[brand] = images
                continue
            train_data[brand] = images[:-2]
            val_data[brand] = [images[-2]]
            test_data[brand] = [images[-1]]
        else:
            train_end = int(num_images * train_size)
            val_end = train_end + int(num_images * val_size)
            train_data[brand] = images[:train_end]
            val_data[brand] = images[train_end:val_end]
            test_data[brand] = images[val_end:]

    return train_data, val_data, test_data

def visualize_embeddings(model, val_loader, device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for val_batch_idx, (val_images, _, _, val_labels, _) in enumerate(tqdm(val_loader, desc="Generating Embeddings")):
            val_images = val_images.to(device)

            val_embeddings = model(val_images)

            embeddings.append(val_embeddings.cpu().numpy())
            labels.extend(val_labels)

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)

    df = pd.DataFrame()
    df["embeddings"] = embeddings.tolist()
    df["labels"] = labels

    wandb.log({"embeddings": df})

class TripletDataset(Dataset):
    def __init__(self, grouped_images, transform=None):
        self.transform = transform
        self.image_list = []
        self.labels = []
        self.weights = []
        self._create_triplets(grouped_images)

    def _create_triplets(self, grouped_images):
        for brand, images in grouped_images.items():
            if len(images) < 2:
                continue
            weight = 1.0 / len(images)
            for anchor in images:
                positive = random.choice([img for img in images if img != anchor])
                negative_brand = random.choice([b for b in grouped_images.keys() if b != brand])
                if not grouped_images[negative_brand]:
                    continue
                negative = random.choice(grouped_images[negative_brand])
                self.image_list.append((anchor, positive, negative))
                self.labels.append(brand)
                self.weights.append(weight)

    def __getitem__(self, idx):
        path_anchor, path_positive, path_negative = self.image_list[idx]
        anchor = self.load_image(path_anchor)
        positive = self.load_image(path_positive)
        negative = self.load_image(path_negative)
        label = self.labels[idx]
        weight = self.weights[idx]
        return anchor, positive, negative, label, weight

    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        return self.transform(image) if self.transform else image

    def __len__(self):
        return len(self.image_list)


def train_and_validate(model, train_loader, val_loader, device, optimizer, criterion, epochs, scheduler, fold_idx=None):
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_valid_triplets = 0

        for batch_idx, (anchors, positives, negatives, labels, weights) in enumerate(train_loader):
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad()

            anchor_out = model(anchors)
            positive_out = model(positives)
            negative_out = model(negatives)
            
            distance_positive = (anchor_out - positive_out).pow(2).sum(1)
            distance_negative = (anchor_out - negative_out).pow(2).sum(1)
            
            mask = (distance_negative > distance_positive).detach()

            if mask.any():
                num_valid_triplets = mask.sum().item()
                total_valid_triplets += num_valid_triplets

                loss = criterion(anchor_out[mask], positive_out[mask], negative_out[mask])
                weighted_loss = (loss * weights[mask]).mean()

                weighted_loss.backward()
                optimizer.step()

                total_train_loss += weighted_loss.item() * len(anchors)

            if (batch_idx + 1) % 10 == 0:
                wandb.log({"batch_train_loss": weighted_loss.item(), "epoch": epoch})

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        wandb.log({"epoch_train_loss": avg_train_loss, "epoch": epoch, "valid_triplets": total_valid_triplets})

        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for val_batch_idx, (val_anchors, val_positives, val_negatives, _, val_weights) in enumerate(val_loader):
                val_anchors, val_positives, val_negatives = val_anchors.to(device), val_positives.to(device), val_negatives.to(device)
                val_weights = val_weights.to(device)

                val_anchor_out = model(val_anchors)
                val_positive_out = model(val_positives)
                val_negative_out = model(val_negatives)

                val_loss = criterion(val_anchor_out, val_positive_out, val_negative_out)
                weighted_val_loss = (val_loss * val_weights).mean()

                total_val_loss += weighted_val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            wandb.log({"epoch_val_loss": avg_val_loss, "epoch": epoch})

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()

        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if fold_idx is not None:
                model_save_path = f'model_fold_{fold_idx}.pth'
                torch.save(model.state_dict(), model_save_path)

    if best_model_state:
        model.load_state_dict(best_model_state)
        best_model_path = 'best_model.pth'
        torch.save(model.state_dict(), best_model_path)

    visualize_embeddings(model, val_loader, device)

    return {"train_loss": avg_train_loss, "val_loss": avg_val_loss}

def run_cross_validation(data_path, n_splits=5, epochs=10, random_seed=69, embedding_size=150, batch_size=32, lr=0.000002, weight_decay=0.001, margin=0.6, triplet_mining=True):
    wandb.init(project="tripplet_loss", entity="DISPRO2")  # Initialize wandb

    raw_data = load_json_data(data_path)
    processed_data = prepare_dataframe(raw_data)
    grouped_images = group_images_by_brand(processed_data)
    all_training_losses = []
    all_validation_losses = []

    # Configure wandb to capture hyperparameters and metadata
    config = wandb.config
    config.embedding_size = embedding_size
    config.batch_size = batch_size
    config.lr = lr
    config.weight_decay = weight_decay
    config.margin = margin
    config.epochs = epochs
    config.n_splits = n_splits
    config.random_seed = random_seed

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    for fold_idx in range(n_splits):
        train_data, val_data = {}, {}
        for brand, paths in grouped_images.items():
            brand_train_data, brand_val_data, _ = safe_split_data({brand: paths}, train_size=0.7, val_size=0.3, test_size=0.0)
            train_data.update(brand_train_data)
            val_data.update(brand_val_data)

        train_dataset = TripletDataset(train_data, transform=train_transform)
        val_dataset = TripletDataset(val_data, transform=val_transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size // 2, shuffle=False, num_workers=8, pin_memory=True)

        model = WatchEmbeddingModel(embedding_size=embedding_size)
        model = model.to("cuda")  # Move the model to GPU after initialization
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.TripletMarginLoss(margin=margin)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.3)

        training_losses, validation_losses = train_and_validate(model, train_loader, val_loader, torch.device("cuda"), optimizer, criterion, epochs, scheduler, fold_idx=fold_idx)
        all_training_losses.append(training_losses)
        all_validation_losses.append(validation_losses)

    wandb.finish()  # Finish the wandb run after the loops

def sweep_config():
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'val_loss',
            'goal': 'minimize'
        },
        'parameters': {
            'embedding_size': {'values': [50, 100, 150]},
            'batch_size': {'values': [16, 32, 64]},
            'lr': {'values': [0.0000001, 0.0000005, 0.000001]},
            'weight_decay': {'values': [0.0001, 0.0005, 0.001]},
            'margin': {'values': [0.5, 1.0, 1.5]},
            'epochs': {'value': 5},
            'n_splits': {'value': 5},
            'random_seed': {'value': 42}
        }
    }
    return sweep_config

def run_cross_validation_with_sweep():
    sweep_id = wandb.sweep(sweep_config(), project="triplett_loss_sweep", entity="DISPRO2")

    def train():
        with wandb.init() as run:
            config = wandb.config
            run_cross_validation(
                data_path='scraping_output/wathec.json',
                n_splits=config.n_splits,
                epochs=config.epochs,
                embedding_size=config.embedding_size,
                batch_size=config.batch_size,
                lr=config.lr,
                weight_decay=config.weight_decay,
                margin=config.margin,
                random_seed=config.random_seed
            )

    wandb.agent(sweep_id, train)

if __name__ == "__main__":
    run_cross_validation_with_sweep()
