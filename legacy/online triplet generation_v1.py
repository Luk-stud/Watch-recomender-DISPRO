import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
import numpy as np
import random
import json
from torch.optim.lr_scheduler import StepLR
from torchvision.utils import make_grid

# Additional modules for data splitting and visualization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load JSON data
def load_json_data(file_path):
    print(f"Loading JSON data from {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)
    print("Data loaded successfully")
    return data

# Prepare data from JSON
def prepare_dataframe(data):
    print("Preparing dataframe")
    corrected_data = []
    for item in data:
        corrected_item = {key.replace(':', '').strip(): value for key, value in item.items()}
        corrected_data.append(corrected_item)
    print("Dataframe prepared")
    return corrected_data

# Group images by brand
def group_images_by_brand(data):
    print("Grouping images by brand")
    grouped = {}
    for item in data:
        brand = item['Brand']
        image_path = item['Image Path']
        if brand in grouped:
            grouped[brand].append(image_path)
        else:
            grouped[brand] = [image_path]
    print("Images grouped by brand")
    return grouped

# Split data into train, validation, and test
def split_data(grouped_images, train_size=0.7, val_size=0.15, test_size=0.15):
    train_data = {}
    val_data = {}
    test_data = {}
    for brand, images in grouped_images.items():
        num_images = len(images)
        if num_images < 2:
            print(f"Skipping brand {brand} with {num_images} image(s). Not enough data to split.")
            continue
        
        random.shuffle(images)
        train_end = int(num_images * train_size)
        val_end = train_end + int(num_images * val_size)
        
        if train_end == 0:
            print(f"Warning: No training data for brand {brand}. Skipping.")
            continue
        
        train_data[brand] = images[:train_end]
        if (val_end - train_end) > 0:
            val_data[brand] = images[train_end:val_end]
        else:
            print(f"Warning: No validation data for brand {brand}. Adjusting train split.")
            train_data[brand] = images  # Adjusting to include all in train if validation split is not possible
        
        if (num_images - val_end) > 0:
            test_data[brand] = images[val_end:]
        else:
            print(f"Warning: No test data for brand {brand}. Adjusting validation split.")
            val_data[brand] = images[train_end:]  # Adjusting to include the rest in validation if test split is not possible

    return train_data, val_data, test_data

def safe_split_data(grouped_images, train_size=0.7, val_size=0.15, test_size=0.15):
    train_data, val_data, test_data = {}, {}, {}
    for brand, images in grouped_images.items():
        num_images = len(images)
        if num_images < 5:  # Consider a threshold to determine "too few"
            print(f"Warning: Only {num_images} images for brand {brand}. Using leave-one-out strategy.")
            if num_images == 1:
                # If only one image, use it for training only to avoid validation/test leakage
                train_data[brand] = images
                continue
            train_data[brand] = images[:-2]
            val_data[brand] = [images[-2]]  # Second last as validation
            test_data[brand] = [images[-1]]  # Last as test
        else:
            # Normal split
            train_end = int(num_images * train_size)
            val_end = train_end + int(num_images * val_size)
            train_data[brand] = images[:train_end]
            val_data[brand] = images[train_end:val_end]
            test_data[brand] = images[val_end:]

    return train_data, val_data, test_data



class TripletDataset(Dataset):
    def __init__(self, grouped_images, transform=None):
        self.grouped_images = grouped_images
        # Assuming each entry in grouped_images is a list of paths, we need to create valid triplets
        self.image_list = self._create_triplets(grouped_images)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _create_triplets(self, grouped_images):
        triplets = []
        for brand, images in grouped_images.items():
            # Continue only if there are at least two images for positive and negative can be different.
            if len(images) < 2:
                continue

            # Fetch different brand images just once per brand
            different_brands = {b: imgs for b, imgs in grouped_images.items() if b != brand and len(imgs) > 0}
            if not different_brands:
                continue  # If no different brands are available, skip

            for anchor in images:
                positive = random.choice([img for img in images if img != anchor])
                negative_brand = random.choice(list(different_brands.keys()))
                negative = random.choice(different_brands[negative_brand])
                triplets.append((anchor, positive, negative))
        return triplets

    def __getitem__(self, idx):
        try:
            path_anchor, path_pos, path_neg = self.image_list[idx]
            anchor = self.load_image(path_anchor)
            positive = self.load_image(path_pos)
            negative = self.load_image(path_neg)
            return anchor, positive, negative
        except Exception as e:
            # Skip or use a default value instead of None
            print(f"Error at index {idx}: {e}, paths: {self.image_list[idx]}")
            return self.__getitem__((idx + 1) % len(self))  # Recursive call to get the next item


    def load_image(self, path):
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_list)
    

    
class WatchEmbeddingModel(nn.Module):
    def __init__(self, embedding_size=400, train_deep_layers=True):
        super(WatchEmbeddingModel, self).__init__()
        base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Freeze all the features layers
        for param in base_model.features.parameters():
            param.requires_grad = False

        # If training deep layers, set the last few blocks of the convolutional layers to train
        if train_deep_layers:
            for param in base_model.features[-3:].parameters():  # Example: Unfreeze last 5 layers
                param.requires_grad = True
        
        self.features = base_model.features
        
        # Modifying the classifier
        self.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Increased dropout rate
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, embedding_size),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for anchors, positives, negatives in train_loader:
        anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
        optimizer.zero_grad()

        anchor_out = model(anchors)
        positive_out = model(positives)
        negative_out = model(negatives)

        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)



import matplotlib.pyplot as plt

def train_and_validate(model, train_loader, val_loader, device, optimizer, criterion, epochs, scheduler, writer, early_stop_patience=5):
    model.to(device)
    training_losses, validation_losses = [], []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(epochs):
        model.train()
        train_loss = []
        for batch_idx, (anchors, positives, negatives, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            anchor_out, positive_out, negative_out = model(anchors), model(positives), model(negatives)

            distance_positive = (anchor_out - positive_out).pow(2).sum(1)
            distance_negative = (anchor_out - negative_out).pow(2).sum(1)
            mask = (distance_negative > distance_positive).detach()

            if mask.any():
                loss = criterion(anchor_out[mask], positive_out[mask], negative_out[mask])
                loss.backward()
                optimizer.step()
                train_loss.append(loss.item())

            # Log gradients and weights histograms every 50 batches
            if (batch_idx + 1) % 50 == 0:
                for name, weight in model.named_parameters():
                    writer.add_histogram(f'{name}', weight, epoch)
                    writer.add_histogram(f'{name}.grad', weight.grad, epoch)

        # Calculate and log the average loss every epoch
        avg_train_loss = np.mean(train_loss)
        training_losses.append(avg_train_loss)
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation phase with embeddings visualization
        model.eval()
        val_loss, embeddings, label_list = 0, [], []
        with torch.no_grad():
            for anchors, positives, negatives, labels in val_loader:
                anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
                anchor_out, positive_out, negative_out = model(anchors), model(positives), model(negatives)
                loss = criterion(anchor_out, positive_out, negative_out)
                val_loss += loss.item()

                embeddings.append(anchor_out)
                label_list.extend(labels)

        avg_val_loss = val_loss / len(val_loader)
        validation_losses.append(avg_val_loss)
        scheduler.step()
        writer.add_scalar('Loss/validation', avg_val_loss, epoch)

        # Log embeddings
        embeddings = torch.cat(embeddings, 0)
        writer.add_embedding(embeddings, metadata=label_list, global_step=epoch, tag="Val Embeddings")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= early_stop_patience:
            print(f'Early stopping after {early_stop_patience} epochs without improvement.')
            break

    writer.close()
    return training_losses, validation_losses


def validate(model, loader, device, criterion):
    model.eval()  # Set model to evaluation mode
    total_val_loss = 0
    with torch.no_grad():
        for anchors, positives, negatives in loader:
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)

            # Forward pass
            anchor_out = model(anchors)
            positive_out = model(positives)
            negative_out = model(negatives)

            # Calculate validation loss
            loss = criterion(anchor_out, positive_out, negative_out)
            total_val_loss += loss.item()

    return total_val_loss / len(loader)  # Return average validation loss



# Script initialization remains the same
if __name__ == "__main__":


    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    file_path = 'scraping_output/updated_completed_log.json'
    raw_data = load_json_data(file_path)
    processed_data = prepare_dataframe(raw_data)
    grouped_images = group_images_by_brand(processed_data)

    # Split the data
    train_data, val_data, test_data = safe_split_data(grouped_images, train_size=0.7, val_size=0.15)

    # Transform and dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    # Add data augmentation to the training transform
    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation transform does not need augmentation
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Instantiate datasets with the appropriate transforms
    train_dataset = TripletDataset(train_data, transform=train_transform)
    val_dataset = TripletDataset(val_data, transform=val_transform)

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=32, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=24, shuffle=False, num_workers=8, pin_memory=True)

    # Model initialization
    model = WatchEmbeddingModel(embedding_size=600).to(device)
    print("Model initialized and moved to device.")

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-4)
    criterion = nn.TripletMarginLoss(margin=3.0)
    scheduler = StepLR(optimizer, step_size=3, gamma=0.07)


    # Train and validate
    epochs = 20
    train_and_validate(model, train_loader, val_loader, device, optimizer, criterion, epochs, scheduler=scheduler)
