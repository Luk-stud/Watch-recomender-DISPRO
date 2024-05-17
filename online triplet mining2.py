import torch
import torch.nn as nn
import wandb
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader
import random
from util_div import *
from util_model import *
from util_visualise_embeddings import *


train_transform = transforms.Compose([
    transforms.Resize((600)),  # Resize the image to a larger size
    transforms.RandomHorizontalFlip(p=0.5),  # Apply random horizontal flip with probability 0.5
    transforms.RandomRotation(degrees=30),
    transforms.CenterCrop(size=(400, 400)),  # Center crop the image to size (224, 224) without changing proportions
   # Apply random rotation with maximum rotation of 30 degrees
    transforms.RandomCrop(256),  # Center crop the image to size (224, 224) without changing proportions
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((400)),  # Resize the image to a larger size
    transforms.CenterCrop(size=(256, 256)),  # Center crop the image to size (224, 224) without changing proportions

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def safe_split_data(grouped_images, train_size=0.8, val_size=0.2, test_size=0):
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

from sklearn.model_selection import train_test_split

def stratified_split_data(grouped_images, val_brands=None, test_brands=None, train_size=0.8, test_size=0.1, random_state=None):
    train_data, val_data, test_data = {}, {}, {}
    
    # Create lists of brands and their corresponding image counts
    brands = list(grouped_images.keys())
    brand_image_counts = [len(images) for images in grouped_images.values()]
    
    # If validation brands are not provided, select them randomly
    if val_brands is None:
        val_brands = train_test_split(brands, train_size=1-train_size, test_size=test_size, stratify=brand_image_counts, random_state=random_state)[1]
        
    # If test brands are not provided, select them randomly
    if test_brands is None:
        train_brands, test_brands = train_test_split(brands, train_size=train_size, test_size=test_size, stratify=brand_image_counts, random_state=random_state)

    # Populate train, validation, and test data dictionaries
    for brand, images in grouped_images.items():
        if brand in val_brands:
            val_data[brand] = images
        elif brand in test_brands:
            test_data[brand] = images
        else:
            train_data[brand] = images
            
    return train_data, val_data, test_data






class TripletDataset(Dataset):
    def __init__(self, grouped_images, transform=None):
        self.transform = transform
        self.image_list = []
        self.labels = []
        self.weights = []  # List to store weights for each triplet
        self._create_triplets(grouped_images)

    def _create_triplets(self, grouped_images):
        for brand, images in grouped_images.items():
            if len(images) < 2:
                continue
            # Calculate weight based on the number of images for the brand
            weight = 1.0 / len(images)
            for anchor in images:
                positive = random.choice([img for img in images if img != anchor])
                negative_brand = random.choice([b for b in grouped_images.keys() if b != brand])
                # Check if there are images available for the selected negative brand
                if not grouped_images[negative_brand]:
                    continue
                negative = random.choice(grouped_images[negative_brand])
                self.image_list.append((anchor, positive, negative))
                self.labels.append(brand)  # Store the brand as label
                self.weights.append(weight)  # Assign weight to the triplet

    def __getitem__(self, idx):
        path_anchor, path_positive, path_negative = self.image_list[idx]
        anchor = self.load_image(path_anchor)
        positive = self.load_image(path_positive)
        negative = self.load_image(path_negative)
        label = self.labels[idx]  # Get the corresponding label
        weight = self.weights[idx]  # Get the corresponding weight
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

        # Training loop
        for batch_idx, (anchors, positives, negatives, labels, weights) in enumerate(train_loader):
            anchors, positives, negatives = anchors.to(device), positives.to(device), negatives.to(device)
            weights = weights.to(device)
            
            optimizer.zero_grad()

            anchor_out = model(anchors)
            positive_out = model(positives)
            negative_out = model(negatives)
            
            distance_positive = (anchor_out - positive_out).pow(2).sum(1)
            distance_negative = (anchor_out - negative_out).pow(2).sum(1)
            
            # Loss that considers only cases where negative is farther than positive from the anchor
            mask = (distance_negative > distance_positive).detach()

            if mask.any():
                num_valid_triplets = mask.sum().item()
                total_valid_triplets += num_valid_triplets

                loss = criterion(anchor_out[mask], positive_out[mask], negative_out[mask])
                pure_loss = loss.mean()  # Calculate the pure loss
                weighted_loss = (loss * weights[mask]).mean()

                weighted_loss.backward()
                optimizer.step()

                total_train_loss += weighted_loss.item() * len(anchors)

            if (batch_idx + 1) % 10 == 0:  # Log every 10 batches
                wandb.log({"batch_train_loss": pure_loss.item(), "epoch": epoch})
                print(f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {weighted_loss.item()}")

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        wandb.log({"epoch_train_loss": avg_train_loss, "epoch": epoch, "valid_triplets": total_valid_triplets})
        print(f"Epoch [{epoch+1}/{epochs}], Average Train Loss: {avg_train_loss}, Total Valid Triplets: {total_valid_triplets}")

        # Validation loop (outside the epoch loop)
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
                pure_val_loss = val_loss.mean()

                total_val_loss += pure_val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader.dataset)
            wandb.log({"epoch_val_loss": avg_val_loss, "epoch": epoch})
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = model.state_dict()

        # Scheduler step (if using learning rate decay)
        scheduler.step()

        # Check for early stopping or other conditions based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if fold_idx is not None:
                model_save_path = f'model_fold_{fold_idx}.pth'
                torch.save(model.state_dict(), model_save_path)
                #wandb.save(model_save_path)

    # Save the best model found during the training
    if best_model_state:
        model.load_state_dict(best_model_state)
        best_model_path = 'best_model.pth'
        torch.save(model.state_dict(), best_model_path)
        #wandb.save(best_model_path)

    
    # Generate and visualize embeddings using the best model
    visualize_embeddings(model, val_loader, device)


    return {"train_loss": avg_train_loss, "val_loss": avg_val_loss}



def run_cross_validation(data_path, val_brands=None, test_brands=None, n_splits=5, epochs=10, random_seed=69, embedding_size=150, batch_size=32, lr=0.000002, weight_decay=0.001, margin=0.6, triplet_mining=True):
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
        # Split data into train, val, and test sets based on provided brands
        train_data, val_data, test_data = stratified_split_data(grouped_images, val_brands=val_brands, test_brands=test_brands, random_state=random_seed)

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
if __name__ == "__main__":
    data_path = 'scraping_output/wathec.json'
    embedding_size = 100  # Reduced embedding size
    batch_size = 32  # Increased batch size
    lr = 0.0000005  # Increased learning rate
    weight_decay = 0.0005  # Decreased weight decay
    margin = 1.5  # Increased margin for triplet loss
    epochs = 5
    n_splits = 1
    val_brands = ['Tissot', 'Omega','Blancpain','Hamilton']  # Specify the brands to be in the validation dataset
    test_brands = []  # Specify the brands to be in the test dataset

    run_cross_validation(data_path, val_brands=val_brands, test_brands=test_brands, n_splits=n_splits, epochs=epochs, embedding_size=embedding_size, batch_size=batch_size, lr=lr, weight_decay=weight_decay, margin=margin)


