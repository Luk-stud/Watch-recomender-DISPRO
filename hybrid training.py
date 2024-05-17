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
from torch.optim.lr_scheduler import StepLR


def stratified_split_data(grouped_images, val_brands=None, train_size=0.8, test_size=0.1, min_samples=2, random_state=None):
    print("Spliting data...")
    # Filter out classes with fewer than min_samples
    filtered_grouped_images = {brand: images for brand, images in grouped_images.items() if len(images) >= min_samples}
    brands = list(filtered_grouped_images.keys())
    
    # Populate train and validation data dictionaries
    train_data = {brand: images for brand, images in filtered_grouped_images.items() if brand not in val_brands}
    val_data = {brand: images for brand, images in filtered_grouped_images.items() if brand in val_brands}
    return train_data, val_data

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
        temp_pairs = []

        for brand, images in self.grouped_images.items():
            num_similar = int(len(images) * (len(images) - 1) / 2 * self.sim_ratio)
            num_dissimilar = num_similar * (1 / self.sim_ratio - 1)

            for _ in range(num_similar):
                i, j = np.random.choice(len(images), 2, replace=False)
                temp_pairs.append(((images[i], images[j]), 1, 1.0))

            other_brands = [b for b in brands if b != brand]
            for _ in range(int(num_dissimilar)):
                other_brand = np.random.choice(other_brands)
                other_images = self.grouped_images[other_brand]
                img1 = np.random.choice(images)
                img2 = np.random.choice(other_images)
                temp_pairs.append(((img1, img2), 0, 0.5))

        # Shuffle all pairs to remove order bias
        random.shuffle(temp_pairs)

        # Unpack shuffled pairs back into class variables
        for pair, label, weight in temp_pairs:
            self.image_pairs.append(pair)
            self.labels.append(label)
            self.weights.append(weight)

    def save_pairs(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.pairs_filename, 'wb') as f:
            # Save the shuffled pairs
            pickle.dump((self.image_pairs, self.labels, self.weights), f)

    def load_pairs(self):
        with open(self.pairs_filename, 'rb') as f:
            self.image_pairs, self.labels, self.weights = pickle.load(f)

    def apply_max_pairs(self):
        if self.max_pairs is not None and len(self.image_pairs) > self.max_pairs:
            indices = np.random.choice(len(self.image_pairs), self.max_pairs, replace=False)
            self.image_pairs = [self.image_pairs[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.weights = [self.weights[i] for i in indices]

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
        self.loss_scale = loss_scale  # A scaling factor for the loss

    def forward(self, output1, output2, labels, weights):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_similar = (1 - labels) * weights * (euclidean_distance ** 2)
        loss_dissimilar = labels * weights * (self.margin - euclidean_distance).clamp(min=0) ** 2
        contrastive_loss = torch.mean(loss_similar + loss_dissimilar)
        return contrastive_loss * self.loss_scale  # Scale the computed loss
    

class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, labels, weights):
        # Normalize the outputs to get unit vectors
        output1 = F.normalize(output1, p=2, dim=1)
        output2 = F.normalize(output2, p=2, dim=1)

        # Cosine similarity
        cosine_similarity = torch.sum(output1 * output2, dim=1)

        # Calculate losses for similar and dissimilar pairs
        loss_similar = (1 - labels) * (1 - cosine_similarity) * weights  # Weighted loss for similar pairs
        loss_dissimilar = labels * torch.clamp(cosine_similarity - self.margin, min=0) * weights  # Weighted loss for dissimilar pairs

        # Combine losses
        loss = torch.mean(loss_similar + loss_dissimilar)
        return loss


def train(model, train_loader, train_dataset, optimizer, scheduler, initial_criterion, final_criterion, switch_epoch, device, epochs, val_loader, log_interval=10):
    print("Starting training...")
    best_val_loss = float('inf')
    criterion = initial_criterion
        
    val_loss = validate(model, val_loader, CosineContrastiveLoss(margin=0.85), device)
    wandb.log({"Validation Loss": val_loss, "Epoch": 0})

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1} starts...")

        if epoch >= switch_epoch:
            criterion = final_criterion
            print(f"Switched to final loss function at Epoch {epoch+1}")

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

        val_loss = validate(model, val_loader, CosineContrastiveLoss(margin=0.85), device)
        wandb.log({"Epoch": epoch + 1, "Validation Loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), model_path)

            artifact = wandb.Artifact('model', type='model')
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss}")
        scheduler.step()  # Update the learning rate after each epoch

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


def main(weights_path=None):
    seed_everything()
    data_path = 'data/watches_database_main.json'
    raw_data = load_json_data(data_path)
    processed_data = prepare_dataframe(raw_data)
    grouped_images = group_images_by_brand(processed_data)
    val_brands = ['Tissot', 'Omega', 'Blancpain', 'Hamilton']

    transform = transforms.Compose([
        transforms.Resize((300)),
        transforms.CenterCrop(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize((256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])



    train_data, val_data = stratified_split_data(grouped_images, val_brands=val_brands)
    saved_pairs_dir = "saved_pairs"
    check_for_saved_pairs(saved_pairs_dir, "train")
    check_for_saved_pairs(saved_pairs_dir, "val")

    wandb.init(project='test', entity='DISPRO2', config={
        "learning_rate": 0.00005,
        "epochs": 5,
        "batch_size": 32,
        "switch_epoch": 0
    })

    train_dataset = PairDataset(train_data, save_dir=saved_pairs_dir, dataset_type="train", max_pairs=25000)
    val_dataset = PairDataset(val_data, save_dir=saved_pairs_dir, dataset_type="val", max_pairs=1000)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)

    embedding_size = 200
    model = WatchEmbeddingModel(embedding_size=embedding_size)
    device = "cuda"
    model.to(device)

    # Load weights if path provided
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Loaded model weights from {weights_path}")
    elif weights_path:
        print(f"Specified model weights not found at {weights_path}, starting training from scratch.")

    initial_criterion = ContrastiveLoss(margin=1.0, loss_scale=1)
    final_criterion = CosineContrastiveLoss(margin=0.85)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.1)

    trained_model = train(model, train_loader, train_dataset, optimizer, scheduler, initial_criterion, final_criterion, wandb.config.switch_epoch, device, wandb.config.epochs, val_loader)

    visualize_brand_embeddings(model, grouped_images, device, val_brands, transform_val)

    wandb.finish()

# Example of calling main with a specific model path
if __name__ == "__main__":
    model_weights_path = "model_euclidian_trained.pth"  # Change to your actual model path or set as None
    main(model_weights_path)


