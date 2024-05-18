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
from sklearn.model_selection import train_test_split
from collections import defaultdict
import os
import random

from torchvision.models import vgg16, VGG16_Weights


# Set a seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def prepare_dataframe(data):
    return [{key.replace(':', '').strip(): value for key, value in item.items()} for item in data]

def group_images_by_brand(data):
    grouped = {}
    for item in data:
        grouped.setdefault(item['Brand'], []).append(item['Image Path'])
    return grouped

def stratified_split_data(grouped_images, val_brands=None, train_size=0.8, test_size=0.1, min_samples=2, random_state=None):
    # Filter out classes with fewer than min_samples
    filtered_grouped_images = {brand: images for brand, images in grouped_images.items() if len(images) >= min_samples}
    brands = list(filtered_grouped_images.keys())
    brand_image_counts = [len(images) for images in filtered_grouped_images.values()]
    
    # Populate train and validation data dictionaries
    train_data = {brand: images for brand, images in filtered_grouped_images.items() if brand not in val_brands}
    val_data = {brand: images for brand, images in filtered_grouped_images.items() if brand in val_brands}
    return train_data, val_data


class WatchEmbeddingModel(nn.Module):
    def __init__(self, embedding_size, train_deep_layers=True):
        super(WatchEmbeddingModel, self).__init__()
        base_model = vgg16(weights=VGG16_Weights.DEFAULT)
        if train_deep_layers:
            for param in base_model.features[-1:].parameters():
                param.requires_grad = True
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.embedder = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(4096, embedding_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.embedder(x)
        return x



import pickle
import os

class PairDataset(Dataset):
    def __init__(self, grouped_images, transform=None, sim_ratio=0.5, save_dir="saved_pairs"):
        self.grouped_images = grouped_images
        self.transform = transform or transforms.Compose([
            transforms.Resize((400)),
            transforms.CenterCrop(size=(256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.sim_ratio = sim_ratio
        self.save_dir = save_dir
        self.pairs_filename = os.path.join(save_dir, "pairs.pkl")
        
        if not os.path.exists(self.pairs_filename):
            self.generate_pairs()
            self.save_pairs()
        else:
            self.load_pairs()

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

    def save_pairs(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.pairs_filename, 'wb') as f:
            pickle.dump((self.image_pairs, self.labels, self.weights), f)

    def load_pairs(self):
        with open(self.pairs_filename, 'rb') as f:
            self.image_pairs, self.labels, self.weights = pickle.load(f)

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
    

def visualize_embeddings(model, val_loader, device, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    model.eval()
    embeddings = []
    labels_list = []
    images_list = []

    # Denormalization function
    def denormalize(tensor):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    with torch.no_grad():
        for img1, img2, labels, weights in val_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            embeddings_output1 = model(img1).cpu().numpy()
            embeddings_output2 = model(img2).cpu().numpy()

            # Process each embedding and label for img1 and img2
            for i in range(img1.size(0)):  # Process one image at a time
                emb1 = embeddings_output1[i]
                emb2 = embeddings_output2[i]
                label = labels[i].item()  # Assuming labels is a tensor

                # Check if embedding is already visualized
                if tuple(emb1) not in embeddings and tuple(emb2) not in embeddings:
                    embeddings.extend([emb1, emb2])
                    labels_list.extend([label, label])

                    # Process and store images
                    img_tensor1 = denormalize(img1[i].unsqueeze(0)).squeeze(0)  # Remove batch dim for visualization
                    img_tensor2 = denormalize(img2[i].unsqueeze(0)).squeeze(0)
                    img1_pil = transforms.ToPILImage()(img_tensor1).convert("RGB")
                    img2_pil = transforms.ToPILImage()(img_tensor2).convert("RGB")
                    images_list.append(wandb.Image(img1_pil, caption=f"Label: {label}"))
                    images_list.append(wandb.Image(img2_pil, caption=f"Label: {label}"))

    # Create a DataFrame-like structure to log to W&B
    data = [[emb, label, img] for emb, label, img in zip(embeddings, labels_list, images_list)]
    table = wandb.Table(data=data, columns=["Embedding", "Label", "Image"])

    # Log the table to WandB
    wandb.log({"Embedding Visualization": table})



def train(model, train_loader, train_dataset, optimizer, criterion, device, epochs, val_loader, log_interval=50):
    visualize_embeddings(model, val_loader, device)
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

def check_for_saved_pairs(directory):
    """ Check if the saved pairs data exists and log the status. """
    pairs_filename = os.path.join(directory, "pairs.pkl")
    if os.path.exists(pairs_filename):
        print("Loading from saved pairs.")
        return True
    else:
        print("No saved pairs found. Generating new pairs.")
        return False
    
def load_pairs(file_path):
    print("loading files")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            image_pairs, labels, weights = pickle.load(f)
        return image_pairs, labels, weights
    else:
        raise FileNotFoundError("The specified file does not exist.")

def main():
    seed_everything()
    data_path = 'scraping_output/wathec.json'
    raw_data = load_json_data(data_path)
    processed_data = prepare_dataframe(raw_data)
    grouped_images = group_images_by_brand(processed_data)
    val_brands = ['Tissot', 'Omega', 'Blancpain', 'Hamilton']

    train_data, val_data = stratified_split_data(grouped_images, val_brands=val_brands)
    
    # Check for saved pairs before initializing datasets
    saved_pairs_dir = "saved_pairs"
    check_for_saved_pairs(saved_pairs_dir)

    wandb.init(project='contrastive_loss', entity='DISPRO2', config={"learning_rate": 0.001, "epochs": 10, "batch_size": 32, "max_pairs": 1000})
    train_dataset = PairDataset(train_data, save_dir=saved_pairs_dir)
    val_dataset = PairDataset(val_data, save_dir=saved_pairs_dir)

    train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=4)

    embedding_size = 128
    model = WatchEmbeddingModel(embedding_size=embedding_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = ContrastiveLoss(margin=1.0, loss_scale=3)
    optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

    train(model, train_loader, train_dataset, optimizer, criterion, device, wandb.config.epochs, val_loader)

    # Visualize embeddings
    visualize_embeddings(model, val_loader, device)

    wandb.finish()

if __name__ == "__main__":
    main()
