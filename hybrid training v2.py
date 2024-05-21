import torch
import wandb
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from PIL import Image
import os
import pickle
import random
from util_div import *
from util_model import *
from util_visualise_embeddings import *
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Function to split data into training and validation sets based on the provided brands
def stratified_split_data(grouped_images, val_brands=None, min_samples=2):
    print("Splitting data...")
    # Filter out classes with fewer than min_samples
    filtered_grouped_images = {brand: images for brand, images in grouped_images.items() if len(images) >= min_samples}
    
    # Split data into training and validation sets
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

        # Calculate weights based on the number of images per brand
        self.brand_weights = self.calculate_brand_weights()

        # Create a mapping from brand names to numerical labels
        self.brand_to_idx = {brand: idx for idx, brand in enumerate(self.grouped_images.keys())}

        if not os.path.exists(self.pairs_filename):
            self.generate_pairs()
            self.save_pairs()
        else:
            self.load_pairs()
            # Validate loaded pairs
            if not self.validate_pairs():
                print(f"Invalid pairs loaded for {dataset_type}. Regenerating pairs.")
                self.generate_pairs()
                self.save_pairs()

        self.apply_max_pairs()

    def calculate_brand_weights(self):
        brand_weights = {}
        total_images = sum(len(images) for images in self.grouped_images.values())
        for brand, images in self.grouped_images.items():
            brand_weights[brand] = total_images / len(images)
        return brand_weights

    def generate_pairs(self):
        self.image_pairs = []
        self.labels = []
        self.weights = []
        self.brand_labels = []
        brands = list(self.grouped_images.keys())
        temp_pairs = []

        for brand, images in self.grouped_images.items():
            num_similar = int(len(images) * (len(images) - 1) / 2 * self.sim_ratio)
            num_dissimilar = int(num_similar * (1 / self.sim_ratio - 1))

            brand_weight = self.brand_weights[brand]

            for _ in range(num_similar):
                i, j = np.random.choice(len(images), 2, replace=False)
                temp_pairs.append(((images[i], images[j]), 1, brand_weight, self.brand_to_idx[brand]))

            other_brands = [b for b in brands if b != brand]
            for _ in range(num_dissimilar):
                other_brand = np.random.choice(other_brands)
                other_images = self.grouped_images[other_brand]
                img1 = np.random.choice(images)
                img2 = np.random.choice(other_images)
                temp_pairs.append(((img1, img2), 0, brand_weight, self.brand_to_idx[other_brand]))

        # Shuffle all pairs to remove order bias
        random.shuffle(temp_pairs)

        # Unpack shuffled pairs back into class variables
        for pair, label, weight, brand_label in temp_pairs:
            self.image_pairs.append(pair)
            self.labels.append(label)
            self.weights.append(weight)
            self.brand_labels.append(brand_label)

    def save_pairs(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        with open(self.pairs_filename, 'wb') as f:
            pickle.dump((self.image_pairs, self.labels, self.weights, self.brand_labels), f)

    def load_pairs(self):
        with open(self.pairs_filename, 'rb') as f:
            data = pickle.load(f)
            if len(data) == 3:
                self.image_pairs, self.labels, self.weights = data
                self.brand_labels = [self.infer_brand_label(pair) for pair in self.image_pairs]
            else:
                self.image_pairs, self.labels, self.weights, self.brand_labels = data

    def infer_brand_label(self, pair):
        for brand, images in self.grouped_images.items():
            if pair[0] in images or pair[1] in images:
                return self.brand_to_idx[brand]
        return -1  # Indicating unknown brand

    def validate_pairs(self):
        return all(brand != -1 for brand in self.brand_labels)

    def apply_max_pairs(self):
        if self.max_pairs is not None and len(self.image_pairs) > self.max_pairs:
            indices = np.random.choice(len(self.image_pairs), self.max_pairs, replace=False)
            self.image_pairs = [self.image_pairs[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
            self.weights = [self.weights[i] for i in indices]
            self.brand_labels = [self.brand_labels[i] for i in indices]

    def __getitem__(self, idx):
        img1_path, img2_path = self.image_pairs[idx]
        img1, img2 = Image.open(img1_path).convert('RGB'), Image.open(img2_path).convert('RGB')
        if self.transform:
            img1, img2 = self.transform(img1), self.transform(img2)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        weight = torch.tensor(self.weights[idx], dtype=torch.float32)
        brand_label = torch.tensor(self.brand_labels[idx], dtype=torch.int64)
        return img1, img2, label, weight, brand_label

    def __len__(self):
        return len(self.image_pairs)

    def hard_mining(self, model, device, num_hard_pairs, batch_size=128):
        print("Performing hard mining...")
        self.image_pairs = []
        self.labels = []
        self.weights = []
        self.brand_labels = []

        model.eval()
        for brand, images in self.grouped_images.items():
            brand_weight = self.brand_weights[brand]
            num_images = len(images)
            img_tensors = []
            for img in images:
                img_tensor = self.transform(Image.open(img).convert('RGB')).unsqueeze(0)
                img_tensors.append(img_tensor)
            img_tensors = torch.cat(img_tensors).to(device)

            # Process images in smaller batches
            embeddings = []
            with torch.no_grad():
                for i in range(0, num_images, batch_size):
                    batch_tensors = img_tensors[i:i + batch_size]
                    batch_embeddings = model(batch_tensors)
                    embeddings.append(batch_embeddings)
            embeddings = torch.cat(embeddings)

            dists = torch.cdist(embeddings, embeddings)
            dists = dists.cpu().numpy()

            num_similar = min(num_hard_pairs, len(images) * (len(images) - 1) // 2)
            num_dissimilar = num_hard_pairs - num_similar

            for _ in range(num_similar):
                i, j = np.unravel_index(np.argmax(dists), dists.shape)
                if i != j:
                    self.image_pairs.append((images[i], images[j]))
                    self.labels.append(1)
                    self.weights.append(brand_weight)
                    self.brand_labels.append(self.brand_to_idx[brand])
                dists[i, j] = -1

            other_brands = [b for b in self.grouped_images.keys() if b != brand]
            for _ in range(num_dissimilar):
                other_brand = np.random.choice(other_brands)
                other_images = self.grouped_images[other_brand]
                img1 = np.random.choice(images)
                img2 = np.random.choice(other_images)
                self.image_pairs.append((img1, img2))
                self.labels.append(0)
                self.weights.append(brand_weight)
                self.brand_labels.append(self.brand_to_idx[other_brand])

        self.apply_max_pairs()
        self.save_pairs()
# Initialize weights for the model
def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# Contrastive Loss Function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, loss_scale=3, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.loss_scale = loss_scale
        self.temperature = temperature

    def forward(self, output1, output2, labels, weights):
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        loss_similar = (1 - labels) * weights * (euclidean_distance ** 2)
        loss_dissimilar = labels * weights * (self.margin - euclidean_distance).clamp(min=0) ** 2
        contrastive_loss = torch.mean(loss_similar + loss_dissimilar)
        return contrastive_loss * self.loss_scale / self.temperature

class OrthogonalRegularization(nn.Module):
    def __init__(self, lambda_orth=1e-4):
        super(OrthogonalRegularization, self).__init__()
        self.lambda_orth = lambda_orth

    def forward(self, embeddings):
        dot_product = torch.matmul(embeddings, embeddings.t())
        identity = torch.eye(embeddings.size(0)).to(embeddings.device)
        orth_loss = torch.norm(dot_product - identity, p='fro')
        return self.lambda_orth * orth_loss

# Class-Balanced Loss Function
class ClassBalancedLoss(nn.Module):
    def __init__(self, brands_count, beta=0.9999):
        super(ClassBalancedLoss, self).__init__()
        self.brands_count = brands_count
        self.beta = beta

    def forward(self, output1, output2, labels, weights, brand_labels):
        effective_num = 1.0 - np.power(self.beta, self.brands_count)
        weights_cb = (1.0 - self.beta) / effective_num
        weights_cb = weights_cb / np.sum(weights_cb) * len(self.brands_count)
        weights_cb = torch.tensor(weights_cb, dtype=torch.float32).to(output1.device)

        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_similar = (1 - labels) * (euclidean_distance ** 2)
        loss_dissimilar = labels * (1 - euclidean_distance).clamp(min=0) ** 2

        loss = (loss_similar + loss_dissimilar) * weights_cb[brand_labels]
        return loss.mean()

# Custom early stopping class
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def train(model, train_loader, optimizer, scheduler, initial_criterion, final_criterion, switch_epoch, device, epochs, val_loader, early_stopping, train_dataset, orthogonal_regularizer, log_interval=1):
    print("Starting training...")
    criterion = initial_criterion

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        print(f"Epoch {epoch+1} starts...")

        # Switch criterion if epoch >= switch_epoch
        if epoch >= switch_epoch:
            criterion = final_criterion
            print(f"Switched to final loss function at Epoch {epoch+1}")

        for i, (img1, img2, labels, weights, brand_labels) in enumerate(train_loader):
            img1, img2 = img1.to(device), img2.to(device)
            labels, weights = labels.to(device), weights.to(device)
            brand_labels = brand_labels.to(device)
            optimizer.zero_grad()
            output1 = model(img1)
            output2 = model(img2)
            if isinstance(criterion, ClassBalancedLoss):
                loss = criterion(output1, output2, labels, weights, brand_labels)
            else:
                loss = criterion(output1, output2, labels, weights)

            # Add orthogonal regularization loss
            orth_loss = orthogonal_regularizer(torch.cat((output1, output2), dim=0))
            loss += orth_loss

            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

            if i % log_interval == 0:
                print(f"At batch {i+1}, loss is {loss.item()}")
                wandb.log({"Epoch": epoch + 1, "Batch Training Loss": loss.item(), "Step": i})

        # Validation step
        val_loss = validate(model, val_loader, criterion, device, orthogonal_regularizer)
        wandb.log({"Epoch": epoch + 1, "Validation Loss": val_loss})

        # Early stopping check
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step(val_loss)  # Update the learning rate after each epoch

        # Save the model locally
        model_path = f"model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to {model_path}")

        # Log the model to wandb
        artifact = wandb.Artifact(f'model_epoch_{epoch + 1}', type='model')
        artifact.add_file(model_path)
        wandb.log_artifact(artifact)

        # Hard mining step
        if epoch < epochs - 1:
            train_dataset.hard_mining(model, device, num_hard_pairs=1000)
            
            # Debugging output to check dataset length after hard mining
            print(f"After hard mining, dataset length is {len(train_dataset)}")
            
            # Reinitialize the DataLoader with the updated dataset
            train_loader = DataLoader(train_dataset, batch_size=train_loader.batch_size, shuffle=True, num_workers=train_loader.num_workers)

    return model

def validate(model, val_loader, criterion, device, orthogonal_regularizer):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img1, img2, labels, weights, brand_labels in val_loader:
            img1, img2 = img1.to(device), img2.to(device)
            labels, weights = labels.to(device), weights.to(device)
            brand_labels = brand_labels.to(device)
            output1 = model(img1)
            output2 = model(img2)
            if isinstance(criterion, ClassBalancedLoss):
                loss = criterion(output1, output2, labels, weights, brand_labels)
            else:
                loss = criterion(output1, output2, labels, weights)

            # Add orthogonal regularization loss
            orth_loss = orthogonal_regularizer(torch.cat((output1, output2), dim=0))
            loss += orth_loss

            val_loss += loss.item()
    val_loss /= len(val_loader)
    return val_loss

# Function to create a weighted sampler for balanced mini-batches
def create_weighted_sampler(dataset):
    brand_labels = dataset.brand_labels
    # Ensure all brand labels are numeric
    valid_indices = [i for i, label in enumerate(brand_labels) if label >= 0]
    if not valid_indices:
        raise ValueError("No valid brand labels found for sampling.")
    valid_brand_labels = [brand_labels[i] for i in valid_indices]
    class_sample_count = np.array([len(np.where(np.array(valid_brand_labels) == t)[0]) for t in np.unique(valid_brand_labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[np.where(np.unique(valid_brand_labels) == t)[0][0]] for t in valid_brand_labels])
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    return sampler

# Function to check if saved pairs exist
def check_for_saved_pairs(directory, dataset_type):
    save_dir = os.path.join(directory, dataset_type)
    pairs_filename = os.path.join(save_dir, "pairs.pkl")
    if os.path.exists(pairs_filename):
        print(f"Loading from saved pairs for {dataset_type}.")
        return True
    else:
        print(f"No saved pairs found for {dataset_type}. Generating new pairs.")
        return False


def main(model_weights_path=None):
    wandb.init(project="DISPRO_hybrid_training_v2")

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

    train_data, val_data = stratified_split_data(grouped_images, val_brands=val_brands)
    saved_pairs_dir = "saved_pairs"
    check_for_saved_pairs(saved_pairs_dir, "train")
    check_for_saved_pairs(saved_pairs_dir, "val")

    train_dataset = PairDataset(train_data, save_dir=saved_pairs_dir, dataset_type="train", max_pairs=2000)
    val_dataset = PairDataset(val_data, save_dir=saved_pairs_dir, dataset_type="val", max_pairs=100)

    batch_size = 32
    num_workers = 4

    sampler = create_weighted_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WatchEmbeddingModel(embedding_size=30).to(device)

    # Initialize model weights
    model.apply(initialize_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
    initial_criterion = ContrastiveLoss(margin=1.0)
    final_criterion = ClassBalancedLoss(brands_count=[len(images) for images in train_data.values()], beta=0.9999)
    orthogonal_regularizer = OrthogonalRegularization(lambda_orth=1e-4)
    
    early_stopping = EarlyStopping(patience=7, verbose=True)

    if model_weights_path:
        model.load_state_dict(torch.load(model_weights_path))
        print(f"Loaded model weights from {model_weights_path}")

    trained_model = train(model, train_loader, optimizer, scheduler, initial_criterion, final_criterion, wandb.config.switch_epoch, device, wandb.config.epochs, val_loader, early_stopping, train_dataset, orthogonal_regularizer)

    visualize_brand_embeddings(model, grouped_images, device, val_brands, transform)

    wandb.finish()

if __name__ == "__main__":
    main()