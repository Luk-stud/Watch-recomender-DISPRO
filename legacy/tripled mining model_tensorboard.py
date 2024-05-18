import os
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from sklearn.model_selection import KFold
from PIL import Image
import shutil
import random

from torch.utils.tensorboard import SummaryWriter

# Define the path to your dataset
data_dir = 'scraping_output/images'

# Define transforms for image preprocessing
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.brands = os.listdir(data_dir)

    def __len__(self):
        return len(self.brands)

    def __getitem__(self, idx):
        # 
        brand_name = self.brands[idx]
        brand_dir = os.path.join(self.data_dir, brand_name)
        models = os.listdir(brand_dir)
        
        # Select anchor and positive images from the same model
        model_name = random.choice(models)
        model_dir = os.path.join(brand_dir, model_name)
        images = os.listdir(model_dir)
        
        anchor_img_name, positive_img_name = np.random.choice(images, 2, replace=False)
        anchor_img_path = os.path.join(model_dir, anchor_img_name)
        positive_img_path = os.path.join(model_dir, positive_img_name)
        anchor_img = Image.open(anchor_img_path)
        positive_img = Image.open(positive_img_path)
        
        # Select a negative image from a different brand
        other_brands = [b for b in self.brands if b != brand_name]
        negative_brand_name = random.choice(other_brands)
        negative_brand_dir = os.path.join(self.data_dir, negative_brand_name)
        negative_model_name = random.choice(os.listdir(negative_brand_dir))
        negative_model_dir = os.path.join(negative_brand_dir, negative_model_name)
        negative_img_name = random.choice(os.listdir(negative_model_dir))
        negative_img_path = os.path.join(negative_model_dir, negative_img_name)
        negative_img = Image.open(negative_img_path)
        
        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img

# Define VGG16 model with triplet loss
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        self.vgg16.classifier[-1] = nn.Linear(4096, 1000)  # Modify the last layer for your dataset
        self.fc = nn.Linear(1000, 256)  # Embedding size

    def forward(self, x):
        x = self.vgg16(x)
        x = self.fc(x)
        return x

# Triplet loss function
class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        loss = torch.relu(distance_positive - distance_negative + self.margin)
        return loss.mean()

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for anchor_inputs, positive_inputs, negative_inputs in dataloaders[phase]:
                anchor_inputs = anchor_inputs.to(device)
                positive_inputs = positive_inputs.to(device)
                negative_inputs = negative_inputs.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward
                with torch.set_grad_enabled(phase == 'train'):
                    anchor_outputs = model(anchor_inputs)
                    positive_outputs = model(positive_inputs)
                    negative_outputs = model(negative_inputs)
                    loss = criterion(anchor_outputs, positive_outputs, negative_outputs)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * anchor_inputs.size(0)

            epoch_loss = running_loss / dataset_sizes[phase]
            writer.add_scalar(f'{phase}/loss', epoch_loss, epoch)

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

        print()

    writer.close()

def safe_split_data(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Safely splits data into train, validation, and test sets and creates the necessary directories.

    Args:
        data_dir (str): Path to the main data directory.
        train_ratio (float): Ratio of data to be allocated for training.
        val_ratio (float): Ratio of data to be allocated for validation.
        test_ratio (float): Ratio of data to be allocated for testing.
        seed (int): Seed for random shuffling.

    Returns:
        None
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios should sum up to 1."

    # Create directories for train, val, and test sets
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Get list of brands
    brands = os.listdir(data_dir)

    # Shuffle brands
    random.seed(seed)
    random.shuffle(brands)

    for brand in brands:
        brand_dir = os.path.join(data_dir, brand)
        if os.path.isdir(brand_dir):
            models = os.listdir(brand_dir)
            num_models = len(models)
            num_train = int(train_ratio * num_models)
            num_val = int(val_ratio * num_models)

            # Create train set
            train_models = models[:num_train]
            for model in train_models:
                src_dir = os.path.join(brand_dir, model)
                dst_dir = os.path.join(train_dir, brand, model)
                os.makedirs(dst_dir, exist_ok=True)
                for img in os.listdir(src_dir):
                    shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))

            # Create val set
            val_models = models[num_train:num_train + num_val]
            for model in val_models:
                src_dir = os.path.join(brand_dir, model)
                dst_dir = os.path.join(val_dir, brand, model)
                os.makedirs(dst_dir, exist_ok=True)
                for img in os.listdir(src_dir):
                    shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))

            # Create test set
            test_models = models[num_train + num_val:]
            for model in test_models:
                src_dir = os.path.join(brand_dir, model)
                dst_dir = os.path.join(test_dir, brand, model)
                os.makedirs(dst_dir, exist_ok=True)
                for img in os.listdir(src_dir):
                    shutil.copy(os.path.join(src_dir, img), os.path.join(dst_dir, img))

# Split data
safe_split_data(data_dir)

# Create dataset
image_datasets = {x: CustomDataset(os.path.join(data_dir, x), transform=data_transforms[x]) for x in ['train', 'val']}

# Create dataloaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=8, shuffle=True, num_workers=4) for x in ['train', 'val']}

# Get dataset sizes
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize model, loss function, and optimizer
model = VGG16().to(device)
criterion = TripletLoss(margin=1.0)
optimizer = optim.Adam(model.parameters(), lr=0.001)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Train the model
train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=25)
