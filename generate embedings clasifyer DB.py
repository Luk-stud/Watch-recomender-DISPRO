import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Define the model class
class WatchClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(WatchClassificationModel, self).__init__()
        self.base_model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        self.base_model.classifier[1] = nn.Linear(self.base_model.classifier[1].in_features, num_classes)
        
    def forward(self, x):
        return self.base_model(x)

# Dataset class to load images
class WatchDataset(Dataset):
    def __init__(self, image_paths, image_info, transform=None):
        self.image_paths = image_paths
        self.image_info = image_info
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if not os.path.exists(img_path):
            return None, None
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

# Custom collate function to filter out None entries
def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return torch.utils.data.default_collate(batch)

# Function to create embeddings
def create_embeddings(model, dataloader, device, image_info):
    model.eval()
    embeddings_dict = {}

    with torch.no_grad():
        for images, img_paths in tqdm(dataloader):
            images = images.to(device)
            outputs = model(images)

            for img_path, output in zip(img_paths, outputs):
                embedding = output.cpu().numpy().flatten().tolist()
                brand = image_info[img_path]['Brand']
                family = image_info[img_path]['Family']
                embeddings_dict[img_path] = {
                    'embedding': embedding,
                    'brand': brand,
                    'family': family
                }

    return embeddings_dict

# Function to save embeddings
def save_embeddings(embeddings_dict, output_file='embeddings_classifier_family_v2.json'):
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f)
    print(f"Embeddings saved to {output_file}")

# Function to load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Function to prepare the dataframe
def prepare_dataframe(data):
    return [{key.replace(':', '').strip(): value for key, value in item.items()} for item in data]

# Function to check image paths
def check_image_paths(df, image_column):
    return df[df[image_column].apply(os.path.exists)]

def main_knn(weights_path):
    data_path = 'data/watches_database_main.json'
    raw_data = load_json_data(data_path)

    processed_data = prepare_dataframe(raw_data)
    grouped_images = {item['Brand']: [] for item in processed_data}
    for item in processed_data:
        grouped_images[item['Brand']].append(item['Image Path'])

    # Create a mapping from image paths to brand and family information
    image_info = {}
    for entry in raw_data:
        img_path = entry['Image Path']
        image_info[img_path] = {
            'Brand': entry.get('Brand:', entry.get('Brand', '')),
            'Family': entry.get('Family:', entry.get('Family', ''))
        }

    # Flatten the grouped images into a list of image paths
    image_paths = [img for brand_images in grouped_images.values() for img in brand_images]

    dataset = WatchDataset(image_paths, image_info)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model and set it to evaluation mode
    model = WatchClassificationModel(num_classes=70)
    model.base_model.classifier[1] = nn.Identity()  # Removing the last layer to get embeddings
    model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model = model.to(device)

    # Create embeddings for all watches
    embeddings_dict = create_embeddings(model, dataloader, device, image_info)

    # Save embeddings to a JSON file
    save_embeddings(embeddings_dict)

# Example of calling main_knn with a specific model path
if __name__ == "__main__":
    model_weights_path = "best_family_model.pth"  # Change to your actual model path
    main_knn(model_weights_path)
