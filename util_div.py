import torch
import numpy as np
import json
import os
import random

# Set a seed for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Prepare data from JSON
def prepare_dataframe(data):
    return [{key.replace(':', '').strip(): value for key, value in item.items()} for item in data]

# Group images by brand
def group_images_by_brand(data):
    grouped = {}
    for item in data:
        brand = item['Brand']
        image_path = item['Image Path']
        grouped.setdefault(brand, []).append(image_path)
    return grouped

