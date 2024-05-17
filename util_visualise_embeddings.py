import torch
import wandb
import torchvision.transforms as transforms
from PIL import Image


def visualize_brand_embeddings(model, grouped_images, device, val_brands, transform, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    print("Visualizing brand embeddings...")
    model.eval()
    embeddings = []
    brands_list = []  # List to store brand names correctly for each image
    images_list = []

    # Denormalization function
    def denormalize(tensor):
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    # Collect images from specified brands
    images_to_process = {}
    for brand in val_brands:
        if grouped_images.get(brand):
            images_to_process[brand] = grouped_images[brand]
            print(f"Found {len(grouped_images[brand])} images for brand {brand}")
        else:
            print(f"No images found for brand {brand}")

    # Process each image
    with torch.no_grad():
        for brand, image_paths in images_to_process.items():
            for image_path in image_paths:
                image = Image.open(image_path).convert("RGB")
                image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and send to device
                embedding_output = model(image).cpu().numpy()[0]  # Get the embedding and remove batch dimension
                embeddings.append(embedding_output)
                brands_list.append(brand)  # Append the correct brand name for this image

                # Process and store images
                img_tensor = denormalize(image.squeeze(0))  # Remove batch dimension for visualization
                img_pil = transforms.ToPILImage()(img_tensor).convert("RGB")
                images_list.append(wandb.Image(img_pil, caption=f"Brand: {brand}"))

    # Create a DataFrame-like structure to log to W&B
    data = [[emb.tolist(), br, img] for emb, br, img in zip(embeddings, brands_list, images_list)]
    table = wandb.Table(data=data, columns=["Embedding", "Brand", "Image"])

    # Log the table to WandB
    wandb.log({"Brand Embedding Visualization": table})

def visualize_collection_embeddings(model, dataset, device):
    print("Visualizing collection embeddings...")
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for image, label in dataset:
            image = image.unsqueeze(0).to(device)  # Add batch dimension and move to device
            embedding = model(image).squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy
            embeddings.append(embedding)
            labels.append(label)

    # Log to Weights & Biases
    data = [[emb.tolist(), label] for emb, label in zip(embeddings, labels)]
    table = wandb.Table(data=data, columns=["Embedding", "Collection"])
    wandb.log({"Collection Embeddings": table})

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
        for anchors, _, _, labels, _ in val_loader:  # Adjust unpacking here
            anchors = anchors.to(device)
            embeddings_output = model(anchors).cpu().numpy()
            embeddings.extend(embeddings_output)

            # Ensure labels are properly handled
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()
            labels_list.extend(labels)

            # Process anchor images for visualization
            for img_tensor in anchors.cpu():
                # Convert tensor to PIL Image, denormalize and convert
                img = transforms.ToPILImage()(denormalize(img_tensor)).convert("RGB")
                images_list.append(wandb.Image(img, caption=f"Label: {labels_list[-1]}"))

    # Create a DataFrame-like structure to log to W&B
    data = [[emb, label, img] for emb, label, img in zip(embeddings, labels_list, images_list)]
    table = wandb.Table(data=data, columns=["Embedding", "Label", "Image"])

    # Log the table to WandB
    wandb.log({"Embedding Visualization": table})