import json
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
import numpy as np
import wandb
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import os

import os
import json
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import numpy as np
import wandb
import tensorflow as tf
from tensorflow.keras.regularizers import l2

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def prepare_dataframe(data):
    return [{key.replace(':', '').strip(): value for key, value in item.items()} for item in data]

def group_images_by_brand(data):
    grouped = {}
    for item in data:
        brand = item['Brand']
        image_path = item['Image Path']
        grouped.setdefault(brand, []).append(image_path)
    return grouped

# Model creation function
def create_vgg_model(num_classes = 92, fc_layer_size = 128, embedding_size = 10, weight_decay=0.0005, dropout_rate=0.5):
    vgg = VGG16(include_top=False, input_shape=(224, 224, 3))
    
    for layer in vgg.layers[:-2]:  # Unfreeze the last 4 layers of the VGG16 model
        layer.trainable = False
    
    for layer in vgg.layers[-2:]:
        layer.trainable = True
    
    x = Flatten()(vgg.output)
    x = Dense(fc_layer_size, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(fc_layer_size // 3, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(embedding_size, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=vgg.input, outputs=output)
    
    return model

def remove_classification_layer(model):
    return Model(inputs=model.input, outputs=model.layers[-3].output)

def denormalize(image, mean, std):
    image = image * std + mean
    return image

def visualize_brand_embeddings(model, grouped_images, val_brands, transform, mean, std):
    print("Visualizing brand embeddings...")
    embeddings = []
    brands_list = []
    images_list = []

    # Collect images from specified brands
    images_to_process = {}
    for brand in val_brands:
        if grouped_images.get(brand):
            images_to_process[brand] = grouped_images[brand]
            print(f"Found {len(grouped_images[brand])} images for brand {brand}")
        else:
            print(f"No images found for brand {brand}")

    # Process each image
    for brand, image_paths in images_to_process.items():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            image = img_to_array(image)  # Convert PIL image to array
            image = tf.convert_to_tensor(image)  # Convert array to TensorFlow tensor
            image = transform(image)
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            embedding_output = model.predict(image)[0]  # Get the embedding and remove batch dimension
            embeddings.append(embedding_output)
            brands_list.append(brand)

            # Process and store images
            img_tensor = denormalize(image.squeeze(0), mean, std)  # Remove batch dimension for visualization
            img_pil = tf.keras.preprocessing.image.array_to_img(img_tensor)
            images_list.append(wandb.Image(img_pil, caption=f"Brand: {brand}"))

    # Debugging: Print number of embeddings and images
    print(f"Number of embeddings: {len(embeddings)}")
    print(f"Number of images: {len(images_list)}")

    # Check if embeddings and images were collected correctly
    if not embeddings or not images_list:
        print("No embeddings or images collected.")
        return

    # Create a DataFrame-like structure to log to W&B
    data = [[emb.tolist(), br, img] for emb, br, img in zip(embeddings, brands_list, images_list)]
    table = wandb.Table(data=data, columns=["Embedding", "Brand", "Image"])

    # Log the table to WandB
    wandb.log({"Brand Embedding Visualization": table})

if __name__ == "__main__":
    wandb.init(project='visualization_project', entity='DISPRO2', config={"task": "Embedding Visualization"})

    device = tf.device("cuda" if tf.config.list_physical_devices('GPU') else "cpu")
    embedding_size = 10  # Ensure this matches the fully connected layer size used during training

    # Initialize model and load pre-trained weights
    num_classes = 50  # Make sure this matches the number of classes during training
    model = create_vgg_model()
    model.load_weights('trained_vgg_model.h5')  # Load the trained weights
    model = remove_classification_layer(model)
    
    # Load and process data
    data_path = 'data/watches_database_main.json'
    raw_data = load_json_data(data_path)
    processed_data = prepare_dataframe(raw_data)
    grouped_images = group_images_by_brand(processed_data)
    val_brands = ['Tissot', 'Omega']

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Resizing(224, 224),
        tf.keras.layers.experimental.preprocessing.Rescaling(1./224),
        tf.keras.layers.experimental.preprocessing.Normalization(mean=mean, variance=np.square(std))
    ])

    # Visualize embeddings for validation brands
    visualize_brand_embeddings(model, grouped_images, val_brands, transform, mean, std)

    # Finish the wandb run
    wandb.finish()
