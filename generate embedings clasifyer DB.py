
#trained_vgg_model.h5
#embeddings_clasifier_vibrant_blaze_33.json
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import numpy as np
import json
import os

class WatchDataset(tf.data.Dataset):
    def __new__(cls, image_paths, transform=None):
        cls.image_paths = image_paths
        cls.transform = transform
        dataset = tf.data.Dataset.from_tensor_slices(image_paths)
        dataset = dataset.filter(cls.file_exists)
        dataset = dataset.map(cls.load_and_preprocess_image)
        return dataset

    @staticmethod
    def file_exists(img_path):
        def _py_file_exists(img_path):
            return np.array(os.path.exists(img_path.numpy().decode('utf-8')), dtype=np.bool_)
        return tf.py_function(_py_file_exists, [img_path], tf.bool)

    @staticmethod
    def load_and_preprocess_image(img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = preprocess_input(img)
        return img, img_path

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = self.add_weight(name='precision', initializer='zeros')
        self.recall = self.add_weight(name='recall', initializer='zeros')
        self.f1 = self.add_weight(name='f1', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(tf.round(y_pred), 'float32')

        tp = tf.reduce_sum(tf.cast(y_true * y_pred, 'float32'))
        predicted_positives = tf.reduce_sum(tf.cast(y_pred, 'float32'))
        possible_positives = tf.reduce_sum(tf.cast(y_true, 'float32'))

        precision = tp / (predicted_positives + tf.keras.backend.epsilon())
        recall = tp / (possible_positives + tf.keras.backend.epsilon())

        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

        self.precision.assign(precision)
        self.recall.assign(recall)
        self.f1.assign(f1)

    def result(self):
        return self.f1

    def reset_state(self):
        self.precision.assign(0)
        self.recall.assign(0)
        self.f1.assign(0)

def create_embeddings(model, dataset, device, image_info):
    embeddings_dict = {}
    for image, img_path in dataset:
        img_path = img_path.numpy().decode('utf-8')
        print(f"Processing image {img_path}...")
        image = tf.expand_dims(image, axis=0)  # Add batch dimension

        # Debug: Check the image tensor being fed into the model
        print(f"Image tensor shape: {image.shape}")
        print(f"Image tensor stats - min: {tf.reduce_min(image).numpy()}, max: {tf.reduce_max(image).numpy()}, mean: {tf.reduce_mean(image).numpy()}")

        output = model.predict(image)
        embedding = output.flatten().tolist()  # Convert to list

        # Add brand and family information
        brand = image_info[img_path]['Brand']
        family = image_info[img_path]['Family']

        embeddings_dict[img_path] = {
            'embedding': embedding,
            'brand': brand,
            'family': family
        }

        # Debug: Print the embedding to check if it is changing
        print(f"Embedding for {img_path}: {embedding[:5]}...")  # Print first 5 elements of the embedding for debugging

    return embeddings_dict

def save_embeddings(embeddings_dict, output_file='embeddings_clasifier_vibrant_blaze_33.json'):
    with open(output_file, 'w') as f:
        json.dump(embeddings_dict, f)
    print(f"Embeddings saved to {output_file}")

def main_knn(weights_path):
    data_path = 'data/watches_database_main.json'
    with open(data_path, 'r') as file:
        raw_data = json.load(file)
    
    processed_data = [{key.replace(':', '').strip(): value for key, value in item.items()} for item in raw_data]
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

    dataset = WatchDataset(image_paths)

    embedding_size = 30
    custom_objects = {'F1Score': F1Score}

    # Load the model with custom objects
    model = load_model(weights_path, custom_objects=custom_objects)
    model = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)  # Assuming the last layer is softmax

    device = 'cpu'  # or 'cuda' if you want to run on GPU

    # Create embeddings for all watches
    embeddings_dict = create_embeddings(model, dataset, device, image_info)
    
    # Save embeddings to a JSON file
    save_embeddings(embeddings_dict)

# Example of calling main_knn with a specific model path and query index
if __name__ == "__main__":
    model_weights_path = "trained_vgg_model.h5"  # Change to your actual model path
    main_knn(model_weights_path)
