import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16, EfficientNetB0
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from PIL import Image
import numpy as np
import wandb
import tensorflow as tf
from wandb.integration.keras import WandbMetricsLogger
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils import resample
from collections import Counter
import pickle
import shutil


# Custom F1 score metric class
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

# Configuration for wandb sweeps
config = {
    'batch_size': 64,
    'fc_layer_size': 2048,
    'embedding_size': 20,
    'weight_decay': 0.0007,
    'dropout_rate': 0.6,
    'learning_rate': 0.0001,
    'epochs': 200,
    'early_stopping_patience': 20,
    'reduce_lr_factor': 0.7,
    'reduce_lr_patience': 5,
    'min_lr': 0.0000001,
    'max_images_per_family': 30  # Maximum number of images per family before oversampling
}

# Initialize wandb with the configuration
wandb.init(project='image-family-classification', config=config)
config = wandb.config

# Verify that TensorFlow is using the GPU
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# Set GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

# Load JSON data
def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Prepare data from JSON
def prepare_dataframe(data):
    return [{key.replace(':', '').strip(): value for key, value in item.items()} for item in data]

# Prepare data
file_path = 'data/watches_database_main.json'  # Replace with your file path
data = load_json_data(file_path)
data = prepare_dataframe(data)

df = pd.DataFrame(data)

# Check if image paths exist
def check_image_paths(df, image_column):
    valid_rows = []
    for _, row in df.iterrows():
        if os.path.exists(row[image_column]):
            valid_rows.append(row)
    return pd.DataFrame(valid_rows)

df = check_image_paths(df, 'Image Path')

# Load the LabelEncoder for subsequent runs
with open('label_encoder.pkl', 'rb') as file:
    family_encoder = pickle.load(file)
df['Family'] = df['Family'].apply(lambda x: x.strip() if pd.notnull(x) else x)
df['Family'] = family_encoder.fit_transform(df['Family'])
df['Family'] = df['Family'].apply(str)  # Convert the 'Family' column to string

# Filter out families with fewer than 5 images
min_images_per_family = 5  # Minimum number of images required per family
family_counts = df['Family'].value_counts()
valid_families = family_counts[family_counts >= min_images_per_family].index
df = df[df['Family'].isin(valid_families)]

# Split the data by family
def split_by_family(df):
    train_df, temp_df = train_test_split(df, test_size=0.4, stratify=df['Family'], random_state=13)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['Family'], random_state=13)
    return train_df, val_df, test_df

train_df, val_df, test_df = split_by_family(df)

# Correct num_classes calculation to ensure it matches the range of labels
num_classes = len(family_encoder.classes_)

# Limit the number of validation images with stratified sampling
max_validation_images = 2000  # Set your desired limit here
if len(val_df) > max_validation_images:
    sss = StratifiedShuffleSplit(n_splits=1, test_size=max_validation_images, random_state=42)
    for _, val_idx in sss.split(val_df, val_df['Family']):
        val_df = val_df.iloc[val_idx]

# Function to resize and crop images
def resize_and_crop(image):
    image = tf.image.central_crop(image, 0.5)  # Central crop to focus on the center of the image
    image = tf.image.random_crop(image, size=[224, 224, 3])  # Random crop to 224x224
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    return image

# Image preprocessing and data generator functions
def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(600, 600))  # Load image at higher resolution
    image = img_to_array(image)
    image = resize_and_crop(image)  # Apply resizing and cropping
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def image_generator(df, batch_size, is_training):
    while True:
        batch_paths = []
        batch_labels = []
        for _, row in df.sample(n=batch_size, replace=True).iterrows():
            batch_paths.append(row['Image Path'])
            batch_labels.append(int(row['Family']))
        batch_images = np.array([load_and_preprocess_image(image_path) for image_path in batch_paths])
        batch_labels = to_categorical(batch_labels, num_classes=num_classes)
        yield batch_images, batch_labels

# Function to cap and oversample minority classes
def cap_and_oversample_data(df, target_column, max_images_per_family):
    df_capped = df.groupby(target_column).apply(lambda x: x.sample(min(len(x), max_images_per_family))).reset_index(drop=True)
    majority_class = df_capped[target_column].mode()[0]  # Majority class
    majority_count = df_capped[target_column].value_counts().max()

    # Separate majority and minority classes
    df_majority = df_capped[df_capped[target_column] == majority_class]
    df_minority = df_capped[df_capped[target_column] != majority_class]

    # Upsample minority classes
    oversampled_minority = []
    for class_value in df_minority[target_column].unique():
        df_class = df_capped[df_capped[target_column] == class_value]
        df_class_upsampled = resample(df_class, replace=True, n_samples=majority_count, random_state=42)
        oversampled_minority.append(df_class_upsampled)

    # Combine majority class with oversampled minority classes
    oversampled_df = pd.concat([df_majority] + oversampled_minority)

    return oversampled_df

# Cap and oversample the training set
train_df_oversampled = cap_and_oversample_data(train_df, 'Family', config.max_images_per_family)

def image_generator_oversampled(df, batch_size, is_training):
    while True:
        batch_paths = []
        batch_labels = []
        for _, row in df.sample(n=batch_size, replace=True).iterrows():
            batch_paths.append(row['Image Path'])
            batch_labels.append(int(row['Family']))
        batch_images = np.array([load_and_preprocess_image(image_path) for image_path in batch_paths])
        batch_labels = to_categorical(batch_labels, num_classes=num_classes)
        yield batch_images, batch_labels

# Model creation function
def create_vgg_model(base_model, num_classes, fc_layer_size, embedding_size, weight_decay=0.0005, dropout_rate=0.5):
    x = base_model.output
    x = Flatten()(x)
    x = Dense(fc_layer_size, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(fc_layer_size // 3, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(embedding_size, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=base_model.input, outputs=output)
    
    return model

def create_base_vgg_model():
    vgg = VGG16(include_top=False, input_shape=(224, 224, 3))
    
    for layer in vgg.layers[:-2]:  # Unfreeze the last 4 layers of the VGG16 model
        layer.trainable = False
    
    for layer in vgg.layers[-2:]:
        layer.trainable = True
    
    return vgg

# Initialize and compile model
base_vgg_model = create_base_vgg_model()
model = create_vgg_model(base_vgg_model, num_classes, config.fc_layer_size, config.embedding_size, config.weight_decay, config.dropout_rate)

# Function to load weights if available
def load_model_weights(model, weights_path):
    if os.path.exists(weights_path):
        model.load_weights(weights_path, by_name=True, skip_mismatch=True)
        print(f"Model weights loaded from {weights_path}")
    else:
        print(f"No weights file found at {weights_path}. Starting with random weights.")

# Load model weights if they exist
weights_path = ''  # Path to the weights file
load_model_weights(model, weights_path)

# Rebuild the classification layer based on the updated classes
x = model.layers[-5].output  # Assuming -5 is the last layer before the classification layer
output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(config.weight_decay))(x)
model = Model(inputs=model.input, outputs=output)

# Compile the model with SGD optimizer
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy', F1Score()])

# Training and validation data generators
train_generator = image_generator_oversampled(train_df_oversampled, config.batch_size, is_training=True)
val_generator = image_generator(val_df, config.batch_size, is_training=False)

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=config.reduce_lr_factor, patience=config.reduce_lr_patience, min_lr=config.min_lr)

# Train model with callbacks
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_df_oversampled) // config.batch_size,
    epochs=config.epochs,
    validation_data=val_generator,
    validation_steps=len(val_df) // config.batch_size,
    callbacks=[early_stopping, reduce_lr, WandbMetricsLogger()]
)

# Evaluate the model
test_generator = image_generator(test_df, config.batch_size, is_training=False)
evaluation_results = model.evaluate(test_generator, steps=len(test_df) // config.batch_size)

# Unpack the evaluation results
test_loss = evaluation_results[0]
test_accuracy = evaluation_results[1]
test_f1_score = evaluation_results[2]

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
print(f'Test F1 Score: {test_f1_score:.4f}')


model_save_path = "claify_family.h5"

# Save the model
model.save(weights_path)

# Function to copy files instead of creating symlinks
def copy_to_wandb(filepath):
    filename = os.path.basename(filepath)
    destination_path = os.path.join(wandb.run.dir, filename)
    shutil.copy(filepath, destination_path)

# Save the model to wandb
copy_to_wandb(weights_path)

# Save the label encoder
label_encoder_path = 'label_encoder.pkl'
with open(label_encoder_path, 'wb') as f:
    pickle.dump(family_encoder, f)

# Save the label encoder to wandb
copy_to_wandb(label_encoder_path)

# Log the final model performance metrics to wandb
wandb.log({
    "test_loss": test_loss,
    "test_accuracy": test_accuracy,
    "test_f1_score": test_f1_score
})

# End the wandb run
wandb.finish()