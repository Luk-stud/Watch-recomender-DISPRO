import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
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

# Sweep configuration
sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'batch_size': {'values': [16, 32, 64]},
        'fc_layer_size': {'values': [128, 256, 512]},
        'embedding_size': {'values': [10, 20, 30]},
        'weight_decay': {'values': [0.0001, 0.0005, 0.001]},
        'dropout_rate': {'values': [0.4, 0.5, 0.6]},
        'learning_rate': {'values': [0.0001, 0.00005, 0.00001]},
        'epochs': {'values': [30, 50, 70]},
        'early_stopping_patience': {'values': [5, 10, 15]},
        'reduce_lr_factor': {'values': [0.2, 0.5, 0.7]},
        'reduce_lr_patience': {'values': [3, 5, 7]},
        'min_lr': {'values': [0.00001, 0.000001, 0.0000001]}
    }
}

# Initialize sweep
sweep_id = wandb.sweep(sweep_config, project='image-brand-classification')

def load_json_data(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def prepare_dataframe(data):
    return [{key.replace(':', '').strip(): value for key, value in item.items()} for item in data]

def check_image_paths(df, image_column):
    valid_rows = []
    for _, row in df.iterrows():
        if os.path.exists(row[image_column]):
            valid_rows.append(row)
    return pd.DataFrame(valid_rows)

def filter_valid_families(df, min_families=2, min_images_per_family=2):
    valid_rows = []
    grouped = df.groupby('Brand')
    for brand, group in grouped:
        family_counts = group['Family'].value_counts()
        valid_families = family_counts[family_counts >= min_images_per_family].index
        if len(valid_families) >= min_families:
            valid_rows.append(group[group['Family'].isin(valid_families)])
    return pd.concat(valid_rows)

def split_by_family(df):
    train_rows = []
    val_rows = []
    test_rows = []
    grouped = df.groupby('Brand')

    for brand, group in grouped:
        families = group['Family'].unique()
        if len(families) >= 3:
            train_families, test_families = train_test_split(families, test_size=0.2, random_state=13)
            train_families, val_families = train_test_split(train_families, test_size=0.25, random_state=13)
        elif len(families) == 2:
            train_families, test_families = train_test_split(families, test_size=0.5, random_state=13)
            val_families = train_families
        else:
            train_families = families
            val_families = families
            test_families = families

        train_rows.append(group[group['Family'].isin(train_families)])
        val_rows.append(group[group['Family'].isin(val_families)])
        test_rows.append(group[group['Family'].isin(test_families)])

    train_df = pd.concat(train_rows)
    val_df = pd.concat(val_rows)
    test_df = pd.concat(test_rows)

    return train_df, val_df, test_df

def resize_and_crop(image):
    image = tf.image.resize(image, [600, 600])
    image = tf.image.central_crop(image, 0.5)
    image = tf.image.resize_with_crop_or_pad(image, 400, 400)
    image = tf.image.random_crop(image, size=[224, 224, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.3)
    image = tf.image.random_contrast(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.2)
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    return image

def load_and_preprocess_image(image_path):
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = resize_and_crop(image)
    image = tf.keras.applications.vgg16.preprocess_input(image)
    return image

def image_generator(df, batch_size, is_training, num_classes):
    while True:
        batch_paths = []
        batch_labels = []
        for _, row in df.sample(n=batch_size, replace=True).iterrows():
            batch_paths.append(row['Image Path'])
            batch_labels.append(int(row['Brand']))
        batch_images = np.array([load_and_preprocess_image(image_path) for image_path in batch_paths])
        batch_labels = to_categorical(batch_labels, num_classes=num_classes)
        yield batch_images, batch_labels

def create_vgg_model(num_classes, fc_layer_size, embedding_size, weight_decay=0.0005, dropout_rate=0.5):
    vgg = VGG16(include_top=False, input_shape=(224, 224, 3))

    for layer in vgg.layers[:-4]:
        layer.trainable = False

    for layer in vgg.layers[-4:]:
        layer.trainable = True

    x = Flatten()(vgg.output)
    x = Dense(fc_layer_size, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(fc_layer_size // 2, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(embedding_size, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=vgg.input, outputs=output)

    return model

def train():
    with wandb.init() as run:
        config = wandb.config

        file_path = 'data/watches_database_main.json'
        data = load_json_data(file_path)
        data = prepare_dataframe(data)

        df = pd.DataFrame(data)
        df = check_image_paths(df, 'Image Path')

        brand_encoder = LabelEncoder()
        df['Brand'] = brand_encoder.fit_transform(df['Brand'])
        df['Brand'] = df['Brand'].apply(str)
        df['Family'] = df['Family'].apply(lambda x: x.strip() if pd.notnull(x) else x)

        min_entries = 5
        brand_counts = df['Brand'].value_counts()
        valid_brands = brand_counts[brand_counts >= min_entries].index
        df = df[df['Brand'].isin(valid_brands)]

        df = filter_valid_families(df)

        train_df, val_df, test_df = split_by_family(df)
        num_classes = len(brand_encoder.classes_)

        max_validation_images = 2000
        if len(val_df) > max_validation_images:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=max_validation_images, random_state=42)
            for _, val_idx in sss.split(val_df, val_df['Brand']):
                val_df = val_df.iloc[val_idx]

        model = create_vgg_model(num_classes, config.fc_layer_size, config.embedding_size, config.weight_decay, config.dropout_rate)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

        train_generator = image_generator(train_df, config.batch_size, is_training=True, num_classes=num_classes)
        val_generator = image_generator(val_df, config.batch_size, is_training=False, num_classes=num_classes)

        early_stopping = EarlyStopping(monitor='val_loss', patience=config.early_stopping_patience, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=config.reduce_lr_factor, patience=config.reduce_lr_patience, min_lr=config.min_lr)

        history = model.fit(
            train_generator,
            steps_per_epoch=len(train_df) // config.batch_size,
            epochs=config.epochs,
            validation_data=val_generator,
            validation_steps=len(val_df) // config.batch_size,
            callbacks=[early_stopping, reduce_lr, WandbMetricsLogger()]
        )

        test_generator = image_generator(test_df, config.batch_size, is_training=False, num_classes=num_classes)
        test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_df) // config.batch_size)
        print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

        model.save('trained_vgg_model.h5')

        with open('label_encoder.pkl', 'wb') as f:
            import pickle
            pickle.dump(brand_encoder, f)

# Start the sweep
wandb.agent(sweep_id, function=train, count=20)
