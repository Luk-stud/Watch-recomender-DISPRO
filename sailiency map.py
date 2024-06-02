import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2





# Create VGG model with additional fully connected layers and adjustable embedding size

def create_vgg_model(num_classes, fc_layer_size, embedding_size, weight_decay=0.0005, dropout_rate=0.5):
    vgg = VGG16(include_top=False, input_shape=(224, 224, 3))
    
    for layer in vgg.layers[:-4]:  # Unfreeze the last 4 layers of the VGG16 model
        layer.trainable = False
    
    for layer in vgg.layers[-4:]:
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

# Instantiate the model
num_classes = 92  # Ensure this matches the number of classes during training
fc_layer_size = 128
embedding_size = 10

model = create_vgg_model(num_classes, fc_layer_size, embedding_size)
model.load_weights('trained_vgg_model.h5')  # Load the trained weights
model = remove_classification_layer(model)

# Load and preprocess an image
img_path = 'scraping_output/images/Omega/aqua-terra/Omega_-_2512.30.00_Seamaster_Aqua_Terra_150M_Automatic_42.2_Chronograph_Stainless_Steel___Silver___Bracelet_Case.jpg'
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

# Convert to tensor and enable gradient computation
input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

# Forward pass to get embeddings and compute gradients
with tf.GradientTape() as tape:
    tape.watch(input_tensor)
    embeddings = model(input_tensor)
    loss = tf.norm(embeddings, ord=2)  # Calculate norm to get a single scalar value

# Compute gradients
grads = tape.gradient(loss, input_tensor)

# Generate saliency map
saliency = tf.reduce_max(tf.abs(grads), axis=-1).numpy().squeeze()

# Plotting the saliency map
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency, cmap='hot')
plt.title('Saliency Map')
plt.colorbar()
plt.axis('off')
plt.show()
