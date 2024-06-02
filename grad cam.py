import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib.pyplot as plt
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
import cv2

import json
import pandas as pd


from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2


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
    x = Dense(fc_layer_size // 2, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(embedding_size, activation='relu', kernel_regularizer=l2(weight_decay))(x)
    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(weight_decay))(x)
    model = Model(inputs=vgg.input, outputs=output)
    
    
    return model


# Instantiate the model
num_classes = 92  # Ensure this matches the number of classes during training
fc_layer_size = 256
embedding_size = 10

model = create_vgg_model(num_classes, fc_layer_size, embedding_size)
model.load_weights('trained_vgg_model.h5')  # Load the trained weights

# Load and preprocess an image
# img_path = 'scraping_output/images/Ralph_Lauren/safari/Ralph_Lauren_-_RLR0250702_Safari_39mm_Chronometer_Aged_Steel___Khaki_Case.jpg'
img_path = 'scraping_output/images/Omega/aqua-terra/Omega_-_2512.30.00_Seamaster_Aqua_Terra_150M_Automatic_42.2_Chronograph_Stainless_Steel___Silver___Bracelet_Case.jpg'
img = load_img(img_path, target_size=(224, 224))
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

# Grad-CAM
def grad_cam_viz(model, img_array, layer_name):
    gradcam = Gradcam(model,
                      model_modifier=ReplaceToLinear(),
                      clone=True)
    score = CategoricalScore([np.argmax(model.predict(img_array))])
    cam = gradcam(score, img_array, penultimate_layer=layer_name)
    
    return cam[0]

# Generate Grad-CAM heatmap
layer_name = 'block5_conv3'  # VGG16 last convolutional layer
heatmap = grad_cam_viz(model, img_array, layer_name)

# Normalize the heatmap
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# Superimpose the heatmap on the original image
original_img = cv2.imread(img_path)
original_img = cv2.resize(original_img, (224, 224))
superimposed_img = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

# Plotting the Grad-CAM heatmap
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(load_img(img_path, target_size=(224, 224)))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(superimposed_img)
plt.title('Grad-CAM')
plt.axis('off')
plt.show()
