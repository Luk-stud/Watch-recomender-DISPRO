import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def resize_and_crop(img):
    img = img.astype(np.uint8)  # Ensure the numpy array is of type uint8
    img = Image.fromarray(img)
    img = img.resize((600, 600))  # Increase the initial image size for better quality
    img = img.crop((150, 150, 450, 450))  # Stronger center crop
    
    # Add stronger random crop
    img = img.resize((600, 600))  # Resize before applying the random crop
    crop_size = 600*0.6  # Stronger random crop size
    max_x = 600 - crop_size
    max_y = 600 - crop_size
    x = np.random.randint(0, max_x)
    y = np.random.randint(0, max_y)
    img = img.crop((x, y, x + crop_size, y + crop_size))
    img = img.resize((224, 224))  # Resize to 224x224 after the random crop
    
    img = np.array(img).astype(np.float32)  # Convert back to float32 after processing
    return img

def visualize_crop(image_path):
    # Load the image
    original_image = Image.open(image_path)
    original_image = original_image.resize((600, 600))  # Resize to 600x600 for consistency
    original_image_np = np.array(original_image)
    
    # Apply resize and crop
    cropped_image_np = resize_and_crop(original_image_np)
    
    # Convert numpy arrays back to images
    cropped_image = Image.fromarray(cropped_image_np.astype('uint8'))
    
    # Plot the original and processed images side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image (600x600)')
    axes[0].axis('off')
    
    axes[1].imshow(cropped_image)
    axes[1].set_title('Cropped Image (224x224)')
    axes[1].axis('off')
    
    plt.show()
# Example usage
image_path = 'scraping_output/images/Robot_Watch/graphic/Robot_Watch_-_1901ST07_Graphic_Sutnar_Chameleon_Green_Case.jpg'  # Replace with the path to your image
visualize_crop(image_path)
