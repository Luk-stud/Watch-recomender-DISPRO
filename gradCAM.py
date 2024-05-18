import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Define your custom model
class WatchEmbeddingModel(torch.nn.Module):
    def __init__(self, embedding_size, train_deep_layers=True):
        super(WatchEmbeddingModel, self).__init__()
        base_model = vgg16(weights=VGG16_Weights.DEFAULT)
        if train_deep_layers:
            for param in base_model.features[-3:].parameters():
                param.requires_grad = True
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.embedder = torch.nn.Sequential(
            torch.nn.Linear(512 * 7 * 7, 4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.7),
            torch.nn.Linear(4096, embedding_size),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.embedder(x)
        return x

# Instantiate the model
model = WatchEmbeddingModel(embedding_size=200)

# Load the model weights
model_weights_path = "model_epoch_1.pth"
model.load_state_dict(torch.load(model_weights_path))

model.eval()

# Load an image and preprocess it
img_path = 'scraping_output/images/Archimede/1950-s\Archimede_-_UA8068BMP-A1.1_1950-2_Stainless_Steel___Silver___Mesh_Case.jpg'
img = Image.open(img_path).convert('RGB')
display_img = img.resize((224, 224))
rgb_img = np.array(display_img)[:, :, ::-1] / 255.0

preprocess = Compose([
    Resize((224, 224)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
input_tensor = preprocess(img).unsqueeze(0)

# Enable gradients to be computed with respect to input image
input_tensor.requires_grad = True

# Forward pass to get embeddings
embeddings = model(input_tensor)
embeddings = embeddings.norm(p=2)  # Calculate norm to get a single scalar value
embeddings.backward()  # Compute gradients

# Generate saliency map
saliency = input_tensor.grad.data.abs().squeeze().max(0)[0].cpu().numpy()

# Plotting the saliency map
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img.resize((224, 224)))
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(saliency, cmap='hot')
plt.title('Saliency Map')
plt.colorbar()
plt.axis('off')
plt.show()