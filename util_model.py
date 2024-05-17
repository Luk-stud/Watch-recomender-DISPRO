
import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights


class WatchEmbeddingModel(nn.Module):
    def __init__(self, embedding_size, train_deep_layers=True):
        super(WatchEmbeddingModel, self).__init__()
        base_model = vgg16(weights=VGG16_Weights.DEFAULT)
        if train_deep_layers:
            for param in base_model.features[-3:].parameters():
                param.requires_grad = True
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.embedder = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.7),
            nn.Linear(4096, embedding_size),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.embedder(x)
        return x
    


