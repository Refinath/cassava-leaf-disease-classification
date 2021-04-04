import config
import timm
import torch.nn as nn


class CustomEfficientNet(nn.Module):
    def __init__(self, model_name=config.MODEL_NAME, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, config.TARGET_SIZE)

    def forward(self, x):
        x = self.model(x)
        return x
