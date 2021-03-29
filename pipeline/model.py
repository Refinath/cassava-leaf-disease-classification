import torch.nn as nn
import config
import timm


class CustomEfficientNet(nn.Module):
    def __init__(self, model_name=config.model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, config.target_size)

    def forward(self, x):
        x = self.model(x)
        return x
