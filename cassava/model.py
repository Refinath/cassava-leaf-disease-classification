from typing import Dict, Tuple

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from cassava import augment, config


class BaseClassifier(nn.Module):
    """
    Contains all the common methods for all the classifier classes
    """

    def __init__(self):
        super().__init__()
        self.transform = augment.get_transforms(data="test")

    def predit_as_json(self, image: np.ndarray, threshold: float = 0.5) -> Dict:

        trans_image = self.transform(image=image)["image"]
        trans_image = trans_image.unsqueeze(0)

        with torch.no_grad():
            self.eval()
            probability_mat = F.softmax(self(trans_image), dim=1).numpy().squeeze()  # 1x5
            class_idx = int(np.argmax(probability_mat))
            class_name = config.LABEL_MAP[class_idx]

        if probability_mat[class_idx] <= threshold:
            return {"class_name": "Not a Cassava leaf!", "confidence": None}
        return {"class_name": class_name, "confidence": probability_mat[class_idx]}


class CassavaClassifier(BaseClassifier):
    def __init__(self, model_name=config.MODEL_NAME, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, config.TARGET_SIZE)

    def forward(self, x):
        x = self.model(x)
        return x


# class EnsembleNet(BaseClassifier):
#     """
#     A model to return average of multiple model
#     https://discuss.pytorch.org/t/combining-trained-models-in-pytorch/28383
#     """

#     def __init__(self, models: Tuple):
#         super().__init__()
#         self.models = models

#     def forward(self, x):
#         preds = [model(x) for model in self.models]  # TODO: verify
#         preds = torch.cat(preds)
#         return torch.mean(preds)
