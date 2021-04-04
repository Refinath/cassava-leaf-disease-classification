"""This module includes the augmentation pipeline"""

import config
from albumentations import (
    Compose,
    HorizontalFlip,
    Normalize,
    RandomResizedCrop,
    Resize,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2


def get_transforms(*, data):
    """Returns augmentation specific to train/valid data"""
    if data == "train":
        return Compose(
            [
                RandomResizedCrop(config.size, config.size),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                ShiftScaleRotate(p=0.5),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    if data == "valid":
        return Compose(
            [
                Resize(config.size, config.size),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )
    return Exception("Unimplemented data transform!")
