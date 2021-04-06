# Cassava leaf disease classification

The idea of this competition is to build a Classifier Model to find the leaf disease type and it has almost 5 class and a im-balanced Class

- [Link to the competition](https://www.kaggle.com/c/cassava-leaf-disease-classification])
- [Github discussion forum😀](https://github.com/p-s-vishnu/cassava-leaf-disease-classification/discussions)

< build-status >

< Problem statement >

< leaf-diseases-GIF>



## Installation

```python
pip install cassava
```





## Example inference

```python
import cv2
from cassava.pretrained import get_model

image = cv2.imread("<insert your image path here>")

# Use cassava.list_models() to see list of all available pretrained models with metrics
model = cassava.get_model("")
model.predict(image: np.array)
>> [{"image_id":str, "class": int,"probability": float}]

model.predict(images: list)
>> [{"image_id":str, "class": int,"probability": float},
>>  {"image_id":str, "class": int,"probability": float},
>>  ...]
```
# Our Pipeline
    1.Model Architecture - Efficeientnet-B4 , Noisy Weights
    2.Image Size         - 512
    3.Optimizer          - Adam
    4.Scheduler          - GradualWarumUpScheduler
    5.Loss               - Focal Cosine Loss
    6.Augmentations      - Hard Augmentations
    7.Epochs             - 10
    8.Early Stopping     - No
    9.Mixed Precision    - Yes

Try Jupyter notebook in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( )

Try Jupyter notebook in Kaggle: [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]()

## Training

> Hyperparameters are defined in the config file.



### Prepare data



### Edit configurations



### Run train script



## Inference

< Inference script GIF>



## Web app

< Web app GIF>

[Heroku Link]()

[AWS link]()

[Code for Web app]( )


## Blog

[Medium link]()


## Acknowledgements

We would like to thank Kaggle community as a whole for providing an avenue to learn and discuss latest data science/machine learning advancements but a hat tip to whose code was used / who inspired us.

1. Teranus
2. Nakama

- - - -
<a href="https://www.buymeacoffee.com/vpkprasanna" target="_blank"><img src="https://github.com/appcraftstudio/buymeacoffee/raw/master/Images/snapshot-bmc-button.png" width="300"></a>

Copyright © 2021 Prasanna Kumar. All rights reserved
