# Cassava leaf disease classification

[Github discussion forum😀](https://github.com/p-s-vishnu/cassava-leaf-disease-classification/discussions)

<build-status>

<Problem statement>

<leaf-diseases-GIF>



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



Try Jupyter notebook in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]( )

Try Jupyter notebook in Kaggle: [![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)]()



## Training

> Hyperparameters are defined in the config file. 



### Prepare data



### Edit configurations



### Run train script





## Web app

<Web app GIF>

[Heroku Link]()

[AWS link]() 

[Code for Web app]( )



## Blog

[Medium link]()



## Acknowledgements

We would like to thank Kaggle community as a whole for providing an avenue to learn and discuss latest data science/machine learning advancements but a hat tip to whose code was used / who inspired us.

1. Teranus  
2. Nakama 
3. 