import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.preprocessing import image
import numpy as np

class FeatureExtractor:
    def __init__(self, model_name='vgg16'):
        self.model_name = model_name
        self.model = self._load_model()
        
    def _load_model(self):
        if self.model_name == 'vgg16':
            return VGG16(weights='imagenet', include_top=False)
        elif self.model_name == 'resnet50':
            return ResNet50(weights='imagenet', include_top=False)
        elif self.model_name == 'inception':
            return InceptionV3(weights='imagenet', include_top=False)
            
    def extract_features(self, img_path):
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        features = self.model.predict(x)
        return features.flatten()