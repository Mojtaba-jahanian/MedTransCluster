import os
import requests
from tqdm import tqdm
import hashlib

class ModelDownloader:
    def __init__(self):
        self.model_urls = {
            'VGG16': 'https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'VGG19': 'https://storage.googleapis.com/tensorflow/keras-applications/vgg19/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'ResNet50': 'https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'InceptionV3': 'https://storage.googleapis.com/tensorflow/keras-applications/inception_v3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'MobileNetV2': 'https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5',
            'DenseNet121': 'https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
            'EfficientNetB0': 'https://storage.googleapis.com/tensorflow/keras-applications/efficientnetb0/efficientnetb0_notop.h5'
        }
        
        self.keras_dir = os.path.expanduser('~/.keras')
        self.models_dir = os.path.join(self.keras_dir, 'models')
        os.makedirs(self.models_dir, exist_ok=True)

    def download_model(self, model_name):
        """Download specific model weights"""
        if model_name not in self.model_urls:
            print(f"Model {model_name} not found in available models")
            return False
            
        url = self.model_urls[model_name]
        filename = url.split('/')[-1]
        
        try:
            self.download_with_resume(url, filename)
            print(f"✓ {model_name} weights downloaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error downloading {model_name} weights: {str(e)}")
            return False

    def download_with_resume(self, url, filename):
        """Download file with resume capability"""
        local_filename = os.path.join(self.models_dir, filename)
        
        # Check if file exists and is complete
        if os.path.exists(local_filename):
            return local_filename
            
        # Streaming download with progress bar
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        # Open temp file for writing
        temp_filename = local_filename + '.temp'
        initial_pos = 0
        
        # Check if partial download exists
        if os.path.exists(temp_filename):
            initial_pos = os.path.getsize(temp_filename)
            headers = {'Range': f'bytes={initial_pos}-'}
            response = requests.get(url, headers=headers, stream=True)

        mode = 'ab' if initial_pos > 0 else 'wb'
        
        with open(temp_filename, mode) as f:
            with tqdm(total=total_size, initial=initial_pos,
                     unit='iB', unit_scale=True, desc=filename) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        # Rename temp file to final filename
        os.rename(temp_filename, local_filename)
        return local_filename

    def download_all_models(self):
        """Download all model weights"""
        for model_name in self.model_urls:
            self.download_model(model_name)