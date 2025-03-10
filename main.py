from src.feature_extractor import FeatureExtractor
from src.clustering import ImageClustering
import os
import numpy as np

def main():
    # Initialize feature extractor
    extractor = FeatureExtractor(model_name='vgg16')
    
    # Initialize clustering
    clustering = ImageClustering(method='kmeans', n_clusters=5)
    
    # Extract features and perform clustering
    features_list = []
    image_paths = []  # Add your image paths here
    
    for img_path in image_paths:
        features = extractor.extract_features(img_path)
        features_list.append(features)
    
    # Perform clustering
    clusters = clustering.fit_predict(np.array(features_list))
    
    print("Clustering completed!")

if __name__ == "__main__":
    main()