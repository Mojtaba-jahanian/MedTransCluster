import os
import json
import pickle
import numpy as np

class CheckpointManager:
    def __init__(self, checkpoint_dir='checkpoints'):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.progress_file = os.path.join(checkpoint_dir, 'progress.json')
        self.initialize_progress()

    def initialize_progress(self):
        """Initialize or load progress tracking"""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as f:
                self.progress = json.load(f)
        else:
            self.progress = {
                'completed_models': [],
                'current_model': None,
                'features_extracted': {}
            }
            self.save_progress()

    def save_progress(self):
        """Save current progress"""
        with open(self.progress_file, 'w') as f:
            json.dump(self.progress, f)

    def save_features(self, model_name, features, image_paths):
        """Save extracted features for a model"""
        features_file = os.path.join(self.checkpoint_dir, f'{model_name}_features.pkl')
        with open(features_file, 'wb') as f:
            pickle.dump({
                'features': features,
                'image_paths': image_paths
            }, f)
        self.progress['features_extracted'][model_name] = True
        self.save_progress()

    def load_features(self, model_name):
        """Load saved features for a model"""
        features_file = os.path.join(self.checkpoint_dir, f'{model_name}_features.pkl')
        if os.path.exists(features_file):
            with open(features_file, 'rb') as f:
                data = pickle.load(f)
            return data['features'], data['image_paths']
        return None, None

    def save_clustering_results(self, model_name, results):
        """Save clustering results"""
        results_file = os.path.join(self.checkpoint_dir, f'{model_name}_clustering.pkl')
        with open(results_file, 'wb') as f:
            pickle.dump(results, f)
        self.progress['completed_models'].append(model_name)
        self.save_progress()

    def load_clustering_results(self, model_name):
        """Load saved clustering results"""
        results_file = os.path.join(self.checkpoint_dir, f'{model_name}_clustering.pkl')
        if os.path.exists(results_file):
            with open(results_file, 'rb') as f:
                return pickle.load(f)
        return None 