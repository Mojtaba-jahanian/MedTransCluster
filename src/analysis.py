import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import time

class ModelAnalyzer:
    def __init__(self):
        self.results = {}
        
    def analyze_model(self, model_name, features, clustering_results):
        """Analyze clustering results for a specific model"""
        model_metrics = {
            'n_features': features.shape[1],
            'processing_time': 0,
            'clustering_scores': {}
        }
        
        for method, labels in clustering_results.items():
            if len(np.unique(labels)) > 1:
                scores = {
                    'silhouette': silhouette_score(features, labels),
                    'calinski': calinski_harabasz_score(features, labels),
                    'davies': davies_bouldin_score(features, labels)
                }
                model_metrics['clustering_scores'][method] = scores
                
        self.results[model_name] = model_metrics
        return model_metrics
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        # Create list to store rows
        rows = []
        
        for model, metrics in self.results.items():
            for method, scores in metrics['clustering_scores'].items():
                row = {
                    'Model': model,
                    'Clustering Method': method,
                    'Silhouette Score': scores['silhouette'],
                    'Calinski-Harabasz Score': scores['calinski'],
                    'Davies-Bouldin Score': scores['davies'],
                    'Number of Features': metrics['n_features']
                }
                rows.append(row)
        
        # Create DataFrame from list of dictionaries
        report = pd.DataFrame(rows)
        return report 