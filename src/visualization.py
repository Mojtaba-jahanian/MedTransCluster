import os
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .utils import create_directory
import numpy as np

class Visualizer:
    """Visualization tools for clustering results."""
    
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        create_directory(output_dir)
        
    def plot_clusters(self, features_reduced, results):
        """Generate interactive plots for each clustering method."""
        for method, labels in results.items():
            df = pd.DataFrame({
                'PC1': features_reduced[:, 0],
                'PC2': features_reduced[:, 1],
                'Cluster': labels
            })
            
            fig = px.scatter(df, x='PC1', y='PC2', color='Cluster',
                           title=f'Clustering Results using {method}',
                           template='plotly_white')
            
            fig.write_html(os.path.join(self.output_dir, f'{method}_clustering.html'))
            
    def plot_comparison(self, features_reduced, results):
        """Create comparison plot of all clustering methods."""
        plt.figure(figsize=(20, 5))
        for idx, (method, labels) in enumerate(results.items(), 1):
            plt.subplot(1, 4, idx)
            plt.scatter(features_reduced[:, 0], features_reduced[:, 1], 
                       c=labels, cmap='viridis')
            plt.title(method)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'comparison.png'))
        plt.close()
        
    def plot_evaluation(self, scores):
        """Plot evaluation metrics."""
        plt.figure(figsize=(10, 5))
        plt.bar(scores.keys(), scores.values())
        plt.title('Clustering Quality (Silhouette Score)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(self.output_dir, 'evaluation.png'))
        plt.close()

class AdvancedVisualizer:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        
    def plot_model_comparison(self, analysis_results):
        """Plot comprehensive model comparison"""
        # Create comparison plots for each metric
        metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        
        fig = make_subplots(rows=len(metrics), cols=1,
                           subplot_titles=metrics,
                           vertical_spacing=0.1)
        
        for idx, metric in enumerate(metrics, 1):
            data = analysis_results.pivot(index='Model', 
                                        columns='Clustering Method',
                                        values=metric)
            
            heatmap = go.Heatmap(z=data.values,
                                x=data.columns,
                                y=data.index,
                                colorscale='RdYlBu',
                                text=np.round(data.values, 3),
                                texttemplate='%{text}',
                                textfont={"size": 10},
                                showscale=True)
            
            fig.add_trace(heatmap, row=idx, col=1)
            
        fig.update_layout(height=1200, width=800,
                         title_text="Model Comparison Across Clustering Methods")
        fig.write_html(f"{self.output_dir}/model_comparison.html")
        
    def plot_clustering_results(self, features_reduced, labels, model_name, method_name):
        """Plot interactive clustering results"""
        df = pd.DataFrame({
            'PC1': features_reduced[:, 0],
            'PC2': features_reduced[:, 1],
            'Cluster': labels
        })
        
        fig = px.scatter(df, x='PC1', y='PC2', color='Cluster',
                        title=f'Clustering Results: {model_name} - {method_name}',
                        template='plotly_white')
        
        fig.write_html(f"{self.output_dir}/{model_name}_{method_name}_clustering.html") 