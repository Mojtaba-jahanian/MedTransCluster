import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ResultsAnalyzer:
    def __init__(self, results_dir='results', analysis_dir='analysis'):
        self.results_dir = results_dir
        self.analysis_dir = analysis_dir
        os.makedirs(analysis_dir, exist_ok=True)
        
    def generate_comprehensive_report(self, comparison_report):
        """Generate comprehensive analysis report"""
        print("\nGenerating comprehensive analysis report...")
        
        try:
            # 1. Model Performance Analysis
            print("- Analyzing model performance...")
            self._analyze_model_performance(comparison_report)
            
            # 2. Clustering Analysis
            print("- Analyzing clustering methods...")
            self._analyze_clustering_methods(comparison_report)
            
            # 3. Generate Summary Report
            print("- Generating summary report...")
            self._generate_summary_report(comparison_report)
            
            print(f"\nAnalysis complete! Results saved in: {self.analysis_dir}")
            
        except Exception as e:
            print(f"Error during analysis: {str(e)}")
            raise

    def _analyze_model_performance(self, df):
        """Analyze and visualize model performance"""
        metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        
        # Performance comparison plot
        fig = make_subplots(rows=len(metrics), cols=1,
                           subplot_titles=metrics)
        
        for idx, metric in enumerate(metrics, 1):
            model_scores = df.groupby('Model')[metric].mean()
            
            fig.add_trace(
                go.Bar(
                    x=model_scores.index,
                    y=model_scores.values,
                    text=np.round(model_scores.values, 3),
                    textposition='auto',
                    name=metric
                ),
                row=idx, col=1
            )
            
        fig.update_layout(height=1000, width=800,
                         title_text="Model Performance Comparison")
        fig.write_html(os.path.join(self.analysis_dir, 'model_performance.html'))

    def _analyze_clustering_methods(self, df):
        """Analyze clustering methods performance"""
        # Clustering methods comparison
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='Clustering Method', y='Silhouette Score', hue='Model')
        plt.xticks(rotation=45)
        plt.title('Clustering Methods Performance')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'clustering_comparison.png'))
        plt.close()
        
        # Performance heatmap
        pivot = df.pivot_table(
            values='Silhouette Score',
            index='Model',
            columns='Clustering Method',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, cmap='RdYlBu', fmt='.3f')
        plt.title('Model-Clustering Performance Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'performance_heatmap.png'))
        plt.close()

    def _generate_summary_report(self, df):
        """Generate summary report"""
        # Best models for each metric
        metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        summary = []
        
        for metric in metrics:
            best_idx = df[metric].idxmax()
            best_row = df.loc[best_idx]
            summary.append({
                'Metric': metric,
                'Best Model': best_row['Model'],
                'Best Method': best_row['Clustering Method'],
                'Score': best_row[metric]
            })
            
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(os.path.join(self.analysis_dir, 'summary.csv'), index=False)
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Analysis Results</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .section {{ margin-bottom: 30px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; }}
            </style>
        </head>
        <body>
            <h1>Model Analysis Results</h1>
            
            <div class="section">
                <h2>Model Performance</h2>
                <iframe src="model_performance.html" width="100%" height="800px"></iframe>
            </div>
            
            <div class="section">
                <h2>Clustering Comparison</h2>
                <img src="clustering_comparison.png" width="100%">
            </div>
            
            <div class="section">
                <h2>Performance Heatmap</h2>
                <img src="performance_heatmap.png" width="100%">
            </div>
            
            <div class="section">
                <h2>Best Models Summary</h2>
                <iframe src="summary.csv" width="100%" height="200px"></iframe>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(self.analysis_dir, 'report.html'), 'w') as f:
            f.write(html_content) 