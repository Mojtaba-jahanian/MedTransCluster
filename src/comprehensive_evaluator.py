import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class ComprehensiveEvaluator:
    def __init__(self, results_dir='results', analysis_dir='analysis'):
        self.results_dir = results_dir
        self.analysis_dir = analysis_dir
        os.makedirs(analysis_dir, exist_ok=True)
        
    def evaluate_all_results(self):
        """ارزیابی جامع تمام نتایج"""
        # خواندن نتایج از فایل مقایسه
        comparison_file = os.path.join(self.results_dir, 'model_comparison.csv')
        if not os.path.exists(comparison_file):
            raise FileNotFoundError("نتایج مقایسه یافت نشد!")
            
        df = pd.read_csv(comparison_file)
        
        # تولید گزارش‌ها و نمودارها
        self._generate_model_comparison(df)
        self._generate_clustering_analysis(df)
        self._generate_performance_metrics(df)
        self._generate_summary_report(df)
        
    def _generate_model_comparison(self, df):
        """مقایسه عملکرد مدل‌های مختلف"""
        # 1. نمودار مقایسه‌ای کلی
        fig = make_subplots(rows=3, cols=1,
                           subplot_titles=['Silhouette Score', 
                                         'Calinski-Harabasz Score',
                                         'Davies-Bouldin Score'])
        
        metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        
        for idx, metric in enumerate(metrics, 1):
            model_scores = df.groupby('Model')[metric].mean()
            
            fig.add_trace(
                go.Bar(x=model_scores.index, 
                      y=model_scores.values,
                      name=metric,
                      text=np.round(model_scores.values, 3),
                      textposition='auto'),
                row=idx, col=1
            )
            
        fig.update_layout(height=1200, width=800,
                         title_text="Model Performance Comparison")
        fig.write_html(os.path.join(self.analysis_dir, 'model_comparison.html'))
        
        # 2. نمودار رادار برای مقایسه چند بعدی
        fig = go.Figure()
        
        # نرمال‌سازی معیارها
        normalized_df = df.copy()
        for metric in metrics:
            if metric != 'Davies-Bouldin Score':  # برای این معیار، مقدار کمتر بهتر است
                normalized_df[metric] = (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
            else:
                normalized_df[metric] = 1 - (df[metric] - df[metric].min()) / (df[metric].max() - df[metric].min())
        
        avg_scores = normalized_df.groupby('Model')[metrics].mean()
        
        for model in avg_scores.index:
            fig.add_trace(go.Scatterpolar(
                r=avg_scores.loc[model],
                theta=metrics,
                name=model,
                fill='toself'
            ))
            
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Multi-dimensional Model Comparison"
        )
        fig.write_html(os.path.join(self.analysis_dir, 'radar_comparison.html'))

    def _generate_clustering_analysis(self, df):
        """تحلیل روش‌های خوشه‌بندی"""
        clustering_methods = df['Clustering Method'].unique()
        
        # 1. مقایسه روش‌های خوشه‌بندی برای هر مدل
        plt.figure(figsize=(15, 8))
        sns.boxplot(data=df, x='Clustering Method', y='Silhouette Score', hue='Model')
        plt.xticks(rotation=45)
        plt.title('Clustering Methods Performance Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'clustering_comparison.png'))
        plt.close()
        
        # 2. نمودار حرارتی عملکرد
        pivot_table = df.pivot_table(
            values='Silhouette Score',
            index='Model',
            columns='Clustering Method',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, cmap='RdYlBu', fmt='.3f')
        plt.title('Model-Clustering Method Performance Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'performance_heatmap.png'))
        plt.close()

    def _generate_performance_metrics(self, df):
        """تولید معیارهای عملکرد تفصیلی"""
        metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        
        # 1. جدول خلاصه آماری
        summary_stats = df.groupby('Model')[metrics].agg(['mean', 'std', 'min', 'max'])
        summary_stats.to_csv(os.path.join(self.analysis_dir, 'performance_statistics.csv'))
        
        # 2. نمودار جعبه‌ای برای هر معیار
        plt.figure(figsize=(15, 5))
        for idx, metric in enumerate(metrics, 1):
            plt.subplot(1, 3, idx)
            sns.boxplot(data=df, x='Model', y=metric)
            plt.xticks(rotation=45)
            plt.title(f'{metric} Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(self.analysis_dir, 'metrics_distribution.png'))
        plt.close()

    def _generate_summary_report(self, df):
        """تولید گزارش خلاصه"""
        report = []
        metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        
        # بهترین مدل برای هر معیار
        for metric in metrics:
            if metric != 'Davies-Bouldin Score':
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmin()
            
            best_row = df.loc[best_idx]
            report.append({
                'Metric': metric,
                'Best Model': best_row['Model'],
                'Best Clustering': best_row['Clustering Method'],
                'Score': best_row[metric]
            })
        
        # ذخیره گزارش
        pd.DataFrame(report).to_csv(os.path.join(self.analysis_dir, 'best_models.csv'), index=False)
        
        # تولید گزارش HTML
        html_content = """
        <html>
        <head>
            <title>Deep Learning Models Evaluation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 30px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>Deep Learning Models Evaluation Report</h1>
            
            <div class="section">
                <h2>Model Performance Comparison</h2>
                <img src="model_comparison.png" width="100%">
            </div>
            
            <div class="section">
                <h2>Clustering Analysis</h2>
                <img src="clustering_comparison.png" width="100%">
                <img src="performance_heatmap.png" width="100%">
            </div>
            
            <div class="section">
                <h2>Performance Metrics Distribution</h2>
                <img src="metrics_distribution.png" width="100%">
            </div>
            
            <div class="section">
                <h2>Interactive Visualizations</h2>
                <ul>
                    <li><a href="model_comparison.html">Detailed Model Comparison</a></li>
                    <li><a href="radar_comparison.html">Multi-dimensional Comparison</a></li>
                </ul>
            </div>
            
            <div class="section">
                <h2>Best Performing Models</h2>
                <iframe src="best_models.csv" width="100%" height="200px"></iframe>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(self.analysis_dir, 'evaluation_report.html'), 'w') as f:
            f.write(html_content)