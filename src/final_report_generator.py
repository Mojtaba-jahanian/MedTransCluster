import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt

class FinalReportGenerator:
    def __init__(self, results_dir='results', report_dir='final_report'):
        self.results_dir = results_dir
        self.report_dir = report_dir
        os.makedirs(report_dir, exist_ok=True)
        
    def generate_final_report(self):
        """تولید گزارش نهایی جامع"""
        # خواندن نتایج
        df = pd.read_csv(os.path.join(self.results_dir, 'model_comparison.csv'))
        
        # 1. خلاصه اجرایی
        self._generate_executive_summary(df)
        
        # 2. مقایسه جامع مدل‌ها
        self._generate_model_comparison(df)
        
        # 3. تحلیل خوشه‌بندی
        self._generate_clustering_analysis(df)
        
        # 4. توصیه‌های کاربردی
        self._generate_recommendations(df)
        
        # 5. تولید گزارش HTML نهایی
        self._compile_final_report()
        
    def _generate_executive_summary(self, df):
        """تولید خلاصه اجرایی"""
        metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        summary = []
        
        # بهترین مدل برای هر معیار
        for metric in metrics:
            if metric != 'Davies-Bouldin Score':
                best_idx = df[metric].idxmax()
            else:
                best_idx = df[metric].idxmin()
            
            best_row = df.loc[best_idx]
            summary.append({
                'Metric': metric,
                'Best Model': best_row['Model'],
                'Best Method': best_row['Clustering Method'],
                'Score': best_row[metric]
            })
        
        # ذخیره خلاصه
        pd.DataFrame(summary).to_csv(
            os.path.join(self.report_dir, 'executive_summary.csv'),
            index=False
        )
        
    def _generate_model_comparison(self, df):
        """تولید مقایسه جامع مدل‌ها"""
        # 1. نمودار رادار برای مقایسه چند بعدی
        metrics = ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        model_scores = df.groupby('Model')[metrics].mean()
        
        # نرمال‌سازی امتیازها
        normalized_scores = (model_scores - model_scores.min()) / (model_scores.max() - model_scores.min())
        
        fig = go.Figure()
        for model in normalized_scores.index:
            fig.add_trace(go.Scatterpolar(
                r=normalized_scores.loc[model],
                theta=metrics,
                name=model,
                fill='toself'
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Multi-dimensional Model Comparison"
        )
        fig.write_html(os.path.join(self.report_dir, 'model_comparison_radar.html'))
        
    def _generate_clustering_analysis(self, df):
        """تحلیل روش‌های خوشه‌بندی"""
        # 1. مقایسه روش‌های خوشه‌بندی
        clustering_comparison = df.groupby('Clustering Method')[
            ['Silhouette Score', 'Calinski-Harabasz Score', 'Davies-Bouldin Score']
        ].mean()
        
        # نمودار مقایسه‌ای
        fig = make_subplots(rows=1, cols=3, subplot_titles=clustering_comparison.columns)
        
        for i, metric in enumerate(clustering_comparison.columns, 1):
            fig.add_trace(
                go.Bar(
                    x=clustering_comparison.index,
                    y=clustering_comparison[metric],
                    name=metric
                ),
                row=1, col=i
            )
            
        fig.update_layout(height=500, width=1200,
                         title_text="Clustering Methods Comparison")
        fig.write_html(os.path.join(self.report_dir, 'clustering_comparison.html'))
        
    def _generate_recommendations(self, df):
        """تولید توصیه‌های کاربردی"""
        recommendations = {
            'General Recommendations': [
                "1. برای تصاویر پزشکی قفسه سینه، مدل X بهترین عملکرد را در استخراج ویژگی‌ها نشان داد.",
                "2. روش خوشه‌بندی Y برای این نوع داده‌ها مناسب‌تر است.",
                "3. برای بهترین نتیجه، ترکیب مدل X و روش Y پیشنهاد می‌شود."
            ],
            'Use Cases': [
                "- تشخیص بیماری‌های ریوی",
                "- دسته‌بندی تصاویر رادیولوژی",
                "- پیش‌پردازش داده‌ها برای یادگیری عمیق"
            ],
            'Future Work': [
                "- بررسی مدل‌های جدیدتر",
                "- ترکیب روش‌های خوشه‌بندی",
                "- بهینه‌سازی پارامترها"
            ]
        }
        
        # ذخیره توصیه‌ها
        with open(os.path.join(self.report_dir, 'recommendations.txt'), 'w', encoding='utf-8') as f:
            for section, items in recommendations.items():
                f.write(f"\n{section}:\n")
                f.write('\n'.join(items))
                f.write('\n')
                
    def _compile_final_report(self):
        """تولید گزارش HTML نهایی"""
        html_content = """
        <html>
        <head>
            <title>Final Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .section { margin-bottom: 30px; }
                h1 { color: #2c3e50; }
                h2 { color: #34495e; }
                .metric { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>Deep Learning Models Analysis for Medical Chest X-rays</h1>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <iframe src="executive_summary.csv" width="100%" height="200px"></iframe>
            </div>
            
            <div class="section">
                <h2>Model Comparison</h2>
                <iframe src="model_comparison_radar.html" width="100%" height="600px"></iframe>
            </div>
            
            <div class="section">
                <h2>Clustering Analysis</h2>
                <iframe src="clustering_comparison.html" width="100%" height="600px"></iframe>
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <iframe src="recommendations.txt" width="100%" height="400px"></iframe>
            </div>
        </body>
        </html>
        """
        
        with open(os.path.join(self.report_dir, 'final_report.html'), 'w') as f:
            f.write(html_content) 