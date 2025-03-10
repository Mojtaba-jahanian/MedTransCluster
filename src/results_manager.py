import os
import pandas as pd
import glob

class ResultsManager:
    def __init__(self, results_dir='results'):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def get_completed_models(self):
        """Get list of models that have complete results"""
        completed_models = set()
        
        # بررسی فایل‌های HTML
        html_files = glob.glob(os.path.join(self.results_dir, '*_clustering.html'))
        for file in html_files:
            model_name = file.split('_')[0].split('\\')[-1]
            completed_models.add(model_name)
            
        # بررسی فایل مقایسه
        comparison_file = os.path.join(self.results_dir, 'model_comparison.csv')
        if os.path.exists(comparison_file):
            df = pd.read_csv(comparison_file)
            completed_models.update(df['Model'].unique())
            
        return list(completed_models)
    
    def is_model_complete(self, model_name):
        """Check if a specific model has complete results"""
        # بررسی وجود همه فایل‌های مورد نیاز برای یک مدل
        required_files = [
            f"{model_name}_KMeans_clustering.html",
            f"{model_name}_DBSCAN_clustering.html",
            f"{model_name}_Hierarchical_clustering.html",
            f"{model_name}_GMM_clustering.html"
        ]
        
        for file in required_files:
            if not os.path.exists(os.path.join(self.results_dir, file)):
                return False
        return True
    
    def load_existing_results(self):
        """Load existing comparison results if available"""
        comparison_file = os.path.join(self.results_dir, 'model_comparison.csv')
        if os.path.exists(comparison_file):
            try:
                df = pd.read_csv(comparison_file)
                if not df.empty:
                    return df
            except Exception as e:
                print(f"Warning: Could not load existing results: {str(e)}")
        return None 