import os

def create_directory(directory):
    """Create directory if it doesn't exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)

def validate_image_directory(directory):
    """Validate that directory exists and contains images."""
    if not os.path.exists(directory):
        raise ValueError(f"Directory {directory} does not exist")
    
    valid_extensions = ('.png', '.jpg', '.jpeg')
    images = [f for f in os.listdir(directory) 
             if f.lower().endswith(valid_extensions)]
    
    if not images:
        raise ValueError(f"No valid images found in {directory}") 