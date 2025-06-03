import os
from pathlib import Path

def setup_directories(dir_list):
    """Create directories if they don't exist"""
    for directory in dir_list:
        if isinstance(directory, str):
            directory = Path(directory)
        
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")

def get_file_extension(file_path):
    """Get the file extension from a path"""
    return os.path.splitext(file_path)[1].lower()

def is_image_file(file_path):
    """Check if file is an image based on extension"""
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    ext = get_file_extension(file_path)
    return ext in valid_extensions

def get_model_version_from_filename(filename):
    """Extract model version from filename
    
    Example: 'grain_model_v1.0.pt' -> 'v1.0'
    """
    try:
        # Extract part between last underscore and file extension
        name = os.path.basename(filename)
        base_name = os.path.splitext(name)[0]
        if '_v' in base_name:
            return base_name.split('_v', 1)[1]
        else:
            return "unknown"
    except:
        return "unknown" 