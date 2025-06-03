import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from flask import Flask, render_template, Response, request, jsonify, send_file
from flask_cors import CORS
import sys
from pathlib import Path
import time
import cv2
import numpy as np
from PIL import Image
import torch
import json
import base64
import io
import tempfile
import requests
import zipfile

cv2.setNumThreads(1)
torch.set_num_threads(1)

# Import components
from components.image_processor import ImageProcessor
from components.model_handler import ModelHandler

app = Flask(__name__)
CORS(app)

# Configuration
app_root_dir = Path(__file__).resolve().parent
models_dir = app_root_dir / "models"
exports_dir = app_root_dir / "exports"
kb_dir = app_root_dir / "knowledge_base"

# Demo configuration
DEMO_IMAGE_DIR = Path(app.static_folder) / 'demo_images'
MODEL_DIR = Path(app.static_folder) / 'model'
MODEL_NAME = "grainboundary_model_ag_v1.pt"

# Remote demo images URL (GitHub releases or CDN)
DEMO_IMAGES_URL = "https://github.com/danbones33/MetallographAI/releases/download/v1.0/demo_images.zip"

# Fallback to models directory if static/model doesn't exist
if not MODEL_DIR.exists() or not (MODEL_DIR / MODEL_NAME).exists():
    MODEL_DIR = models_dir
    print(f"Using fallback model directory: {MODEL_DIR}")

# Ensure directories exist
DEMO_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Global instances
analysis_stats = {}
model_handler_instance = None
image_processor_instance = None

try:
    image_processor = ImageProcessor()
    model_handler = ModelHandler(model_dir=models_dir)
except Exception as e:
    print(f"Warning: Could not initialize some components: {e}")

def get_model_handler():
    global model_handler_instance
    if model_handler_instance is None:
        try:
            print(f"Initializing ModelHandler with model_dir: {str(MODEL_DIR)}")
            model_handler_instance = ModelHandler(model_dir=str(MODEL_DIR))
            model_path = MODEL_DIR / MODEL_NAME
            if not model_path.exists():
                print(f"Critical: Model file not found at {model_path}")
                return None 
            print(f"Loading model: {MODEL_NAME}")
            model_handler_instance.load_model(MODEL_NAME)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error initializing ModelHandler or loading model: {e}")
            model_handler_instance = None
            return None
    return model_handler_instance

def get_image_processor():
    global image_processor_instance
    if image_processor_instance is None:
        print("Initializing ImageProcessor.")
        image_processor_instance = ImageProcessor()
    return image_processor_instance

def pil_to_base64(pil_image, format="jpeg"):
    buffered = io.BytesIO()
    pil_image.save(buffered, format=format.upper())
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def cv2_to_base64(cv2_image, format=".jpg"):
    is_success, buffer = cv2.imencode(format, cv2_image)
    if not is_success:
        raise ValueError(f"Could not encode cv2 image to {format}")
    return base64.b64encode(buffer).decode('utf-8')

def download_demo_images():
    """Download and extract demo images if not available locally."""
    if DEMO_IMAGE_DIR.exists() and any(DEMO_IMAGE_DIR.iterdir()):
        return True  # Images already available
    
    try:
        print(f"Downloading demo images from {DEMO_IMAGES_URL}...")
        response = requests.get(DEMO_IMAGES_URL, timeout=30)
        if response.status_code == 200:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                tmp_file.write(response.content)
                tmp_file.flush()
                
                with zipfile.ZipFile(tmp_file.name, 'r') as zip_ref:
                    zip_ref.extractall(DEMO_IMAGE_DIR)
                
                os.unlink(tmp_file.name)
                print(f"Demo images downloaded and extracted to {DEMO_IMAGE_DIR}")
                return True
    except Exception as e:
        print(f"Failed to download demo images: {e}")
        return False

@app.route('/')
def index():
    """Serves the start screen."""
    return render_template('index.html')

@app.route('/live_demo_feed')
def live_demo_feed():
    def generate_updates():
        handler = get_model_handler()
        img_processor = get_image_processor()

        if handler is None:
            error_data = {"type": "error", "message": "Failed to load AI model."}
            yield f"data: {json.dumps(error_data)}\n\n"
            return

        if img_processor is None:
            error_data = {"type": "error", "message": "Failed to load Image Processor."}
            yield f"data: {json.dumps(error_data)}\n\n"
            return

        print("DEBUG: SSE - Attempting to process images...")
        image_files_paths = [f for f in DEMO_IMAGE_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']]
        images_to_process_paths = image_files_paths[:50]
        total_images = len(images_to_process_paths)
        print(f"DEBUG: SSE - Found {total_images} images to process.")

        if not images_to_process_paths:
            # Try to download demo images
            yield f"data: {json.dumps({'type': 'status', 'message': 'Downloading demo images...'})}\n\n"
            download_success = download_demo_images()
            if download_success:
                # Retry finding images after download
                image_files_paths = [f for f in DEMO_IMAGE_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']]
                images_to_process_paths = image_files_paths[:50]
                total_images = len(images_to_process_paths)
                print(f"DEBUG: SSE - After download, found {total_images} images to process.")
            else:
                yield f"data: {json.dumps({'type': 'status', 'message': 'Download failed, checking alternative sources...'})}\n\n"
            
            if not images_to_process_paths:
                error_data = {"type": "error", "message": "Live demo temporarily unavailable. Please try the /analyze endpoint to upload your own metallographic images for AI analysis!"}
                yield f"data: {json.dumps(error_data)}\n\n"
                return

        yield f"data: {json.dumps({'type': 'start', 'total_images': total_images})}\n\n"
        time.sleep(0.1)

        for i, image_path in enumerate(images_to_process_paths):
            try:
                print(f"DEBUG: SSE - Processing image {i+1}/{total_images}: {image_path.name}")
                pil_image_original = Image.open(image_path).convert('RGB')
                cv_image_bgr_original = cv2.cvtColor(np.array(pil_image_original), cv2.COLOR_RGB2BGR)

                current_result = handler.run_inference(
                    image_data=cv_image_bgr_original.copy(),
                    model_name=MODEL_NAME
                )

                overlay_cv_image = img_processor.create_overlay(cv_image_bgr_original.copy(), current_result)

                original_b64 = pil_to_base64(pil_image_original)
                overlay_b64 = cv2_to_base64(overlay_cv_image)

                image_stats = {
                    "filename": image_path.name,
                    "grain_count": current_result.get('grain_count', 'N/A'),
                    "avg_grain_size": current_result.get('avg_grain_size', 'N/A'),
                    "confidence": current_result.get('average_pixel_confidence', 'N/A')
                }
                if isinstance(image_stats["avg_grain_size"], (float, int)):
                     image_stats["avg_grain_size"] = f"{image_stats['avg_grain_size']:.2f}"
                if isinstance(image_stats["confidence"], (float, int)):
                     image_stats["confidence"] = f"{image_stats['confidence']:.3f}"

                data = {
                    "type": "update",
                    "current_image_index": i + 1,
                    "total_images": total_images,
                    "image_filename": image_path.name,
                    "original_image_base64": original_b64,
                    "overlay_image_base64": overlay_b64,
                    "stats": image_stats
                }
                yield f"data: {json.dumps(data)}\n\n"
                time.sleep(1)

            except Exception as e_img:
                print(f"Error processing image {image_path.name} for SSE: {e_img}")
                error_data = {
                    "type": "image_error",
                    "filename": image_path.name,
                    "message": str(e_img),
                    "current_image_index": i + 1,
                    "total_images": total_images
                }
                yield f"data: {json.dumps(error_data)}\n\n"
                time.sleep(1)

        yield f"data: {json.dumps({'type': 'end', 'message': 'Demo complete!'})}\n\n"

    return Response(generate_updates(), mimetype='text/event-stream')

@app.route('/start_demo', methods=['POST'])
def start_demo():
    """Triggers the image analysis demo."""
    global analysis_stats
    
    analysis_stats = {
        "total_images_processed": 0,
        "total_grains_found": 0,
        "average_grain_size_overall": 0.0,
        "total_processing_time_seconds": 0.0,
        "images_processed_details": [],
        "error": None
    }
    
    return jsonify({"status": "Demo started", "redirect": "/results"})

@app.route('/results')
def results():
    global analysis_stats
    return render_template('results.html', stats=analysis_stats)

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "port": os.environ.get('PORT', 'not set'),
        "python_version": os.environ.get('PYTHON_VERSION', 'not set')
    })

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    
    file = request.files['image']
    
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    try:
        result = model_handler.process_image(image)
        
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'result.png')
        cv2.imwrite(output_path, result)
        
        return send_file(output_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 