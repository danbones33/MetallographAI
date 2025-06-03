import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_DUPLICATE_LIB_OK']='True' # As per OMP error hint

from flask import Flask, render_template, Response
import sys
from pathlib import Path
import time # Added for timing
import cv2 # Added for image processing
import numpy as np # Added for image processing
from PIL import Image # Added for image processing
import torch # Added for torch.set_num_threads
import json # For SSE
import base64 # For image encoding
import io # For BytesIO

cv2.setNumThreads(1) # Force OpenCV to use a single thread to avoid OMP conflicts
torch.set_num_threads(1) # Force PyTorch to use a single thread

# Add project root to sys.path to allow importing components
# Assuming 'components' is in the parent directory of 'grain_analyzer_demo'
# If 'grain_analyzer_demo' is at the root, and 'components' is also at the root:
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Now try to import. This assumes 'components' is discoverable.
# If 'ModelHandler' or 'ImageProcessor' are directly in 'components'
from components.model_handler import ModelHandler #, SimpleGrainModel
from components.image_processor import ImageProcessor

app = Flask(__name__)

# --- Configuration ---
# Ensure the demo_images directory exists within static
DEMO_IMAGE_DIR = Path(app.static_folder) / 'demo_images'
MODEL_DIR = Path(app.static_folder) / 'model' # Model will be stored here
MODEL_NAME = "grainboundary_model_ag_v1.pt" # Or your preferred model

# Ensure directories exist
DEMO_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Placeholder for analysis results
analysis_stats = {}
# Global model handler to avoid reloading model on every request if app were more complex
# For this demo, initializing per call is fine, but for performance in general app, consider global.
model_handler_instance = None
image_processor_instance = None # Global instance for ImageProcessor

def get_model_handler():
    global model_handler_instance
    if model_handler_instance is None:
        try:
            print(f"Initializing ModelHandler with model_dir: {str(MODEL_DIR)}")
            model_handler_instance = ModelHandler(model_dir=str(MODEL_DIR))
            model_path = MODEL_DIR / MODEL_NAME
            if not model_path.exists():
                print(f"Critical: Model file not found at {model_path}")
                # In a real app, you might raise an error or return a specific status
                return None 
            print(f"Loading model: {MODEL_NAME}")
            model_handler_instance.load_model(MODEL_NAME)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error initializing ModelHandler or loading model: {e}")
            model_handler_instance = None # Reset on error
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
    # cv2_image is expected to be BGR
    is_success, buffer = cv2.imencode(format, cv2_image)
    if not is_success:
        raise ValueError(f"Could not encode cv2 image to {format}")
    return base64.b64encode(buffer).decode('utf-8')

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

        if img_processor is None: # Should not happen with current simple init
            error_data = {"type": "error", "message": "Failed to load Image Processor."}
            yield f"data: {json.dumps(error_data)}\n\n"
            return

        print("DEBUG: SSE - Attempting to process images...")
        image_files_paths = [f for f in DEMO_IMAGE_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']]
        images_to_process_paths = image_files_paths[:50] # Limit to 50
        total_images = len(images_to_process_paths)
        print(f"DEBUG: SSE - Found {total_images} images to process.")

        if not images_to_process_paths:
            error_data = {"type": "error", "message": "No demo images found."}
            yield f"data: {json.dumps(error_data)}\n\n"
            return

        yield f"data: {json.dumps({'type': 'start', 'total_images': total_images})}\n\n"
        time.sleep(0.1) # Give client time to set up

        for i, image_path in enumerate(images_to_process_paths):
            try:
                print(f"DEBUG: SSE - Processing image {i+1}/{total_images}: {image_path.name}")
                pil_image_original = Image.open(image_path).convert('RGB')
                cv_image_bgr_original = cv2.cvtColor(np.array(pil_image_original), cv2.COLOR_RGB2BGR)

                # Run inference
                current_result = handler.run_inference(
                    image_data=cv_image_bgr_original.copy(), # Pass a copy if run_inference modifies it
                    model_name=MODEL_NAME
                )

                # Create overlay using ImageProcessor
                # The create_overlay method in ImageProcessor expects results from ModelHandler
                # and the original image in BGR format
                overlay_cv_image = img_processor.create_overlay(cv_image_bgr_original.copy(), current_result)


                # Prepare data for SSE message
                original_b64 = pil_to_base64(pil_image_original)
                overlay_b64 = cv2_to_base64(overlay_cv_image) # create_overlay returns BGR

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
                time.sleep(1) # Pause to allow user to see update

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
                time.sleep(1) # Pause

        yield f"data: {json.dumps({'type': 'end', 'message': 'Demo complete!'})}\n\n"

    return Response(generate_updates(), mimetype='text/event-stream')

@app.route('/start_demo', methods=['POST'])
def start_demo():
    """Triggers the image analysis demo."""
    global analysis_stats

    # Initialize/reset stats with the full default structure at the beginning
    analysis_stats = {
        "total_images_processed": 0,
        "total_grains_found": 0,
        "average_grain_size_overall": 0.0,
        "total_processing_time_seconds": 0.0,
        "images_processed_details": [],
        "error": None  # Explicitly set error to None initially
    }
    
    handler = get_model_handler()
    if handler is None:
        # Set the error message in the already structured analysis_stats
        analysis_stats["error"] = "Failed to load the AI model. Please check server logs or ensure the model file is correctly placed in 'static/model/' and is not corrupted."
        return render_template('results.html', stats=analysis_stats)

    # If handler is fine, the rest of the code will populate the existing analysis_stats dictionary
    # The previous full reset of analysis_stats after the handler check is no longer needed here

    total_grains_accumulator = 0
    sum_of_per_image_avg_grain_sizes = 0.0
    images_contributing_to_avg_size = 0
    total_processing_time_accumulator = 0.0

    # --- Image processing starts only if model loaded successfully ---
    # The debug prints for image directory are moved inside this try block
    try:
        print("DEBUG: Attempting to process images...")
        print(f"DEBUG: DEMO_IMAGE_DIR absolute path: {DEMO_IMAGE_DIR.resolve()}")
        print(f"DEBUG: DEMO_IMAGE_DIR exists: {DEMO_IMAGE_DIR.exists()}")
        if DEMO_IMAGE_DIR.exists():
            print(f"DEBUG: Contents of DEMO_IMAGE_DIR: {list(DEMO_IMAGE_DIR.iterdir())}")
        else:
            print(f"DEBUG: DEMO_IMAGE_DIR does not exist at the path shown above.")

        image_files_paths = [f for f in DEMO_IMAGE_DIR.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg', '.bmp', '.tiff']]
        print(f"DEBUG: Found image files before filtering by extension: {[f.name for f in DEMO_IMAGE_DIR.iterdir() if f.is_file()]}")
        print(f"DEBUG: Found image files after filtering: {[f.name for f in image_files_paths]}")
        
        images_to_process_paths = image_files_paths[:50]
        print(f"DEBUG: Number of images to process: {len(images_to_process_paths)}")
        
        analysis_stats["total_images_processed"] = len(images_to_process_paths)

        if not images_to_process_paths:
            analysis_stats["error"] = "No demo images found in the demo_images directory. Please check the path and ensure images are present."
            print(f"DEBUG: Setting error due to no images - {analysis_stats['error']}") 
            # No return here; let it fall through to render_template at the end of the function.
            # This ensures any previous model error isn't accidentally cleared before rendering.
        else: # Only proceed to process images if images_to_process_paths is not empty
            for image_path in images_to_process_paths:
                start_time_single_image = time.time()
                try:
                    pil_image = Image.open(image_path).convert('RGB')
                    # Convert PIL (RGB) to OpenCV (BGR) format for analyze_image_data
                    cv_image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                    # Corrected method call to run_inference
                    current_result = handler.run_inference(
                        image_data=cv_image_bgr, 
                        model_name=MODEL_NAME # Pass the configured model name
                    )
                    
                    if current_result:
                        # Store details if needed (optional)
                        # analysis_stats["images_processed_details"].append({
                        #     "filename": image_path.name,
                        #     "grain_count": current_result.get('grain_count', 0),
                        #     "avg_grain_size": current_result.get('avg_grain_size', 0.0)
                        # })

                        if current_result.get('processed_mask_type') == 'binary':
                            grains_in_image = current_result.get('grain_count', 0)
                            avg_size_in_image = current_result.get('avg_grain_size', 0.0)
                            
                            total_grains_accumulator += grains_in_image
                            if grains_in_image > 0 and avg_size_in_image > 0: # Only include valid sizes
                                sum_of_per_image_avg_grain_sizes += avg_size_in_image
                                images_contributing_to_avg_size += 1
                        # else: multi-class, no grain count / avg size in the same way
                    
                except Exception as e_img:
                    print(f"Error processing image {image_path.name}: {e_img}")
                    # Optionally log per-image errors to display to user
                    analysis_stats["images_processed_details"].append({
                        "filename": image_path.name, "error": str(e_img)
                    })

                end_time_single_image = time.time()
                total_processing_time_accumulator += (end_time_single_image - start_time_single_image)

            analysis_stats["total_grains_found"] = total_grains_accumulator
            if images_contributing_to_avg_size > 0:
                analysis_stats["average_grain_size_overall"] = sum_of_per_image_avg_grain_sizes / images_contributing_to_avg_size
            else:
                analysis_stats["average_grain_size_overall"] = 0.0
            analysis_stats["total_processing_time_seconds"] = total_processing_time_accumulator

    except Exception as e_outer:
        print(f"An error occurred during the demo run (image processing stage): {e_outer}")
        analysis_stats["error"] = f"An unexpected error occurred during image processing: {e_outer}"
    # The final render_template is outside the try/except for image processing
    return render_template('results.html', stats=analysis_stats)

@app.route('/results')
def results():
    """Displays the results screen."""
    global analysis_stats
    return render_template('results.html', stats=analysis_stats)

if __name__ == '__main__':
    # Make sure to place your model file (e.g., grainboundary_model_ag_v1.pt) 
    # into the 'grain_analyzer_demo/static/model/' directory.
    # And your 50 demo images into 'grain_analyzer_demo/static/demo_images/'
    print(f"Starting Flask app for Live Demo...")
    print(f"Static folder is: {app.static_folder}")
    print(f"Expected model location: {MODEL_DIR / MODEL_NAME}")
    print(f"Demo images directory: {DEMO_IMAGE_DIR}")
    # Ensure model is loaded on startup for faster first request, or handle loading on first /start_demo call
    # For simplicity, get_model_handler() will be called on first /start_demo
    app.run(debug=True, host='0.0.0.0', port=5000) # Changed to allow access from network 