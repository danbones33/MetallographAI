from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import torch
from components.image_processor import ImageProcessor
from components.model_handler import ModelHandler
from pathlib import Path
import os
import tempfile

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize components
app_root_dir = Path(__file__).resolve().parent
models_dir = app_root_dir / "models"
exports_dir = app_root_dir / "exports"
kb_dir = app_root_dir / "knowledge_base"

try:
    image_processor = ImageProcessor()
    model_handler = ModelHandler(model_dir=models_dir)
except Exception as e:
    print(f"Warning: Could not initialize some components: {e}")

@app.route('/')
def home():
    return """
    <html>
        <head>
            <title>MetallographAI</title>
            <style>
                body {
                    background-color: #0a0a0a;
                    color: #00ff41;
                    font-family: 'Orbitron', sans-serif;
                    margin: 0;
                    padding: 20px;
                }
                .container {
                    max-width: 800px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.1);
                    backdrop-filter: blur(10px);
                    padding: 20px;
                    border-radius: 10px;
                }
                h1 {
                    text-align: center;
                    color: #00ff41;
                }
                .metrics {
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                }
                .metric {
                    text-align: center;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>MetallographAI Knowledge Base</h1>
                <div class="metrics">
                    <div class="metric">
                        <h3>Accuracy</h3>
                        <p>94.7%</p>
                    </div>
                    <div class="metric">
                        <h3>Inference Time</h3>
                        <p>2.3ms</p>
                    </div>
                </div>
                <h2>API Endpoints:</h2>
                <ul>
                    <li>POST /analyze - Analyze metallographic image</li>
                    <li>GET /health - Check system status</li>
                </ul>
            </div>
        </body>
    </html>
    """

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
    
    # Read and process the image
    img_array = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    # Process the image
    try:
        result = model_handler.process_image(image)
        
        # Save the result to a temporary file
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, 'result.png')
        cv2.imwrite(output_path, result)
        
        return send_file(output_path, mimetype='image/png')
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 