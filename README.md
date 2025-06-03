# Grain Size Analyzer

A desktop application for analyzing grain size in metallographic or microscopic images using computer vision and machine learning.

## Features

- **Image Analysis Mode**: Analyze individual images to detect grain boundaries and measure grain sizes
- **Training Mode**: Train or retrain models with new labeled data
- **Batch Processing**: Analyze multiple images at once
- **Export**: Export results to CSV or PDF reports

## Setup

1. Clone this repository:
   ```
   git clone [repository-url]
   cd grain-size-analyzer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

## Directory Structure

- `/components`: Core components of the application
- `/images`: Default location for storing input images
- `/labels`: Stores image annotations for training
- `/models`: Stores trained PyTorch models
- `/exports`: Destination for exported reports
- `/knowledge_base`: Knowledge base for model training

## Usage

### Analyzing Images

1. Click "Load Image" or "Load Folder" to import images
2. Select a model from the dropdown menu
3. Click "Analyze" to process the image
4. Review the results with the overlay visualization
5. Click "Export" to save the results as CSV or PDF

### Training a Model

1. In the "Training" tab, click "Add Labeled Data" to import labeled images
2. Click "Train Model" to retrain the model with the new data
3. Once training is complete, the new model will be available in the model dropdown

### Batch Processing

1. In the "Batch Processing" tab, load a folder of images
2. Configure batch processing settings
3. Click "Process Batch" to analyze all images
4. Results will be exported to the specified directory

## Development

The code is organized into modular components:

- `app.py`: Main application entry point
- `components/image_processor.py`: Image processing and visualization
- `components/model_handler.py`: PyTorch model management
- `components/utils.py`: Utility functions

## Customization

You can extend this application by:

1. Implementing more sophisticated models in `model_handler.py`
2. Adding new analysis features in `image_processor.py`
3. Enhancing the UI by modifying the main `app.py` file

## Requirements

- Python 3.8+
- PyTorch
- OpenCV
- PyQt5
- Other dependencies in requirements.txt 