import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path
from .utils import get_model_version_from_filename
from .data_loader import MicroscopyDataset
from torch.utils.data import DataLoader
from torch.optim import Adam
import logging
import traceback
from PIL import Image

logger = logging.getLogger("GrainAnalyzer.ModelHandler")

class SimpleGrainModel(nn.Module):
    """A simple PyTorch model for grain boundary segmentation."""
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleGrainModel, self).__init__()
        
        # Encoder (down-sampling path)
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Decoder (up-sampling path)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256) # 256 (from upconv) + 256 (from enc3 skip)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128) # 128 (from upconv) + 128 (from enc2 skip)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)   # 64 (from upconv) + 64 (from enc1 skip)

        # Output layer
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        # self.sigmoid = nn.Sigmoid() # To get probabilities for the mask -- REMOVED for multi-class compatibility

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)    # Becomes 64 channels
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)   # Becomes 128 channels
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)   # Becomes 256 channels
        p3 = self.pool(e3)
        
        # Bottleneck
        b = self.bottleneck(p3) # Becomes 512 channels
        
        # Decoder
        # For skip connections, ensure dimensions match if input size isn't power of 2.
        # For simplicity, assuming input dimensions are divisible by 8 (2^3 for 3 pool layers)
        d3 = self.upconv3(b)
        d3 = torch.cat([d3, e3], dim=1) # Concatenate along channel dimension
        d3 = self.dec3(d3)
        
        d2 = self.upconv2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.upconv1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        # Output
        out_mask = self.out_conv(d1)
        return out_mask # REMOVED self.sigmoid()

class ModelHandler:
    """Handles loading, saving, and inference for grain analysis models"""
    def __init__(self, model_dir, images_dir=None, labels_dir=None):
        self.model_dir = Path(model_dir)
        self.images_dir = Path(images_dir) if images_dir else None
        self.labels_dir = Path(labels_dir) if labels_dir else None
        self.current_model = None
        self.current_model_name = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_size = (128, 128) # Changed from (256, 256)
        self.transform = transforms.Compose([
            transforms.ToTensor(), # Converts image to [C, H, W] and scales to [0,1]
            transforms.Resize(self.input_size, antialias=True), # Resize to model's expected input
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]) # Normalization
        ])
        logger.info(f"ModelHandler initialized. Model directory: {self.model_dir}, Device: {self.device}, Input Size: {self.input_size}")
        
    def get_available_models(self):
        """Get list of available models in the model directory"""
        logger.info(f"Checking for models in: {self.model_dir}")
        
        if not self.model_dir.exists():
            logger.info(f"Model directory {self.model_dir} not found. Creating it and dummy model.")
            # _create_dummy_model now handles directory creation internally
            self._create_dummy_model()
            # After creating a dummy model, the directory should contain it.
            # We must ensure glob is called *after* potential dummy creation if dir didn't exist.
            model_files_list = list(self.model_dir.glob("*.pt"))
        else:
            logger.info(f"Model directory {self.model_dir} exists.")
            # Get the list of .pt files first
            model_files_list = list(self.model_dir.glob("*.pt"))
            
            if not model_files_list:
                logger.info(f"Model directory {self.model_dir} is empty of .pt files. Calling _create_dummy_model.")
                self._create_dummy_model()
                # After creating a dummy model, re-list the files
                model_files_list = list(self.model_dir.glob("*.pt"))
            
        logger.info(f"Found model files: {[f.name for f in model_files_list] if model_files_list else 'None'}")
        return [f.name for f in model_files_list] if model_files_list else []
    
    def _create_dummy_model(self):
        """Create a dummy model for initial use"""
        logger.info(f"Attempting to create dummy model in: {self.model_dir}")
        try:
            # Create a model instance that matches the new SimpleGrainModel structure
            dummy_model = SimpleGrainModel(in_channels=3, out_channels=1) 
            if not self.model_dir.exists():
                logger.info(f"Model directory {self.model_dir} does not exist for dummy. Creating it now.")
                os.makedirs(self.model_dir, exist_ok=True)

            save_path = self.model_dir / "grain_model_v1.0.pt"
            torch.save(dummy_model.state_dict(), save_path)
            logger.info(f"Successfully created dummy model at {save_path}")
        except Exception as e:
            logger.error(f"Failed to create dummy model: {str(e)}")
            logger.error(traceback.format_exc())
            raise # Re-raise the exception so the app knows something went wrong.
    
    def load_model(self, model_name):
        """Load a model from the model directory
        
        Args:
            model_name: Name of the model file
        """
        model_path = self.model_dir / model_name
        logger.info(f"Loading model: {model_path}")
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
        
        try:
            # Determine out_channels based on model name or by inspecting state_dict
            num_out_channels = 1 # Default to 1 for binary segmentation
            if "metaldam_segmentation" in model_name.lower(): # Check for MetalDAM model (multi-class)
                num_out_channels = 5 
            elif "grainboundary_model" in model_name.lower(): # Check for new grain boundary model (binary)
                num_out_channels = 1
            # Add more sophisticated checks or load metadata if available in future
            
            logger.info(f"Initializing SimpleGrainModel with out_channels={num_out_channels} for {model_name}")
            model = SimpleGrainModel(in_channels=3, out_channels=num_out_channels)
            
            # Load state dict
            # If we loaded state_dict above for inspection, we can reuse it here.
            # For now, loading it again.
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval() # Set model to evaluation mode
            
            self.current_model = model
            self.current_model_name = model_name
            self.current_model_channels = num_out_channels # Store the number of channels
            logger.info(f"Model {model_name} loaded successfully with {num_out_channels} output channels and moved to {self.device}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def run_inference(self, image_data, model_name): # Changed image_path_str back to image_data
        """Run inference on an image using the specified model
        
        Args:
            image_data: OpenCV image (BGR format NumPy array)
            model_name: Name of the model to use
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Attempting inference with model: {model_name}") # Removed image data from this log for brevity
        
        # Get original image dimensions from the NumPy array
        if not isinstance(image_data, np.ndarray):
            logger.error(f"Input image_data is not a NumPy array. Type: {type(image_data)}")
            # You might want to return None or raise an error here
            # For now, trying to proceed assuming it might be convertible, or let it fail at PIL conversion
            # return None 
            pass # Let it try to convert to PIL image later, might fail informatively

        original_h, original_w = image_data.shape[:2]

        # Convert BGR NumPy array to PIL Image
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"Failed to convert input image_data (shape: {image_data.shape if isinstance(image_data, np.ndarray) else 'N/A'}) to PIL Image: {e}")
            raise RuntimeError("Image data conversion to PIL Image failed.") from e

        # original_w, original_h = pil_image.size # This was from when image_path_str was used

        if self.current_model is None or self.current_model_name != model_name:
            logger.info(f"Current model is '{self.current_model_name}', requested '{model_name}'. Loading model.")
            try:
                self.load_model(model_name)
            except Exception as e:
                logger.error(f"Failed to load model {model_name} for inference: {str(e)}")
                raise RuntimeError(f"Could not load model {model_name} for inference. Check logs.") from e
        
        if self.current_model is None:
            logger.error("Inference called but no model is loaded.")
            raise RuntimeError("Cannot run inference: no model is currently loaded.")

        self.current_model.eval()
        
        try:
            # The transform now includes resizing to self.input_size
            img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
            logger.debug(f"Image preprocessed for inference, tensor shape: {img_tensor.shape}")
        except Exception as e:
            logger.error(f"Error during image preprocessing for inference: {str(e)}")
            logger.error(traceback.format_exc())
            raise RuntimeError("Image preprocessing failed for inference.") from e
        
        logger.debug(f"Running inference on device: {self.device}")
        results = {}
        with torch.no_grad():
            try:
                pred_mask_tensor = self.current_model(img_tensor) # Shape: [1, num_classes, H_model, W_model]
                
                average_pixel_confidence = 0.0
                # confidence_map_for_debug = None # This was commented out in user's file

                if self.current_model_channels == 1: # Binary segmentation
                    sigmoid_output = torch.sigmoid(pred_mask_tensor)
                    confidence_map = 1.0 - 2.0 * torch.abs(sigmoid_output - 0.5)
                    average_pixel_confidence = torch.mean(confidence_map).item()
                    # confidence_map_for_debug = confidence_map.squeeze().cpu().numpy()

                    threshold = 0.8 
                    logger.info(f"Using threshold: {threshold} for binary mask generation.")
                    binary_mask_128 = (sigmoid_output > threshold).squeeze().cpu().numpy().astype(np.uint8)
                    results['processed_mask_type'] = 'binary'
                    
                    segmentation_map_resized = cv2.resize(binary_mask_128, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                    results['raw_mask'] = segmentation_map_resized

                    try:
                        contours, _ = cv2.findContours(segmentation_map_resized.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        logger.info(f"Found {len(contours)} contours for binary analysis on resized mask.")
                        
                        grain_sizes = [cv2.contourArea(c) for c in contours]
                        results['grain_boundaries'] = contours 
                        results['grain_centers'] = [(int(cv2.moments(c)['m10'] / (cv2.moments(c)['m00'] + 1e-6)), 
                                                   int(cv2.moments(c)['m01'] / (cv2.moments(c)['m00'] + 1e-6))) for c in contours if cv2.moments(c)['m00'] > 0]
                        results['grain_sizes'] = grain_sizes
                        results['grain_count'] = len(contours)
                        results['avg_grain_size'] = np.mean(grain_sizes) if grain_sizes else 0.0
                        logger.info(f"Binary grain metrics calculated: Count={results.get('grain_count', 'N/A')}, AvgSize={results.get('avg_grain_size', 'N/A'):.2f}")
                    except Exception as grain_calc_e:
                        logger.error(f"CRITICAL: Error during binary grain metric calculation: {grain_calc_e}", exc_info=True)
                        results['grain_boundaries'] = []
                        results['grain_centers'] = []
                        results['grain_sizes'] = []
                        results['grain_count'] = -1 # Indicate error
                        results['avg_grain_size'] = -1.0 # Indicate error
                    
                    debug_mask_path = self.model_dir.parent / "debug_model_output_128x128_binary.png"
                    cv2.imwrite(str(debug_mask_path), binary_mask_128 * 255)
                    logger.info(f"Debug 128x128 model output binary mask saved to {debug_mask_path}")

                else: # Multi-class segmentation
                    softmax_output = torch.softmax(pred_mask_tensor, dim=1)
                    confidence_values, predicted_classes_tensor = torch.max(softmax_output, dim=1)
                    average_pixel_confidence = torch.mean(confidence_values).item()
                    # confidence_map_for_debug = confidence_values.squeeze().cpu().numpy()

                    predicted_classes_128 = predicted_classes_tensor.squeeze().cpu().numpy().astype(np.uint8)
                    results['processed_mask_type'] = 'multiclass'
                    
                    segmentation_map_resized = cv2.resize(predicted_classes_128, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
                    results['raw_mask'] = segmentation_map_resized # This is the class index map

                    # Initialize for phase percentage calculation
                    phase_percentages = {} 
                    total_pixels = original_h * original_w

                    # DEBUG: Save the resized segmentation map (class indices)
                    debug_seg_map_path = self.model_dir.parent / "debug_segmentation_map_resized_classes.png"
                    try:
                        unique_classes = np.unique(segmentation_map_resized)
                        logger.info(f"DEBUG: Unique classes in segmentation_map_resized: {unique_classes}")
                        # Scale for visibility: if 5 classes (0-4), map to 0, 63, 127, 191, 255
                        # Handle case where max class is 0 to avoid division by zero or no scaling
                        max_class_val = unique_classes.max()
                        scaling_factor = 255.0 / max_class_val if max_class_val > 0 else 0 
                        visible_seg_map = (segmentation_map_resized.astype(np.float32) * scaling_factor).clip(0,255).astype(np.uint8)
                        
                        if len(unique_classes) == 1 and unique_classes[0] == 0:
                             visible_seg_map.fill(50) # Dark gray for all class 0
                        elif len(unique_classes) == 1 and unique_classes[0] == 1:
                             visible_seg_map.fill(100) # Medium gray for all class 1
                        # Add more specific single-class visualizations if needed

                        cv2.imwrite(str(debug_seg_map_path), visible_seg_map)
                        logger.info(f"DEBUG: Resized class segmentation map saved to {debug_seg_map_path}")
                    except Exception as e_save_seg:
                        logger.error(f"DEBUG: Failed to save segmentation_map_resized: {e_save_seg}", exc_info=True)

                    # Define MetalDAM class names and colors (BGR)
                    # Order should match class indices 0 through N-1
                    metaldam_class_names = ["Matrix", "Austenite", "Bainite", "Martensite", "Carbide/Inclusion"]
                    # BGR Colors:
                    metaldam_colors_bgr = [
                        (180, 180, 180), # 0: Matrix (Light Gray)
                        (255, 100, 100), # 1: Austenite (Light Blue)
                        (100, 255, 100), # 2: Bainite (Light Green)
                        (100, 100, 255), # 3: Martensite (Light Red)
                        (100, 255, 255)  # 4: Carbide/Inclusion (Light Yellow)
                    ]
                    
                    # Create colorized phase map
                    phase_map_color = np.zeros_like(image_data) 
                    logger.info(f"DEBUG MH: phase_map_color initialized. Shape: {phase_map_color.shape}, dtype: {phase_map_color.dtype}, unique values present: {np.unique(phase_map_color.reshape(-1,3), axis=0)}")

                    # This loop calculates phase_percentages, ensure it's initialized above
                    for class_idx, color_bgr in enumerate(metaldam_colors_bgr):
                        if class_idx < len(metaldam_class_names): # Ensure we don't go out of bounds for names
                            class_name = metaldam_class_names[class_idx]
                            mask_for_class = (segmentation_map_resized == class_idx)
                            
                            if class_idx == 0: # Specifically log for Matrix (class 0)
                                if np.any(mask_for_class):
                                    logger.info(f"DEBUG MH: Class 0: Applying color {color_bgr}. Sum of mask: {np.sum(mask_for_class)}.")
                                    
                                    # Log the state of a few pixels in phase_map_color *before* assignment in the masked region
                                    example_coords = np.transpose(np.where(mask_for_class))
                                    log_limit = min(3, len(example_coords)) # Log up to 3 example pixels
                                    
                                    if log_limit > 0:
                                        logger.info(f"DEBUG MH: Class 0: Example pixels in phase_map_color BEFORE assignment to {color_bgr}:")
                                        for i in range(log_limit):
                                            coord = example_coords[i]
                                            logger.info(f"  - Pixel at ({coord[0]}, {coord[1]}): {phase_map_color[coord[0], coord[1]]}")

                                    phase_map_color[mask_for_class] = color_bgr # Assignment

                                    # Log the state of the same few pixels *after* assignment
                                    if log_limit > 0:
                                        logger.info(f"DEBUG MH: Class 0: Example pixels in phase_map_color AFTER assignment to {color_bgr}:")
                                        for i in range(log_limit):
                                            coord = example_coords[i]
                                            logger.info(f"  - Pixel at ({coord[0]}, {coord[1]}): {phase_map_color[coord[0], coord[1]]}")
                                    
                                    unique_values_in_mask = np.unique(phase_map_color[mask_for_class], axis=0)
                                    logger.info(f"DEBUG MH: Class 0: Unique values in phase_map_color[mask_for_class] (should ideally be just {color_bgr}): {unique_values_in_mask}")
                                else:
                                    logger.info(f"DEBUG MH: Class 0: No pixels for this class (mask_for_class is all False).")
                            
                            elif np.any(mask_for_class): # For other classes, just apply if mask is not empty
                                phase_map_color[mask_for_class] = color_bgr
                            
                            if np.any(mask_for_class):
                                count = np.sum(mask_for_class)
                            
                            pixel_count = np.sum(mask_for_class)
                            phase_percentages[class_name] = (pixel_count / total_pixels) * 100 if total_pixels > 0 else 0
                        else: # Should not happen if color list matches expected classes
                            logger.warning(f"Class index {class_idx} is out of bounds for metaldam_class_names.")

                    results['phase_map_color'] = phase_map_color.copy() # Force a deep copy
                    results['phase_percentages'] = phase_percentages # Dict of {name: percentage}
                    
                    # DEBUG: Save the generated color phase map to disk
                    debug_phase_map_path = self.model_dir.parent / "debug_generated_phase_map_color.png"
                    try:
                        logger.info(f"DEBUG MH: Just before saving debug_generated_phase_map_color.png. Unique values in entire phase_map_color: {np.unique(phase_map_color.reshape(-1,3), axis=0)}")
                        cv2.imwrite(str(debug_phase_map_path), phase_map_color)
                        logger.info(f"DEBUG: Color phase map saved to {debug_phase_map_path}")
                    except Exception as e_save:
                        logger.error(f"DEBUG: Failed to save color phase map: {e_save}")

                    # Existing placeholder grain metrics for multi-class
                    results['grain_count'] = 0 
                    results['avg_grain_size'] = 0.0
                    results['grain_boundaries'] = []
                    results['grain_centers'] = []
                    results['grain_sizes'] = []
                    logger.info(f"Multi-class segmentation: Color phase map generated. Percentages: {phase_percentages}")

                    # Debug save the 128x128 multi-class argmax mask (class 0)
                    debug_mask_path = self.model_dir.parent / "debug_model_output_128x128_multiclass_argmax.png"
                    # Need to visualize this properly, e.g. by mapping class indices to colors
                    # For now, just save the raw class indices if that's helpful, or a specific class's binary mask
                    # cv2.imwrite(str(debug_mask_path), predicted_classes_128) # This won't be visually clear
                    # Let's save class 0 mask for debugging like before
                    binary_mask_for_class_0_128 = (predicted_classes_128 == 0).astype(np.uint8)
                    cv2.imwrite(str(debug_mask_path), binary_mask_for_class_0_128 * 255) 
                    logger.info(f"Debug 128x128 model output (class 0) mask saved to {debug_mask_path}")

                results['average_pixel_confidence'] = average_pixel_confidence
                logger.info(f"Calculated average_pixel_confidence: {average_pixel_confidence:.4f}")
                
                try:
                    keys_list_str = str(list(results.keys()))
                    logger.info(f"Inference successful. Results keys: {keys_list_str}")
                except Exception as log_keys_e:
                    logger.error(f"Error trying to log results.keys(): {log_keys_e}. Current keys (attempt 2): {results.keys() if isinstance(results, dict) else 'Not a dict'}", exc_info=True)

            except Exception as e:
                logger.error(f"Error during model prediction or mask processing: {str(e)}")
                logger.error(traceback.format_exc())
                raise RuntimeError("Model prediction or mask processing failed.") from e
        
        logger.info(f"FINAL CHECK before return from run_inference. Results keys: {list(results.keys()) if isinstance(results, dict) else 'Results not a dict or None'}")
        return results

    def get_next_model_version_name(self, base_name="grain_model"):
        """Generates the next available model version name."""
        versions = [get_model_version_from_filename(f.name) for f in self.model_dir.glob(f"{base_name}_v*.pt")]
        major_versions = [int(v.split('.')[0]) for v in versions if v and '.' in v]
        
        if not major_versions:
            next_major = 1
        else:
            next_major = max(major_versions) + 1 # Increment major version for new training
            
        new_model_name = f"{base_name}_v{next_major}.0.pt"
        logger.info(f"Next model name determined: {new_model_name}")
        return new_model_name

    def save_model(self, model_state_dict, name=None):
        """Save the model state_dict. If no name is provided, generate one."""
        if not self.model_dir.exists():
            os.makedirs(self.model_dir, exist_ok=True)
            logger.info(f"Created model directory: {self.model_dir}")

        if name is None:
            name = self.get_next_model_version_name()
        
        save_path = self.model_dir / name
        try:
            torch.save(model_state_dict, save_path)
            logger.info(f"Model saved to {save_path}")
            return str(save_path)
        except Exception as e:
            logger.error(f"Error saving model {name} to {save_path}: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def train_model(self, epochs=10, batch_size=4, learning_rate=0.001,
                    train_images_dir=None, train_labels_dir=None, 
                    val_images_dir=None, val_labels_dir=None, 
                    model_save_name=None, progress_callback=None,
                    load_weights_from_path=None, num_classes=1, is_binary_segmentation=True):
        """
        Train the grain segmentation model.

        Args:
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and validation.
            learning_rate (float): Learning rate for the optimizer.
            train_images_dir (str, optional): Path to training images. Defaults to self.images_dir.
            train_labels_dir (str, optional): Path to training labels. Defaults to self.labels_dir.
            val_images_dir (str, optional): Path to validation images.
            val_labels_dir (str, optional): Path to validation labels.
            model_save_name (str, optional): Name to save the trained model. If None, a new version is generated.
            progress_callback (function, optional): Callback function to report progress (epoch, batch, loss, val_loss).
            load_weights_from_path (str, optional): Path to a .pt file to load initial weights from.
            num_classes (int): Number of output classes for the model.
            is_binary_segmentation (bool): Flag to indicate if this is binary segmentation (affects mask processing and loss).
        Returns:
            str: Path to the saved trained model.
        """
        logger.info(f"Starting model training: epochs={epochs}, batch_size={batch_size}, lr={learning_rate}")

        # Use class defaults if specific dirs are not provided
        train_img_path = Path(train_images_dir) if train_images_dir else self.images_dir
        train_lbl_path = Path(train_labels_dir) if train_labels_dir else self.labels_dir

        if not train_img_path or not train_lbl_path:
            logger.error("Training image or label directory not provided or set in ModelHandler.")
            raise ValueError("Training image and label directories must be specified.")
        if not train_img_path.exists() or not train_lbl_path.exists():
            logger.error(f"Training data path not found. Images: {train_img_path}, Labels: {train_lbl_path}")
            raise FileNotFoundError("Training data path not found.")

        # Define transformations for training data (can be more extensive than inference)
        train_transform = transforms.Compose([
            transforms.ToTensor(), # Converts image to [C, H, W] and scales to [0,1]
            transforms.Resize(self.input_size, antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # Add more augmentations as needed, e.g., random rotation
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        if is_binary_segmentation:
            # Mask transform for binary segmentation (0/255 input -> 0.0/1.0 output float tensor)
            mask_transform = transforms.Compose([
                transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor() # Converts PIL (L mode, 0-255) to FloatTensor [1,H,W] scaled to [0,1]
            ])
        else: # Multi-class segmentation (e.g., MetalDAM)
            mask_transform = transforms.Compose([
                transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.NEAREST), 
                transforms.PILToTensor(), 
                transforms.Lambda(lambda x: x.squeeze(0).to(torch.long))
            ])
        
        logger.info(f"Creating training dataset from: Images='{train_img_path}', Labels='{train_lbl_path}'")
        try:
            train_dataset = MicroscopyDataset(image_dir=str(train_img_path), 
                                            mask_dir=str(train_lbl_path), 
                                            image_transform=train_transform, 
                                            mask_transform=mask_transform)
            # Improved check for empty dataset
            if not train_dataset or len(train_dataset) == 0: 
                 logger.error("Training dataset is empty or failed to initialize (0 samples found).")
                 raise ValueError("Training dataset could not be created or is empty (0 samples found).")
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            logger.info(f"Training DataLoader created. Number of samples: {len(train_dataset)}, Number of batches: {len(train_loader)}")

            # Check if DataLoader is empty (e.g. batch_size > num_samples)
            if len(train_loader) == 0:
                logger.error(f"Training DataLoader is empty (0 batches). Number of samples: {len(train_dataset)}, batch_size: {batch_size}. This can happen if batch_size > number of samples.")
                raise ValueError("Training DataLoader is empty. Ensure batch_size is not greater than the number of samples.")

        except Exception as e:
            logger.error(f"Failed to create training dataset/loader: {e}")
            logger.error(traceback.format_exc())
            raise

        val_loader = None
        if val_images_dir and val_labels_dir:
            val_img_path = Path(val_images_dir)
            val_lbl_path = Path(val_labels_dir)
            if val_img_path.exists() and val_lbl_path.exists():
                 # Simpler transform for validation, typically no augmentation beyond resize and normalize
                val_image_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(self.input_size, antialias=True),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                if is_binary_segmentation:
                    val_mask_transform = transforms.Compose([
                        transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.NEAREST),
                        transforms.ToTensor()
                    ])
                else: # Multi-class
                    val_mask_transform = transforms.Compose([
                        transforms.Resize(self.input_size, interpolation=transforms.InterpolationMode.NEAREST),
                        transforms.PILToTensor(),
                        transforms.Lambda(lambda x: x.squeeze(0).to(torch.long))
                    ])
                try:
                    val_dataset = MicroscopyDataset(image_dir=str(val_img_path), 
                                                    mask_dir=str(val_lbl_path), 
                                                    image_transform=val_image_transform, 
                                                    mask_transform=val_mask_transform)
                    # Improved check for empty dataset
                    if not val_dataset or len(val_dataset) == 0: # Improved check
                        logger.warning("Validation dataset is empty or failed to initialize (0 samples found). Proceeding without validation.")
                        val_loader = None # Explicitly set to None
                    else:
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
                        logger.info(f"Validation DataLoader created. Number of samples: {len(val_dataset)}, Number of batches: {len(val_loader)}")
                        if len(val_loader) == 0: # Check if val_loader is accidentally empty
                            logger.warning(f"Validation DataLoader is empty (0 batches) despite dataset having {len(val_dataset)} samples. Batch size: {batch_size}. Validation will be skipped.")
                            val_loader = None # Skip validation if loader is empty
                except Exception as e:
                    logger.warning(f"Failed to create validation dataset/loader: {e}. Proceeding without validation.")
                    val_loader = None # Ensure val_loader is None if creation fails
            else:
                logger.warning("Validation image or label directory not found. Proceeding without validation.")

        # Initialize model, optimizer, and loss function
        model = SimpleGrainModel(in_channels=3, out_channels=num_classes).to(self.device)

        if load_weights_from_path:
            weights_path = Path(load_weights_from_path)
            if weights_path.exists() and weights_path.is_file():
                try:
                    model.load_state_dict(torch.load(weights_path, map_location=self.device))
                    logger.info(f"Successfully loaded initial weights from: {weights_path}")
                except Exception as e:
                    logger.error(f"Error loading weights from {weights_path}: {e}. Starting with fresh weights.", exc_info=True)
            else:
                logger.warning(f"Specified weights path {weights_path} not found or not a file. Starting with fresh weights.")

        optimizer = Adam(model.parameters(), lr=learning_rate)
        
        if is_binary_segmentation:
            criterion = nn.BCEWithLogitsLoss() # For binary segmentation
        else:
            criterion = nn.CrossEntropyLoss() # For multi-class segmentation

        logger.info(f"Model, optimizer, and loss function initialized for training on {self.device}. Num classes: {num_classes}, Criterion: {type(criterion).__name__}")

        for epoch in range(epochs):
            model.train() # Set model to training mode
            running_loss = 0.0
            for batch_idx, (images, masks) in enumerate(train_loader):
                images = images.to(self.device)
                masks = masks.to(self.device) # Expected shape [B, 1, H, W]

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images) # Model outputs raw logits if Sigmoid is part of loss

                # Calculate loss
                loss = criterion(outputs, masks)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                
                log_msg = f"Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                logger.debug(log_msg)
                if progress_callback:
                    progress_callback(epoch=epoch+1, total_epochs=epochs,
                                      batch=batch_idx+1, total_batches=len(train_loader),
                                      loss=loss.item(), val_loss=None) # val_loss will be updated later

            epoch_loss = running_loss / len(train_loader)
            logger.info(f"Epoch [{epoch+1}/{epochs}] completed. Average Training Loss: {epoch_loss:.4f}")

            # Validation phase
            val_epoch_loss = None
            if val_loader:
                model.eval() # Set model to evaluation mode
                running_val_loss = 0.0
                with torch.no_grad():
                    for val_images, val_masks in val_loader:
                        val_images = val_images.to(self.device)
                        val_masks = val_masks.to(self.device)
                        
                        val_outputs = model(val_images)
                        val_loss = criterion(val_outputs, val_masks)
                        running_val_loss += val_loss.item()
                
                val_epoch_loss = running_val_loss / len(val_loader)
                logger.info(f"Epoch [{epoch+1}/{epochs}] Validation Loss: {val_epoch_loss:.4f}")
                if progress_callback: # Update progress with validation loss
                     progress_callback(epoch=epoch+1, total_epochs=epochs,
                                      batch=len(train_loader), total_batches=len(train_loader), # Indicate end of epoch batches
                                      loss=epoch_loss, val_loss=val_epoch_loss)


        logger.info("Training finished.")
        
        # Save the trained model
        final_model_name = model_save_name if model_save_name else self.get_next_model_version_name()
        saved_model_path = self.save_model(model.state_dict(), name=final_model_name)
        
        logger.info(f"Trained model saved as {saved_model_path}")
        
        # Optionally, update current_model to the newly trained one
        # self.load_model(Path(saved_model_path).name) 
        # logger.info(f"Newly trained model '{Path(saved_model_path).name}' is now loaded as current model.")

        return saved_model_path

# Example Dice Loss (if needed, place it appropriately, e.g., in utils or here if specific)
# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1.):
#         super(DiceLoss, self).__init__()
#         self.smooth = smooth
# 
#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits) # Convert logits to probabilities
#         # Flatten label and prediction tensors
#         probs = probs.view(-1)
#         targets = targets.view(-1)
#         
#         intersection = (probs * targets).sum()                            
#         dice = (2.*intersection + self.smooth)/(probs.sum() + targets.sum() + self.smooth)  
#         return 1 - dice 