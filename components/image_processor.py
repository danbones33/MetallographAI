import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

class ImageProcessor:
    def __init__(self):
        """Initialize the image processor with default parameters"""
        self.visualization_color = (0, 255, 0)  # Green color for overlays
        self.line_thickness = 2
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        
    def preprocess_image(self, image):
        """Preprocess image for model input
        
        Args:
            image: OpenCV image (BGR format)
            
        Returns:
            Preprocessed image suitable for model input
        """
        # Basic preprocessing - resize to standard dimensions
        resized = cv2.resize(image, (512, 512))
        
        # Convert to float and normalize to 0-1
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def create_overlay(self, image, results):
        """Create visualization overlay on the image
        
        Args:
            image: Original image (BGR format)
            results: Dictionary with analysis results
            
        Returns:
            Image with visualization overlays
        """
        overlay_img = image.copy()
        raw_mask = results.get('raw_mask')

        # Check if raw_mask is a multi-class segmentation map
        # A simple check: if it's 2D and its max value suggests class indices (e.g., < 10 or so)
        # and grain_boundaries list is empty (because we skip populating it for multi-class in model_handler)
        is_multiclass_map = (
            raw_mask is not None and 
            raw_mask.ndim == 2 and 
            (results.get('grain_count', 0) == 0 and not results.get('grain_boundaries'))
            # Add more robust check if needed, e.g., based on a flag from results
        )

        if is_multiclass_map:
            # Define a color map for the classes (BGR format)
            # Class 0 (Matrix): Dark Gray
            # Class 1 (Austenite): Blue
            # Class 2 (Martensite/Austenite): Green
            # Class 3 (Precipitate): Red
            # Class 4 (Defect): Yellow
            # Background/Other: Black (if any pixel value is outside 0-4)
            color_map = np.array([
                [64, 64, 64],    # 0: Matrix (Dark Gray)
                [255, 0, 0],     # 1: Austenite (Blue)
                [0, 255, 0],     # 2: Martensite/Austenite (Green)
                [0, 0, 255],     # 3: Precipitate (Red)
                [0, 255, 255],   # 4: Defect (Yellow)
            ], dtype=np.uint8)

            # Create an empty color image of the same size as raw_mask
            color_segmented_img = np.zeros((raw_mask.shape[0], raw_mask.shape[1], 3), dtype=np.uint8)
            
            for class_idx in range(len(color_map)):
                color_segmented_img[raw_mask == class_idx] = color_map[class_idx]
            
            # Blend the color segmented image with the original image
            # Ensure overlay_img is compatible (e.g. same size as color_segmented_img)
            # raw_mask from model_handler is already resized to original image dimensions.
            if overlay_img.shape[:2] != color_segmented_img.shape[:2]:
                # This case should ideally not happen if raw_mask is correctly resized in ModelHandler
                # If it does, resize color_segmented_img to match overlay_img for blending
                color_segmented_img = cv2.resize(color_segmented_img, 
                                                 (overlay_img.shape[1], overlay_img.shape[0]), 
                                                 interpolation=cv2.INTER_NEAREST)

            alpha = 0.6 # Transparency factor for the overlay
            cv2.addWeighted(color_segmented_img, alpha, overlay_img, 1 - alpha, 0, overlay_img)
            # No text summary for grain count/size for multi-class map by default

        else:
            # Fallback to existing binary contour visualization
            if 'grain_boundaries' in results and results['grain_boundaries']:
                boundaries = results['grain_boundaries']
                cv2.polylines(overlay_img, boundaries, isClosed=True, 
                            color=self.visualization_color, thickness=self.line_thickness)
            
            if 'grain_centers' in results and results['grain_centers']:
                centers = results['grain_centers']
                for center in centers:
                    cv2.circle(overlay_img, center, radius=3, 
                             color=self.visualization_color, thickness=-1)
            
            if 'avg_grain_size' in results and results.get('grain_count', 0) > 0:
                avg_size = results['avg_grain_size']
                grain_count = results.get('grain_count', 0)
                text_lines = [
                    f"Grains: {grain_count}",
                    f"Avg size: {avg_size:.2f} μm"
                ]
                y_pos = 30
                for line in text_lines:
                    cv2.putText(overlay_img, line, (10, y_pos), self.font, 
                              self.font_scale, self.visualization_color, 2)
                    y_pos += 30
        
        return overlay_img
    
    def export_to_csv(self, results, output_path):
        """Export analysis results to CSV
        
        Args:
            results: Dictionary with analysis results
            output_path: Path to save the CSV file
        """
        # Extract data for CSV
        data = {}
        
        # Convert any complex structures to serializable format
        if 'grain_sizes' in results:
            data['Grain_ID'] = list(range(1, len(results['grain_sizes']) + 1))
            data['Size_um'] = results['grain_sizes']
        
        # Add summary statistics
        if 'avg_grain_size' in results:
            data['Average_Size'] = [results['avg_grain_size']] * len(data.get('Grain_ID', [1]))
        
        if 'grain_count' in results:
            data['Total_Grains'] = [results['grain_count']] * len(data.get('Grain_ID', [1]))
        
        # Create and save DataFrame
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
    
    def export_to_pdf(self, results, image_path, output_path):
        """Export analysis results and visualization to PDF
        
        Args:
            results: Dictionary with analysis results
            image_path: Path to the original image
            output_path: Path to save the PDF file
        """
        # Read the original image
        original_img = cv2.imread(image_path)
        
        # Create overlay visualization
        overlay_img = self.create_overlay(original_img, results)
        
        # Convert images to RGB for matplotlib
        original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        overlay_rgb = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2RGB)
        
        # Create a PDF with matplotlib
        with PdfPages(output_path) as pdf:
            # First page: Title and summary
            plt.figure(figsize=(8.5, 11))
            plt.title("Grain Size Analysis Report", fontsize=16)
            
            # Add summary text
            summary_text = [
                f"Sample: {Path(image_path).stem}",
                f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                f"Total grains: {results.get('grain_count', 'N/A')}",
                f"Average grain size: {results.get('avg_grain_size', 'N/A'):.2f} μm",
                f"Min grain size: {min(results.get('grain_sizes', [0])):.2f} μm" if 'grain_sizes' in results else "",
                f"Max grain size: {max(results.get('grain_sizes', [0])):.2f} μm" if 'grain_sizes' in results else "",
            ]
            
            plt.text(0.1, 0.5, "\n".join(summary_text), fontsize=12)
            plt.axis('off')
            pdf.savefig()
            plt.close()
            
            # Second page: Original and processed images
            plt.figure(figsize=(8.5, 11))
            
            plt.subplot(2, 1, 1)
            plt.title("Original Image")
            plt.imshow(original_rgb)
            plt.axis('off')
            
            plt.subplot(2, 1, 2)
            plt.title("Analysis Overlay")
            plt.imshow(overlay_rgb)
            plt.axis('off')
            
            plt.tight_layout()
            pdf.savefig()
            plt.close()
            
            # Third page: Grain size distribution if available
            if 'grain_sizes' in results:
                plt.figure(figsize=(8.5, 11))
                plt.title("Grain Size Distribution")
                plt.hist(results['grain_sizes'], bins=20, alpha=0.7)
                plt.xlabel("Grain Size (μm)")
                plt.ylabel("Frequency")
                plt.grid(True, alpha=0.3)
                pdf.savefig()
                plt.close() 