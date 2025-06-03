import sys
import os
from pathlib import Path
import csv # Added for process_loaded_and_report

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                           QListWidget, QStatusBar, QComboBox,
                           QMessageBox, QLineEdit, QFormLayout, QGroupBox, QListWidgetItem,
                           QSizePolicy)
from PyQt5.QtGui import QPixmap, QImage, QFont, QIcon
from PyQt5.QtCore import Qt

import cv2
import numpy as np
import torch

from components.image_processor import ImageProcessor
from components.model_handler import ModelHandler
from components.utils import setup_directories
import logging

logger = logging.getLogger("GrainAnalyzerApp")

DARK_THEME_QSS = """
QWidget {
    background-color: #2b2b2b;
    color: #f0f0f0;
    font-family: 'Segoe UI', Arial, sans-serif; /* Modern font */
    font-size: 10pt;
}
QMainWindow {
    background-color: #2b2b2b;
}
QStatusBar {
    background-color: #222222;
    color: #f0f0f0;
}
QLabel {
    color: #f0f0f0;
}
QPushButton {
    background-color: #4a4a4a;
    color: #f0f0f0;
    border: 1px solid #5a5a5a;
    padding: 6px 12px;
    border-radius: 4px;
    min-height: 20px; /* Ensure buttons are not too small */
}
QPushButton:hover {
    background-color: #5a5a5a;
    border: 1px solid #6a6a6a;
}
QPushButton:pressed {
    background-color: #3a3a3a;
}
QPushButton:disabled {
    background-color: #383838;
    color: #787878;
}
QLineEdit {
    background-color: #3c3c3c;
    color: #f0f0f0;
    border: 1px solid #5a5a5a;
    padding: 4px;
    border-radius: 3px;
}
QComboBox {
    background-color: #3c3c3c;
    color: #f0f0f0;
    border: 1px solid #5a5a5a;
    padding: 4px;
    border-radius: 3px;
    min-height: 20px; 
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border-left-width: 1px;
    border-left-color: #5a5a5a;
    border-left-style: solid;
    border-top-right-radius: 3px;
    border-bottom-right-radius: 3px;
    background-color: #4a4a4a;
}
QComboBox::down-arrow {
    image: url(noexist.png); /* Fallback if no icon, or provide one */
    /* For a more robust solution, consider QProxyStyle to draw the arrow */
}
QComboBox QAbstractItemView { /* Dropdown list style */
    background-color: #3c3c3c;
    color: #f0f0f0;
    border: 1px solid #5a5a5a;
    selection-background-color: #5a5a5a;
}
QListWidget {
    background-color: #3c3c3c;
    color: #f0f0f0;
    border: 1px solid #5a5a5a;
    padding: 2px;
    border-radius: 3px;
}
QListWidget::item {
    padding: 3px;
}
QListWidget::item:selected {
    background-color: #5a5a5a; /* Highlight selected item */
    color: #ffffff;
}
QGroupBox {
    background-color: #383838; /* Slightly different for contrast */
    color: #e0e0e0; /* Lighter title color */
    border: 1px solid #5a5a5a;
    border-radius: 5px;
    margin-top: 1ex; /* Space for title */
    font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left; /* Position of the title */
    padding: 0 5px;
    left: 10px; /* Indent title slightly */
    color: #e0e0e0;
}
/* Style for the image label specifically, maintaining a neutral background */
QLabel#imageDisplayLabel {
    background-color: #404040; /* Darker than main bg for image area */
    border: 1px solid #5a5a5a;
}
"""

CONFIDENCE_THRESHOLD = 0.30 # Adjusted from 0.75

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Grain Size Analyzer - Inference")
        self.setGeometry(100, 100, 1100, 750)
        
        self.setup_paths()
        
        self.image_processor = ImageProcessor()
        self.model_handler = ModelHandler(
            model_dir=self.models_dir,
        )
        
        # Initialize state variables that might be accessed by init_ui or its calls
        self.current_image_path = None
        self.current_result = None
        
        self.init_ui()
        
        self.refresh_models_and_update_dropdown()
        
        self.statusBar().showMessage("Ready")
        self.setStyleSheet(DARK_THEME_QSS)

    def setup_paths(self):
        app_root_dir = Path(__file__).resolve().parent
        self.models_dir = app_root_dir / "models"
        self.exports_dir = app_root_dir / "exports"
        self.kb_dir = app_root_dir / "knowledge_base"

        setup_directories([
            self.models_dir, self.exports_dir, self.kb_dir
        ])

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QHBoxLayout(central_widget)
        
        # === Left panel (file list) ===
        left_panel_group = QGroupBox("File Explorer")
        left_layout = QVBoxLayout(left_panel_group)
        
        self.file_list = QListWidget()
        self.file_list.itemClicked.connect(self.load_selected_image)
        
        self.load_btn = QPushButton("Load Image")
        self.load_btn.clicked.connect(self.load_image)
        
        self.load_folder_btn = QPushButton("Load Folder")
        self.load_folder_btn.clicked.connect(self.load_folder)
        
        left_layout.addWidget(self.file_list)
        controls_layout_left = QHBoxLayout()
        controls_layout_left.addWidget(self.load_btn)
        controls_layout_left.addWidget(self.load_folder_btn)
        left_layout.addLayout(controls_layout_left)
        left_panel_group.setFixedWidth(300)
        
        # === Center panel (image view) ===
        self.image_label = QLabel("No image loaded")
        self.image_label.setObjectName("imageDisplayLabel")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        
        # === Right panel (controls) ===
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Model selection GroupBox
        model_group_box = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group_box)
        self.model_dropdown = QComboBox()
        model_layout.addWidget(self.model_dropdown)
        
        # Actions GroupBox
        actions_group_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group_box)
        self.analyze_btn = QPushButton("Analyze Single Image")
        self.analyze_btn.clicked.connect(self.analyze_image)

        self.process_folder_btn = QPushButton("Process Loaded & Report")
        self.process_folder_btn.clicked.connect(self.process_loaded_and_report)
        self.process_folder_btn.setEnabled(False)

        self.export_btn = QPushButton("Export Single Result")
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)

        actions_layout.addWidget(self.analyze_btn)
        actions_layout.addWidget(self.process_folder_btn)
        actions_layout.addWidget(self.export_btn)
        
        right_layout.addWidget(model_group_box)
        right_layout.addWidget(actions_group_box)
        right_layout.addStretch()

        right_panel.setFixedWidth(280)
        
        # Add panels to main layout
        main_layout.addWidget(left_panel_group)
        main_layout.addWidget(self.image_label, 1)
        main_layout.addWidget(right_panel)
        
        self.setCentralWidget(central_widget)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
        self.update_action_buttons_state()

    def select_directory(self, line_edit_widget, dialog_title):
        pass

    def refresh_models_and_update_dropdown(self):
        logger.info("Refreshing model list...")
        self.update_model_dropdown()

    def update_model_dropdown(self):
        self.model_dropdown.clear()
        current_model_selection = None

        try:
            models = self.model_handler.get_available_models()
            if not models:
                self.model_dropdown.addItem("No models found")
                self.model_dropdown.setEnabled(False)
                logger.warning("No models found in the directory.")
            else:
                self.model_dropdown.addItems(models)
                self.model_dropdown.setEnabled(True)
                logger.info(f"Models loaded into dropdown: {models}")
                if models:
                    self.model_dropdown.setCurrentIndex(0)
        except Exception as e:
            logger.error(f"Failed to update model dropdown: {e}", exc_info=True)
            self.model_dropdown.addItem("Error loading models")
            self.model_dropdown.setEnabled(False)
            QMessageBox.critical(self, "Model Load Error", f"Could not load models: {e}")
        self.update_action_buttons_state()

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", str(self.kb_dir), 
                                                   "Image Files (*.png *.jpg *.bmp *.tif *.tiff)")
        if file_path:
            self.display_image(file_path)
            # If a single image is loaded via the button, clear the list and add this image.
            self.file_list.clear()
            
            file_name = os.path.basename(file_path)
            list_item = QListWidgetItem(file_name)  # Correctly create QListWidgetItem
            list_item.setData(Qt.UserRole, file_path) # Store full path
            self.file_list.addItem(list_item)
            self.update_action_buttons_state()
                
    
    def load_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "Select Folder", str(self.kb_dir))
        if folder_path:
            self.file_list.clear()
            logger.info(f"Loading images from folder: {folder_path}")
            valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
            image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]
            if not image_files:
                self.statusBar().showMessage(f"No images found in {folder_path}")
                logger.info(f"No image files found in selected folder: {folder_path}")
                return
            for file_name in image_files:
                # list_item = QListWidget.item(file_name) # Old incorrect line
                list_item = QListWidgetItem(file_name) # Correctly create QListWidgetItem
                list_item.setData(Qt.UserRole, os.path.join(folder_path, file_name))
                self.file_list.addItem(list_item)
            self.statusBar().showMessage(f"Loaded {len(image_files)} images from {folder_path}")
            self.update_action_buttons_state()

    def load_selected_image(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path:
            self.display_image(file_path)

    def display_image(self, file_path):
        self.current_image_path = file_path
        try:
            img_cv = cv2.imread(file_path)
            if img_cv is None:
                self.statusBar().showMessage(f"Error: Could not load image {file_path}")
                self.image_label.setText("Error loading image")
                QMessageBox.warning(self, "Image Load Error", f"Failed to load image: {file_path}")
                return

            height, width, channel = img_cv.shape
            bytes_per_line = 3 * width
            q_img = QImage(img_cv.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            pixmap = QPixmap.fromImage(q_img)
            
            label_w = self.image_label.width()
            label_h = self.image_label.height()
            
            if label_w < 10 or label_h < 10:
                label_w = 600
                label_h = 500
                
            scaled_pixmap = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
            self.statusBar().showMessage(f"Loaded image: {file_path}")
            self.current_result = None
            self.export_btn.setEnabled(False)
            logger.info(f"Displayed image: {file_path}")
        except Exception as e:
            logger.error(f"Error displaying image {file_path}: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error displaying image: {e}")
            self.image_label.setText(f"Could not display image.\nError: {e}")
            QMessageBox.critical(self, "Display Error", f"An error occurred while displaying the image: {e}")
            self.export_btn.setEnabled(False)
            logger.info(f"Analysis complete for {self.current_image_path} with {self.model_dropdown.currentText()}.")

    def analyze_image(self):
        if not self.current_image_path:
            self.statusBar().showMessage("No image loaded to analyze.")
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return

        selected_model = self.model_dropdown.currentText()
        if not selected_model or selected_model == "No models found" or selected_model == "Error loading models":
            self.statusBar().showMessage("No model selected or available.")
            QMessageBox.warning(self, "No Model", "Please select a valid model.")
            return

        self.statusBar().showMessage(f"Analyzing image with {selected_model}...")
        QApplication.processEvents()

        try:
            img_to_process_cv = cv2.imread(self.current_image_path)
            if img_to_process_cv is None:
                raise ValueError(f"Failed to reload image for analysis: {self.current_image_path}")

            self.current_result = None # Explicitly clear before getting new result
            QApplication.processEvents() # Allow UI to process this if it had any pending effect
            self.current_result = self.model_handler.run_inference(img_to_process_cv, selected_model)
            
            if self.current_result and ('raw_mask' in self.current_result or 'phase_map_color' in self.current_result):
                
                processed_image_to_display = None
                grain_count = self.current_result.get('grain_count', 'N/A') # Default to N/A if not found
                avg_grain_size = self.current_result.get('avg_grain_size', 0.0)
                avg_confidence = self.current_result.get('average_pixel_confidence', 0.0)
                phase_percentages = self.current_result.get('phase_percentages')

                if 'phase_map_color' in self.current_result:
                    # Multi-class model: display the pre-rendered color phase map
                    processed_image_to_display = self.current_result['phase_map_color']
                    status_msg = f"Phase analysis complete. Confidence: {avg_confidence:.3f}"
                    if phase_percentages:
                        percentages_str = ", ".join([f"{name}: {perc:.1f}%" for name, perc in phase_percentages.items()])
                        status_msg += f" | Phases: {percentages_str}"
                    
                    # --- BEGIN DEBUG --- 
                    try:
                        debug_app_phase_map_path = Path(__file__).resolve().parent / "debug_app_direct_from_current_result.png"
                        # Save directly from self.current_result to be absolutely sure
                        if self.current_result and 'phase_map_color' in self.current_result and isinstance(self.current_result['phase_map_color'], np.ndarray):
                            cv2.imwrite(str(debug_app_phase_map_path), self.current_result['phase_map_color'])
                            logger.info(f"DEBUG APP: Phase map directly from self.current_result saved to {debug_app_phase_map_path}")
                        else:
                            logger.error("DEBUG APP: self.current_result['phase_map_color'] not available or not a numpy array for direct save.")
                        if processed_image_to_display is None: # This check remains for the variable actually used for display
                             logger.error("DEBUG APP: processed_image_to_display is NONE before calling display_processed_image for multi-class!")
                    except Exception as e_debug_save:
                        logger.error(f"DEBUG APP: Failed to save debug_app_direct_from_current_result.png: {e_debug_save}")
                    # --- END DEBUG ---
                elif 'raw_mask' in self.current_result and self.current_result.get('processed_mask_type') == 'binary':
                    # Binary model: create overlay with grain boundaries
                    overlay_image = img_to_process_cv.copy()
                    if self.current_result.get('grain_boundaries'):
                        cv2.drawContours(overlay_image, self.current_result['grain_boundaries'], -1, (0, 255, 0), 1)
                    processed_image_to_display = overlay_image
                    status_msg = f"Grain analysis complete. Grains: {grain_count}, Avg size: {avg_grain_size:.2f} px². Confidence: {avg_confidence:.3f}"
                else:
                    # Fallback or unexpected result content
                    logger.warning("Result dictionary is missing expected keys for display (phase_map_color or raw_mask for binary).")
                    self.statusBar().showMessage("Analysis complete, but no standard visualization available.")
                    self.export_btn.setEnabled(False)
                    return # Exit if nothing standard to show

                if processed_image_to_display is not None:
                    self.display_processed_image(processed_image_to_display)
                
                # Review recommendation logic (common for both binary and multi-class based on confidence)
                needs_review_confidence = avg_confidence < CONFIDENCE_THRESHOLD
                review_text = ""
                if needs_review_confidence:
                    review_text = " (Review Recommended - Low Confidence)"
                    QMessageBox.information(self, "Review Recommended", 
                                            f"The average processing confidence for this image was {avg_confidence:.3f}, which is below the threshold of {CONFIDENCE_THRESHOLD}. Manual review is recommended.")
                
                # For binary, also specifically mention if zero grains found, even if confidence is high
                if self.current_result.get('processed_mask_type') == 'binary' and grain_count == 0:
                    if not review_text: # if not already flagged for low confidence
                        review_text = " (Review Recommended - Zero Grains)"
                    else: # if already flagged, append
                        review_text += " & Zero Grains"
                    
                    QMessageBox.information(self, "Review Recommended - Zero Grains",
                                            f"The analysis for {Path(self.current_image_path).name} "
                                            "resulted in 0 grains detected. Manual review is recommended.")

                self.statusBar().showMessage(status_msg + review_text)
                self.export_btn.setEnabled(True)
                logger.info(f"Analysis complete for {self.current_image_path} with {selected_model}. Confidence: {avg_confidence:.3f}, Review flags: {review_text if review_text else 'None'}")
            else:
                self.statusBar().showMessage("Analysis failed or produced no mask/phase map.")
                QMessageBox.warning(self, "Analysis Failed", "The analysis did not produce a valid result or mask.")
                self.export_btn.setEnabled(False)

        except Exception as e:
            logger.error(f"Error during analysis: {e}", exc_info=True)
            self.statusBar().showMessage(f"Analysis error: {e}")
            QMessageBox.critical(self, "Analysis Error", f"An error occurred during analysis: {e}")
            self.export_btn.setEnabled(False)
            logger.info(f"Analysis complete for {self.current_image_path} with {self.model_dropdown.currentText()}.")

    def display_processed_image(self, img):
        try:
            if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
                logger.error(f"display_processed_image received an invalid image type or dtype. Type: {type(img)}, Dtype: {img.dtype if hasattr(img, 'dtype') else 'N/A'}")
                self.image_label.setText("Error: Invalid image format for display.")
                return

            # Clear previous image/text to avoid showing stale content if conversion fails
            self.image_label.setText("Processing display...") 
            QApplication.processEvents() # Ensure text update is shown

            img_contiguous = np.ascontiguousarray(img) 
            height, width, channel = img_contiguous.shape
            
            if channel != 3:
                logger.error(f"display_processed_image received image with {channel} channels, expected 3.")
                self.image_label.setText("Error: Image not 3-channel for display.")
                return

            bytes_per_line = 3 * width
            # Revert to RGB888 and rgbSwapped() as it's generally more common / tested
            q_img = QImage(img_contiguous.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            if q_img.isNull():
                logger.error("display_processed_image: QImage conversion resulted in a null image.")
                self.image_label.setText("Error: Failed to convert image for display.")
                return

            pixmap = QPixmap.fromImage(q_img)
            
            if pixmap.isNull():
                logger.error("display_processed_image: QPixmap conversion resulted in a null pixmap.")
                self.image_label.setText("Error: Failed to create pixmap for display.")
                return

            label_w = self.image_label.width()
            label_h = self.image_label.height()
            
            if label_w < 10 or label_h < 10:
                label_w = 600
                label_h = 500
                
            scaled_pixmap = pixmap.scaled(label_w, label_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image_label.setPixmap(scaled_pixmap)
        except Exception as e:
            logger.error(f"Error displaying processed image: {e}", exc_info=True)
            self.statusBar().showMessage(f"Error displaying processed image: {e}")
            QMessageBox.critical(self, "Display Error", f"An error occurred displaying the processed image: {e}")

    def export_results(self):
        if not self.current_result or self.current_image_path is None:
            QMessageBox.warning(self, "No Results", "No results to export. Please analyze an image first.")
            return

        base_name = Path(self.current_image_path).stem
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Results", 
                                                   str(self.exports_dir / f"{base_name}_analysis.png"), 
                                                   "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options)
        
        if file_path:
            try:
                pixmap_to_save = self.image_label.pixmap()
                if pixmap_to_save and not pixmap_to_save.isNull():
                    pixmap_to_save.save(file_path)
                    self.statusBar().showMessage(f"Results exported to {file_path}")
                    logger.info(f"Results (overlay image) exported to {file_path}")
                    
                    raw_mask_path = Path(file_path).parent / f"{Path(file_path).stem}_mask.png"
                    if self.current_result.get('raw_mask') is not None:
                        mask_to_save = self.current_result['raw_mask']
                        if mask_to_save.ndim == 2:
                            if mask_to_save.dtype != np.uint8:
                                mask_to_save = (mask_to_save * 255).astype(np.uint8) if mask_to_save.max() <=1 else mask_to_save.astype(np.uint8)
                            cv2.imwrite(str(raw_mask_path), mask_to_save)
                            logger.info(f"Raw mask saved to {raw_mask_path}")
                        elif mask_to_save.ndim == 3:
                            logger.warning("Raw mask has 3 dimensions, direct saving with cv2.imwrite might not be intended for grayscale.")

                else:
                    QMessageBox.warning(self, "Export Error", "No image to save.")
            except Exception as e:
                logger.error(f"Error exporting results: {e}", exc_info=True)
                QMessageBox.critical(self, "Export Error", f"Failed to export results: {e}")

    def update_action_buttons_state(self):
        """Enable/disable action buttons based on current state."""
        has_image_loaded = self.current_image_path is not None
        has_model_selected = self.model_dropdown.currentIndex() >= 0 and \
                             self.model_dropdown.currentText() not in ["No models found", "Error loading models"]
        
        # Analyze Single Image button state
        self.analyze_btn.setEnabled(has_image_loaded and has_model_selected)
        
        # Export Single Result button state (already handled by analyze_image and display_image)
        # self.export_btn.setEnabled(self.current_result is not None) 
        
        # Process Loaded & Report button state
        can_batch_process = self.file_list.count() > 1 and has_model_selected
        self.process_folder_btn.setEnabled(can_batch_process)

    def process_loaded_and_report(self):
        """Processes all images in the file_list and generates a consolidated report."""
        num_images = self.file_list.count()
        if num_images <= 1:
            QMessageBox.information(self, "Not Enough Images", 
                                    "Please load multiple images (e.g., using 'Load Folder') to use batch processing.")
            return

        selected_model = self.model_dropdown.currentText()
        if not selected_model or selected_model in ["No models found", "Error loading models"]:
            QMessageBox.warning(self, "No Model", "Please select a valid model for batch processing.")
            return

        self.statusBar().showMessage(f"Starting batch processing of {num_images} images with {selected_model}...")
        QApplication.processEvents()

        all_results_summary = []
        processed_count = 0
        error_count = 0

        for i in range(num_images):
            item = self.file_list.item(i)
            image_path = item.data(Qt.UserRole)
            image_filename = os.path.basename(image_path)
            
            self.statusBar().showMessage(f"Batch: Processing {i+1}/{num_images}: {image_filename}...")
            QApplication.processEvents()
            
            current_image_result_summary = {
                'image_filename': image_filename,
                'grain_count': 'N/A',
                'average_grain_size_px2': 'N/A',
                'average_pixel_confidence': 'N/A',
                'phase_percentages': 'N/A', # New field for CSV
                'needs_review': 'N/A'
            }

            try:
                img_to_process_cv = cv2.imread(image_path)
                if img_to_process_cv is None:
                    logger.warning(f"Batch: Failed to load image {image_path}. Skipping.")
                    current_image_result_summary['grain_count'] = 'Error - Could not load'
                    current_image_result_summary['needs_review'] = True # Flag unreadable images
                    error_count += 1
                    all_results_summary.append(current_image_result_summary)
                    continue

                result = self.model_handler.run_inference(img_to_process_cv, selected_model)
                
                avg_confidence = result.get('average_pixel_confidence', 0.0)
                needs_review_flag = avg_confidence < CONFIDENCE_THRESHOLD
                current_image_result_summary['average_pixel_confidence'] = f"{avg_confidence:.3f}"

                if result.get('processed_mask_type') == 'multiclass':
                    phase_percentages = result.get('phase_percentages')
                    if phase_percentages:
                        percentages_str_list = [f"{name}: {perc:.1f}%" for name, perc in phase_percentages.items()]
                        current_image_result_summary['phase_percentages'] = "; ".join(percentages_str_list)
                    # For multi-class, grain_count and avg_grain_size remain N/A as per ModelHandler logic
                    current_image_result_summary['grain_count'] = 'N/A (Phase Analysis)'
                    current_image_result_summary['average_grain_size_px2'] = 'N/A (Phase Analysis)'
                    current_image_result_summary['needs_review'] = needs_review_flag # Review based on confidence
                    logger.info(f"Batch: Successfully processed {image_filename} (Multi-class). Confidence: {avg_confidence:.3f}, Review: {needs_review_flag}")
                
                elif result.get('processed_mask_type') == 'binary' and result.get('grain_count') is not None and result.get('avg_grain_size') is not None:
                    gc = result.get('grain_count', 0)
                    ags = result.get('avg_grain_size', 0.0)
                    current_image_result_summary['grain_count'] = gc
                    current_image_result_summary['average_grain_size_px2'] = f"{ags:.2f}"
                    current_image_result_summary['phase_percentages'] = 'N/A (Grain Analysis)'
                    current_image_result_summary['needs_review'] = needs_review_flag or (gc == 0)
                    logger.info(f"Batch: Successfully processed {image_filename} (Binary). Grains: {gc}, Confidence: {avg_confidence:.3f}, Review: {current_image_result_summary['needs_review']}")
                else:
                    # This handles cases where metrics are missing unexpectedly for binary, or unknown processed_mask_type
                    logger.warning(f"Batch: Analysis failed or produced incomplete metrics for {image_filename}. Confidence: {avg_confidence:.3f}. Flagging for review.")
                    current_image_result_summary['grain_count'] = current_image_result_summary.get('grain_count', 'Error - Incomplete Metrics')
                    if result and result.get('grain_count') is None and result.get('processed_mask_type') != 'multiclass':
                         current_image_result_summary['grain_count'] = 'Error - No grain count'
                    current_image_result_summary['average_grain_size_px2'] = current_image_result_summary.get('average_grain_size_px2', 'N/A')
                    current_image_result_summary['phase_percentages'] = 'N/A' if result.get('processed_mask_type') != 'multiclass' else current_image_result_summary.get('phase_percentages', 'Error - No phase data')
                    current_image_result_summary['needs_review'] = True 
                    error_count += 1
                
                processed_count +=1
                all_results_summary.append(current_image_result_summary)

            except Exception as e:
                logger.error(f"Batch: Error processing {image_filename}: {e}", exc_info=True)
                current_image_result_summary['grain_count'] = f'Error - {str(e)[:30]}...'
                current_image_result_summary['needs_review'] = True # Flag images that caused exceptions
                error_count += 1
                all_results_summary.append(current_image_result_summary)
            
            if img_to_process_cv is not None:
                self.display_image(image_path) # Display the original image first
                QApplication.processEvents() # Allow UI to update

                if result and 'phase_map_color' in result: # Check for multi-class phase map first
                    self.display_processed_image(result['phase_map_color'])
                elif result and 'raw_mask' in result and result.get('grain_boundaries') and result.get('processed_mask_type') == 'binary': 
                    # Fallback to binary overlay if no phase map but has grain boundaries
                    overlay_image = img_to_process_cv.copy()
                    cv2.drawContours(overlay_image, result['grain_boundaries'], -1, (0, 255, 0), 1)
                    self.display_processed_image(overlay_image)
                # If neither, the original image displayed by self.display_image() remains.
            QApplication.processEvents()

        self.statusBar().showMessage(f"Batch processing complete. {processed_count-error_count}/{num_images} images successfully processed. {error_count} errors.")

        if not all_results_summary:
            QMessageBox.information(self, "Batch Complete", "No results to report.")
            return

        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        report_path, _ = QFileDialog.getSaveFileName(self, "Save Batch Report", 
                                                   str(self.exports_dir / "batch_analysis_report_with_confidence.csv"), 
                                                   "CSV Files (*.csv);;All Files (*)", options=options)
        
        if report_path:
            try:
                # import csv # Already imported at top of file
                headers = ['image_filename', 'grain_count', 'average_grain_size_px2', 'average_pixel_confidence', 'phase_percentages', 'needs_review']
                with open(report_path, 'w', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers)
                    writer.writeheader()
                    writer.writerows(all_results_summary)
                logger.info(f"Batch report saved to {report_path}")
                QMessageBox.information(self, "Report Saved", f"Batch analysis report saved to:\n{report_path}")
            except Exception as e:
                logger.error(f"Failed to save batch report: {e}", exc_info=True)
                QMessageBox.critical(self, "Save Error", f"Failed to save batch report: {e}")
        else:
            QMessageBox.information(self, "Report Not Saved", "Batch analysis report was not saved.")

if __name__ == '__main__':
    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "grain_analyzer_app.log"

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)

    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

    logger.info("Application starting...")
    app = QApplication(sys.argv)

    # Define the icon path relative to the app.py file
    app_icon_path = Path(__file__).resolve().parent / "assets" / "app_icon.png"

    # Set application icon (for taskbar, etc.)
    if app_icon_path.exists():
        app.setWindowIcon(QIcon(str(app_icon_path)))
        logger.info(f"Application icon set from {app_icon_path}")
    else:
        # Create assets directory if it doesn't exist, so user knows where to put it
        (Path(__file__).resolve().parent / "assets").mkdir(exist_ok=True)
        logger.warning(f"Application icon not found at {app_icon_path}. Please create it. Using default icon.")

    main_win = MainWindow()
    
    # Set window icon for the main window instance (for title bar)
    # This is often redundant if app.setWindowIcon() works globally, but good for robustness
    if app_icon_path.exists():
        main_win.setWindowIcon(QIcon(str(app_icon_path)))
        # No need to log again, app log is sufficient

    main_win.show()
    logger.info("MainWindow shown.")
    sys.exit(app.exec_()) 