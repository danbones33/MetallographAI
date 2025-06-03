import sys
import os
import traceback
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("GrainAnalyzer")

try:
    logger.info("Importing Qt modules...")
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QFileDialog, QLabel, 
                            QListWidget, QStatusBar, QTabWidget, QComboBox,
                            QMessageBox, QProgressBar)
    from PyQt5.QtGui import QPixmap, QImage
    from PyQt5.QtCore import Qt, QThread, pyqtSignal
    logger.info("Qt modules imported successfully")
except Exception as e:
    logger.error(f"Error importing Qt modules: {str(e)}")
    logger.error(traceback.format_exc())
    print(f"ERROR: Could not import Qt modules: {str(e)}")
    sys.exit(1)

try:
    logger.info("Importing OpenCV and NumPy...")
    import cv2
    import numpy as np
    logger.info("OpenCV and NumPy imported successfully")
except Exception as e:
    logger.error(f"Error importing OpenCV/NumPy: {str(e)}")
    logger.error(traceback.format_exc())
    print(f"ERROR: Could not import OpenCV/NumPy: {str(e)}")
    sys.exit(1)

try:
    logger.info("Importing PyTorch...")
    import torch
    logger.info(f"PyTorch imported successfully. CUDA available: {torch.cuda.is_available()}")
except Exception as e:
    logger.error(f"Error importing PyTorch: {str(e)}")
    logger.error(traceback.format_exc())
    print(f"ERROR: Could not import PyTorch: {str(e)}")
    sys.exit(1)

try:
    # First ensure the components directory exists
    components_dir = Path("components")
    if not components_dir.exists():
        logger.error("Components directory not found. Creating it now.")
        components_dir.mkdir(exist_ok=True)
        
        # Create an empty __init__.py if it doesn't exist
        init_file = components_dir / "__init__.py"
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write("# Components initialization\n")
    
    logger.info("Importing components...")
    import components
    from components.image_processor import ImageProcessor
    from components.model_handler import ModelHandler
    from components.utils import setup_directories
    logger.info("Components imported successfully")
except Exception as e:
    logger.error(f"Error importing components: {str(e)}")
    logger.error(traceback.format_exc())
    print(f"ERROR: Could not import components: {str(e)}")
    print("Please ensure the 'components' directory exists and contains the required files.")
    sys.exit(1)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing MainWindow")
        self.setWindowTitle("Grain Size Analyzer")
        self.setGeometry(100, 100, 1200, 800)
        
        try:
            # Initialize application paths
            logger.info("Setting up paths...")
            self.setup_paths()
            logger.info("Paths set up successfully")
            
            # Initialize components
            logger.info("Initializing components...")
            self.image_processor = ImageProcessor()
            self.model_handler = ModelHandler(self.models_dir)
            logger.info("Components initialized successfully")
            
            # Set up UI
            logger.info("Initializing UI...")
            self.init_ui()
            logger.info("UI initialized successfully")
            
            # Initialize state variables
            self.current_image_path = None
            self.current_result = None
            
            logger.info("Getting available models...")
            self.available_models = self.model_handler.get_available_models()
            logger.info(f"Available models: {self.available_models}")
            
            self.update_model_dropdown()
            
            self.statusBar().showMessage("Ready")
            logger.info("MainWindow initialization complete")
        except Exception as e:
            logger.error(f"Error in MainWindow initialization: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Initialization Error", 
                              f"An error occurred during initialization: {str(e)}\n\n"
                              f"Check app_debug.log for details.")

    def setup_paths(self):
        # Create base app directory
        base_dir = Path.home() / "GrainSizeAnalyzer"
        logger.info(f"Base directory: {base_dir}")
        
        self.images_dir = base_dir / "images"
        self.labels_dir = base_dir / "labels"
        self.models_dir = base_dir / "models"
        self.exports_dir = base_dir / "exports"
        self.kb_dir = base_dir / "knowledge_base"
        
        # Create directories if they don't exist
        try:
            dirs_to_create = [
                base_dir, self.images_dir, self.labels_dir, 
                self.models_dir, self.exports_dir, self.kb_dir
            ]
            logger.info(f"Creating directories: {dirs_to_create}")
            setup_directories(dirs_to_create)
            logger.info("Directories created/verified successfully")
        except Exception as e:
            logger.error(f"Error creating directories: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def init_ui(self):
        try:
            # Main widget and layout
            central_widget = QWidget()
            main_layout = QHBoxLayout()
            
            # === Left panel (file list) ===
            left_panel = QWidget()
            left_layout = QVBoxLayout()
            
            # Image list section
            self.file_list_label = QLabel("Image Files:")
            self.file_list = QListWidget()
            self.file_list.itemClicked.connect(self.load_selected_image)
            
            # Load image button
            self.load_btn = QPushButton("Load Image")
            self.load_btn.clicked.connect(self.load_image)
            
            # Load folder button
            self.load_folder_btn = QPushButton("Load Folder")
            self.load_folder_btn.clicked.connect(self.load_folder)
            
            # Add widgets to left panel
            left_layout.addWidget(self.file_list_label)
            left_layout.addWidget(self.file_list)
            left_layout.addWidget(self.load_btn)
            left_layout.addWidget(self.load_folder_btn)
            left_panel.setLayout(left_layout)
            left_panel.setFixedWidth(250)
            
            # === Center panel (image view) ===
            center_panel = QWidget()
            center_layout = QVBoxLayout()
            
            # Image display
            self.image_label = QLabel("No image loaded")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
            
            # Add image to center panel
            center_layout.addWidget(self.image_label)
            center_panel.setLayout(center_layout)
            
            # === Right panel (controls) ===
            right_panel = QWidget()
            right_layout = QVBoxLayout()
            
            # Model selection
            model_layout = QVBoxLayout()
            model_layout.addWidget(QLabel("Select Model:"))
            self.model_dropdown = QComboBox()
            self.model_dropdown.currentIndexChanged.connect(self.change_model)
            model_layout.addWidget(self.model_dropdown)
            
            # Control buttons
            self.analyze_btn = QPushButton("Analyze Image")
            self.analyze_btn.clicked.connect(self.analyze_image)
            
            self.export_btn = QPushButton("Export Results")
            self.export_btn.clicked.connect(self.export_results)
            
            # Tab widget for different modes
            tab_widget = QTabWidget()
            
            # Analysis tab
            analysis_tab = QWidget()
            analysis_layout = QVBoxLayout()
            analysis_layout.addLayout(model_layout)
            analysis_layout.addWidget(self.analyze_btn)
            analysis_layout.addWidget(self.export_btn)
            analysis_layout.addStretch()
            analysis_tab.setLayout(analysis_layout)
            
            # Training tab
            training_tab = QWidget()
            training_layout = QVBoxLayout()
            
            self.add_labels_btn = QPushButton("Add Labeled Data")
            self.add_labels_btn.clicked.connect(self.add_labeled_data)
            
            self.train_model_btn = QPushButton("Train Model")
            self.train_model_btn.clicked.connect(self.train_model)
            
            training_layout.addWidget(self.add_labels_btn)
            training_layout.addWidget(self.train_model_btn)
            training_layout.addStretch()
            training_tab.setLayout(training_layout)
            
            # Batch processing tab
            batch_tab = QWidget()
            batch_layout = QVBoxLayout()
            
            self.batch_process_btn = QPushButton("Process Batch")
            self.batch_process_btn.clicked.connect(self.process_batch)
            
            batch_layout.addWidget(self.batch_process_btn)
            batch_layout.addStretch()
            batch_tab.setLayout(batch_layout)
            
            # Add tabs to widget
            tab_widget.addTab(analysis_tab, "Analysis")
            tab_widget.addTab(training_tab, "Training")
            tab_widget.addTab(batch_tab, "Batch Processing")
            
            # Add tab widget to right panel
            right_layout.addWidget(tab_widget)
            right_panel.setLayout(right_layout)
            right_panel.setFixedWidth(250)
            
            # Add panels to main layout
            main_layout.addWidget(left_panel)
            main_layout.addWidget(center_panel, 1)  # 1 is the stretch factor
            main_layout.addWidget(right_panel)
            
            # Set the main layout
            central_widget.setLayout(main_layout)
            self.setCentralWidget(central_widget)
            
            # Add status bar
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            
            # Add progress bar to status bar
            self.progress_bar = QProgressBar()
            self.progress_bar.setVisible(False)
            self.status_bar.addPermanentWidget(self.progress_bar)
            
            logger.info("UI setup completed")
        except Exception as e:
            logger.error(f"Error in UI initialization: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def update_model_dropdown(self):
        """Update the model dropdown with available models"""
        try:
            self.model_dropdown.clear()
            if self.available_models:
                logger.info(f"Adding models to dropdown: {self.available_models}")
                self.model_dropdown.addItems(self.available_models)
            else:
                logger.warning("No models available")
                self.model_dropdown.addItem("No models available")
        except Exception as e:
            logger.error(f"Error updating model dropdown: {str(e)}")
            logger.error(traceback.format_exc())
    
    def load_image(self):
        """Open file dialog to load an image"""
        try:
            logger.info(f"Opening file dialog at {self.images_dir}")
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Open Image", str(self.images_dir), 
                "Image files (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
            )
            
            if file_path:
                logger.info(f"Selected image: {file_path}")
                self.display_image(file_path)
                self.current_image_path = file_path
                self.status_bar.showMessage(f"Loaded image: {os.path.basename(file_path)}")
        except Exception as e:
            logger.error(f"Error loading image: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def load_folder(self):
        """Open folder dialog to load multiple images"""
        try:
            logger.info(f"Opening folder dialog at {self.images_dir}")
            folder_path = QFileDialog.getExistingDirectory(
                self, "Open Folder", str(self.images_dir)
            )
            
            if folder_path:
                logger.info(f"Selected folder: {folder_path}")
                self.file_list.clear()
                image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
                count = 0
                for file in Path(folder_path).iterdir():
                    if file.suffix.lower() in image_extensions:
                        self.file_list.addItem(file.name)
                        count += 1
                
                logger.info(f"Added {count} images to the list")
                self.status_bar.showMessage(f"Loaded folder: {os.path.basename(folder_path)}")
        except Exception as e:
            logger.error(f"Error loading folder: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to load folder: {str(e)}")
    
    def load_selected_image(self, item):
        """Load image selected from the file list"""
        try:
            file_name = item.text()
            parent_dir = str(Path(self.images_dir))
            file_path = os.path.join(parent_dir, file_name)
            
            logger.info(f"Selected image from list: {file_path}")
            
            if os.path.exists(file_path):
                self.display_image(file_path)
                self.current_image_path = file_path
                logger.info(f"Loaded image: {file_path}")
            else:
                logger.warning(f"Image does not exist: {file_path}")
        except Exception as e:
            logger.error(f"Error loading selected image: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to load selected image: {str(e)}")
    
    def display_image(self, file_path):
        """Display the selected image in the image label"""
        try:
            # Load image using OpenCV
            logger.info(f"Reading image with OpenCV: {file_path}")
            img = cv2.imread(file_path)
            
            if img is None:
                logger.error(f"Failed to read image: {file_path}")
                QMessageBox.critical(self, "Error", f"Failed to read image: {file_path}")
                return
            
            # Convert to RGB format (from BGR)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            h, w, ch = img_rgb.shape
            logger.info(f"Image dimensions: {w}x{h}, {ch} channels")
            
            # Convert to QImage
            q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            
            # Scale the image to fit in the label while maintaining aspect ratio
            pixmap = QPixmap.fromImage(q_img)
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            # Display the image
            self.image_label.setPixmap(scaled_pixmap)
            logger.info("Image displayed successfully")
        except Exception as e:
            logger.error(f"Error displaying image: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to display image: {str(e)}")
    
    def analyze_image(self):
        """Analyze the current image using the selected model"""
        try:
            if not self.current_image_path:
                logger.warning("No image loaded for analysis")
                QMessageBox.warning(self, "Warning", "Please load an image first.")
                return
                
            if not self.available_models:
                logger.warning("No models available for analysis")
                QMessageBox.warning(self, "Warning", "No models available. Please add a model first.")
                return
            
            self.status_bar.showMessage("Analyzing image...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(50)  # Set to 50% while processing
            
            # Get selected model
            selected_model = self.model_dropdown.currentText()
            logger.info(f"Analyzing with model: {selected_model}")
            
            # In a real app, this would run in a background thread
            # Load image
            img = cv2.imread(self.current_image_path)
            
            if img is None:
                logger.error(f"Failed to read image for analysis: {self.current_image_path}")
                QMessageBox.critical(self, "Error", f"Failed to read image for analysis")
                self.progress_bar.setVisible(False)
                return
            
            # Process image and get results
            logger.info("Running inference...")
            results = self.model_handler.run_inference(img, selected_model)
            logger.info(f"Inference results: {results.keys()}")
            
            # Update with overlay visualization
            logger.info("Creating visualization overlay...")
            overlay_img = self.image_processor.create_overlay(img, results)
            
            # Display the result
            logger.info("Displaying processed image...")
            self.display_processed_image(overlay_img)
            
            # Store results for later export
            self.current_result = results
            
            self.status_bar.showMessage("Analysis complete")
            logger.info("Analysis completed successfully")
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Analysis failed: {str(e)}")
            self.status_bar.showMessage("Analysis failed")
        finally:
            self.progress_bar.setVisible(False)
    
    def display_processed_image(self, img):
        """Display processed image with overlays"""
        try:
            # Convert BGR to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to QImage and display
            h, w, ch = img_rgb.shape
            q_img = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img)
            
            scaled_pixmap = pixmap.scaled(
                self.image_label.width(), self.image_label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            logger.info("Processed image displayed successfully")
        except Exception as e:
            logger.error(f"Error displaying processed image: {str(e)}")
            logger.error(traceback.format_exc())
    
    def export_results(self):
        """Export the current analysis results"""
        try:
            if not self.current_result:
                logger.warning("No results to export")
                QMessageBox.warning(self, "Warning", "No results to export. Please analyze an image first.")
                return
            
            # Get file path for saving
            logger.info(f"Opening save dialog at {self.exports_dir}")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Results", str(self.exports_dir / "results.csv"), 
                "CSV Files (*.csv);;PDF Files (*.pdf)"
            )
            
            if not file_path:
                logger.info("Export canceled")
                return
                
            # Export based on file extension
            if file_path.endswith('.csv'):
                logger.info(f"Exporting to CSV: {file_path}")
                self.image_processor.export_to_csv(self.current_result, file_path)
            elif file_path.endswith('.pdf'):
                logger.info(f"Exporting to PDF: {file_path}")
                self.image_processor.export_to_pdf(
                    self.current_result, 
                    self.current_image_path, 
                    file_path
                )
            
            self.status_bar.showMessage(f"Results exported to {file_path}")
            logger.info(f"Results exported successfully to {file_path}")
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            logger.error(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
    
    def change_model(self):
        """Handle model change from dropdown"""
        try:
            selected_model = self.model_dropdown.currentText()
            if selected_model != "No models available":
                logger.info(f"Model changed to: {selected_model}")
                self.status_bar.showMessage(f"Selected model: {selected_model}")
        except Exception as e:
            logger.error(f"Error changing model: {str(e)}")
            logger.error(traceback.format_exc())
    
    def add_labeled_data(self):
        """Add new labeled data for training"""
        QMessageBox.information(
            self, 
            "Information", 
            "This would open an interface to add new labeled data. Feature to be implemented."
        )
        logger.info("Add labeled data feature not implemented yet")
    
    def train_model(self):
        """Start training process with labeled data"""
        QMessageBox.information(
            self, 
            "Information", 
            "This would start the model training process. Feature to be implemented."
        )
        logger.info("Train model feature not implemented yet")
    
    def process_batch(self):
        """Process a batch of images"""
        QMessageBox.information(
            self, 
            "Information", 
            "This would start batch processing of multiple images. Feature to be implemented."
        )
        logger.info("Batch processing feature not implemented yet")

if __name__ == "__main__":
    try:
        logger.info("Starting application...")
        app = QApplication(sys.argv)
        logger.info("QApplication created")
        
        window = MainWindow()
        logger.info("MainWindow created")
        
        window.show()
        logger.info("Window shown")
        
        exit_code = app.exec_()
        logger.info(f"Application exiting with code: {exit_code}")
        sys.exit(exit_code)
    except Exception as e:
        logger.critical(f"Fatal error in main: {str(e)}")
        logger.critical(traceback.format_exc())
        print(f"FATAL ERROR: {str(e)}")
        if 'window' in locals():
            QMessageBox.critical(window, "Fatal Error", 
                              f"A fatal error occurred: {str(e)}\n\n"
                              f"Check app_debug.log for details.")
        sys.exit(1) 