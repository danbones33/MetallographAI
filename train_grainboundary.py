import logging
from pathlib import Path
import torch

from components.model_handler import ModelHandler

# --- Configuration ---
NEW_DATASET_BASE_DIR = Path(r"G:/apps/Grain size/knowledge_base/datasets/GRAIN DATA SET")
# Assuming RG/ and RGMask/ are the correct folders for images and masks respectively
TRAIN_IMAGES_DIR = NEW_DATASET_BASE_DIR / "RG" / "images" # dataset has RG/images structure, not RG/ directly
TRAIN_LABELS_DIR = NEW_DATASET_BASE_DIR / "RGMask" / "masks" # dataset has RGMask/masks structure, not RGMask/ directly

# It seems the Kaggle dataset has subfolders like 'train', 'test', 'val' inside RG/ and RGMask/
# For a quick start, let's assume you might have moved all relevant images/masks directly into
# RG/images and RGMask/masks, or you want to use a subset. Adjust path if structure is deeper.
# Example: TRAIN_IMAGES_DIR = NEW_DATASET_BASE_DIR / "RG" / "train" / "images" 
# If the images and masks are directly in RG/ and RGMask/ respectively:
TRAIN_IMAGES_DIR = NEW_DATASET_BASE_DIR / "AG"
TRAIN_LABELS_DIR = NEW_DATASET_BASE_DIR / "AGMask"

# Optional: Validation set paths (adjust if you have them)
VAL_IMAGES_DIR = None
VAL_LABELS_DIR = None

MODEL_SAVE_DIR = Path(r"G:/apps/Grain size/models")
MODEL_SAVE_NAME = "grainboundary_model_ag_v1.pt"

EPOCHS = 30 # Adjust as needed
BATCH_SIZE = 2 # Adjust based on your GPU memory
LEARNING_RATE = 0.001

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainGrainBoundary")

# --- Main Training Script ---
if __name__ == "__main__":
    logger.info("Starting Grain Boundary dataset training script...")
    logger.info(f"Using PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training will use device: {device}")

    if not TRAIN_IMAGES_DIR.exists() or not TRAIN_LABELS_DIR.exists():
        logger.error(f"Training data directories not found!")
        logger.error(f"Checked for images at: {TRAIN_IMAGES_DIR}")
        logger.error(f"Checked for labels at: {TRAIN_LABELS_DIR}")
        logger.error("Please ensure the dataset is correctly placed and paths are correct.")
        exit()
    
    logger.info(f"Training images directory: {TRAIN_IMAGES_DIR}")
    logger.info(f"Training labels directory: {TRAIN_LABELS_DIR}")

    try:
        model_handler = ModelHandler(model_dir=MODEL_SAVE_DIR)
        
        logger.info(f"Starting training for {EPOCHS} epochs...")
        
        def training_progress(epoch, total_epochs, batch, total_batches, loss, val_loss=None):
            log_str = f"Epoch: {epoch}/{total_epochs} | Batch: {batch}/{total_batches} | Loss: {loss:.4f}"
            if val_loss is not None:
                log_str += f" | Val Loss: {val_loss:.4f}"
            if batch % 10 == 0 or batch == total_batches:
                 logger.info(log_str)

        saved_model_path = model_handler.train_model(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            train_images_dir=str(TRAIN_IMAGES_DIR),
            train_labels_dir=str(TRAIN_LABELS_DIR),
            val_images_dir=str(VAL_IMAGES_DIR) if VAL_IMAGES_DIR else None,
            val_labels_dir=str(VAL_LABELS_DIR) if VAL_LABELS_DIR else None,
            model_save_name=MODEL_SAVE_NAME,
            progress_callback=training_progress,
            num_classes=1,  # For binary segmentation (boundary vs. non-boundary)
            is_binary_segmentation=True # Use BCEWithLogitsLoss and 0-1 mask scaling
        )
        
        logger.info(f"Training complete. Model saved to: {saved_model_path}")

    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)

    logger.info("Grain Boundary dataset training script finished.") 