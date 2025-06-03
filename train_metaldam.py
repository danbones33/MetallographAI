import logging
from pathlib import Path
import torch # Ensure torch is imported for device selection etc.

# Assuming your components are structured in a 'components' package
from components.model_handler import ModelHandler
from components.data_loader import MicroscopyDataset # Though not directly used, good for context

# --- Configuration ---
METALDAM_BASE_DIR = Path(r"G:/apps/Grain size/training/MetalDAM")
TRAIN_IMAGES_DIR = METALDAM_BASE_DIR / "images"
TRAIN_LABELS_DIR = METALDAM_BASE_DIR / "labels" # These are the ground truth masks

# Optional: If you have a separate validation set from MetalDAM (e.g., using val.txt)
# VAL_IMAGES_DIR = METALDAM_BASE_DIR / "val_images" # Adjust if you create these
# VAL_LABELS_DIR = METALDAM_BASE_DIR / "val_labels" # Adjust if you create these
VAL_IMAGES_DIR = None # Set to None if not using a separate validation set for this script
VAL_LABELS_DIR = None # Set to None if not using a separate validation set for this script

MODEL_SAVE_DIR = Path(r"G:/apps/Grain size/models") # Directory where models are saved by ModelHandler
MODEL_SAVE_NAME = "metaldam_segmentation_v1.pt" # Desired name for the trained model

EPOCHS = 25 # Number of training epochs (adjust as needed)
BATCH_SIZE = 4 # Batch size (adjust based on your GPU memory)
LEARNING_RATE = 0.001 # Learning rate for the optimizer

# --- Configuration for resuming training ---
RESUME_TRAINING = True # Set to True to load weights and continue training
LOAD_WEIGHTS_FROM = MODEL_SAVE_DIR / "metaldam_segmentation_v1.pt" # Path to the model from v1 training
NEW_MODEL_SAVE_NAME = "metaldam_segmentation_v2.pt" # Name for the model after this training session
ADDITIONAL_EPOCHS = 25 # Number of *additional* epochs to train for

# --- Setup Logging ---
# Basic logging configuration to see outputs from ModelHandler and this script
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TrainMetalDAM")

# --- Main Training Script ---
if __name__ == "__main__":
    logger.info("Starting MetalDAM dataset training script...")
    logger.info(f"Using PyTorch version: {torch.__version__}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training will use device: {device}")

    current_epochs = ADDITIONAL_EPOCHS if RESUME_TRAINING else EPOCHS
    current_model_save_name = NEW_MODEL_SAVE_NAME if RESUME_TRAINING else MODEL_SAVE_NAME
    weights_to_load = str(LOAD_WEIGHTS_FROM) if RESUME_TRAINING and LOAD_WEIGHTS_FROM.exists() else None

    if RESUME_TRAINING and not weights_to_load:
        logger.warning(f"Resume training was set, but weights file {LOAD_WEIGHTS_FROM} not found. Starting fresh training for {current_epochs} epochs.")
        # Fallback to fresh training if specified weights are not found
        weights_to_load = None 
    elif RESUME_TRAINING:
        logger.info(f"Resuming training from: {weights_to_load}")

    if not TRAIN_IMAGES_DIR.exists() or not TRAIN_LABELS_DIR.exists():
        logger.error(f"Training data directories not found!")
        logger.error(f"Checked for images at: {TRAIN_IMAGES_DIR}")
        logger.error(f"Checked for labels at: {TRAIN_LABELS_DIR}")
        logger.error("Please ensure the MetalDAM dataset is correctly placed and paths are correct.")
        exit()
    
    logger.info(f"Training images directory: {TRAIN_IMAGES_DIR}")
    logger.info(f"Training labels directory: {TRAIN_LABELS_DIR}")
    if VAL_IMAGES_DIR and VAL_LABELS_DIR:
        logger.info(f"Validation images directory: {VAL_IMAGES_DIR}")
        logger.info(f"Validation labels directory: {VAL_LABELS_DIR}")
    else:
        logger.info("No separate validation set provided for this training run.")

    try:
        # Initialize ModelHandler - it knows where to save models from its own init
        # The images_dir and labels_dir in ModelHandler init are defaults,
        # train_model will use the specific ones passed to it.
        model_handler = ModelHandler(model_dir=MODEL_SAVE_DIR)
        
        logger.info(f"Starting training for {current_epochs} epochs...")
        
        # Define a simple progress callback (optional)
        def training_progress(epoch, total_epochs, batch, total_batches, loss, val_loss=None):
            log_str = f"Epoch: {epoch}/{total_epochs} | Batch: {batch}/{total_batches} | Loss: {loss:.4f}"
            if val_loss is not None:
                log_str += f" | Val Loss: {val_loss:.4f}"
            if batch % 10 == 0 or batch == total_batches: # Log every 10 batches or at the end of epoch
                 logger.info(log_str)


        saved_model_path = model_handler.train_model(
            epochs=current_epochs,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            train_images_dir=str(TRAIN_IMAGES_DIR),
            train_labels_dir=str(TRAIN_LABELS_DIR),
            val_images_dir=str(VAL_IMAGES_DIR) if VAL_IMAGES_DIR else None,
            val_labels_dir=str(VAL_LABELS_DIR) if VAL_LABELS_DIR else None,
            model_save_name=current_model_save_name,
            progress_callback=training_progress,
            load_weights_from_path=weights_to_load
        )
        
        logger.info(f"Training complete. Model saved to: {saved_model_path}")

    except FileNotFoundError as fnf_error:
        logger.error(f"File not found during training setup: {fnf_error}")
        logger.error("Please check your dataset paths and ensure files exist.")
    except ValueError as val_error:
        logger.error(f"Value error during training setup or execution: {val_error}")
        logger.error("This might be due to empty datasets, incorrect batch sizes, or other configuration issues.")
    except Exception as e:
        logger.error(f"An unexpected error occurred during training: {e}", exc_info=True)

    logger.info("MetalDAM dataset training script finished.") 