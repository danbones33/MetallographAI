import os
from pathlib import Path
import cv2
from PIL import Image # For converting NumPy array to PIL Image for transforms
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import logging
import traceback # Added import for traceback

logger = logging.getLogger("GrainAnalyzer.DataLoader")

class MicroscopyDataset(Dataset):
    """Custom PyTorch Dataset for microscopy images and their segmentation masks."""
    def __init__(self, image_dir, mask_dir, image_transform=None, mask_transform=None):
        """
        Args:
            image_dir (str or Path): Directory with all the input images.
            mask_dir (str or Path): Directory with all the segmentation masks.
            image_transform (callable, optional): Optional transform to be applied on an image sample.
            mask_transform (callable, optional): Optional transform to be applied on a mask sample.
        """
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        
        self.image_filenames = sorted([f for f in os.listdir(self.image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])
        self.mask_filenames = sorted([f for f in os.listdir(self.mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))])

        # Basic validation: Check if the number of images and masks match
        if len(self.image_filenames) != len(self.mask_filenames):
            logger.error(f"Mismatch in number of images ({len(self.image_filenames)}) and masks ({len(self.mask_filenames)})")
            # Further checks could compare filenames without extensions
            raise ValueError("Number of images and masks do not match.")
        if len(self.image_filenames) == 0:
            logger.warning(f"No images found in {self.image_dir}. Dataset is empty.")

        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        logger.info(f"MicroscopyDataset initialized. Found {len(self.image_filenames)} samples.")
        logger.info(f"Image directory: {self.image_dir}, Mask directory: {self.mask_dir}")

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_dir / self.image_filenames[idx]
        mask_name = self.mask_dir / self.mask_filenames[idx]
        
        # It's good practice to ensure filenames (sans extension) match for image-mask pairs
        # For simplicity, this is currently assumed by sorted lists of the same length.

        try:
            # Load image using OpenCV, convert to RGB, then to PIL Image
            image_cv = cv2.imread(str(img_name))
            if image_cv is None:
                logger.error(f"Failed to load image: {img_name}")
                # Return None or a placeholder, or raise error. For now, raising.
                raise IOError(f"Could not read image: {img_name}")
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
            image_pil = Image.fromarray(image_rgb)

            # Load mask using OpenCV (as grayscale), then to PIL Image
            mask_cv = cv2.imread(str(mask_name), cv2.IMREAD_GRAYSCALE)
            if mask_cv is None:
                logger.error(f"Failed to load mask: {mask_name}")
                raise IOError(f"Could not read mask: {mask_name}")
            mask_pil = Image.fromarray(mask_cv)

        except Exception as e:
            logger.error(f"Error loading image/mask for index {idx} ({img_name}, {mask_name}): {e}")
            # Depending on policy, either raise e, or return a dummy sample, or skip.
            # Raising error is often better during development.
            raise

        if self.image_transform:
            try:
                image = self.image_transform(image_pil)
            except Exception as e:
                logger.error(f"Error applying image transform to {img_name}: {e}")
                raise
        else: # Default to ToTensor if no transform provided
            image = transforms.ToTensor()(image_pil)

        if self.mask_transform:
            try:
                # The provided transform is now responsible for outputting a LongTensor
                # of class indices, typically of shape [H, W] or [C, H, W] where C=1.
                mask = self.mask_transform(mask_pil)
            except Exception as e:
                logger.error(f"Error applying mask transform to {mask_name}: {e}")
                raise
        else:
            # Default behavior for masks if no transform is provided:
            # Convert PIL image to Tensor, preserve integer values, convert to Long, squeeze channel.
            # Assumes mask_pil contains class indices (0, 1, 2, ...).
            # Note: This default does not include resizing. A proper mask_transform is highly recommended.
            temp_mask_tensor = transforms.PILToTensor()(mask_pil) # Shape [1, H, W], dtype torch.uint8 (for typical L mode)
            mask = temp_mask_tensor.squeeze(0).to(torch.long)     # Shape [H, W], dtype torch.long
            
        # Ensure mask is binary (0 or 1) if it came from grayscale with values like 0 and 255
        # Target for BCEWithLogitsLoss should be float and values 0.0 or 1.0
        # mask = (mask > 0.5).float() # Assuming mask values are scaled [0,1] by ToTensor # THIS LINE IS REMOVED FOR MULTI-CLASS
        # If masks are 0/255, then mask = (mask / 255.0 > 0.5).float() would be more robust before ToTensor.
        # Or ensure mask_transform handles this (e.g. ToTensor() then a lambda)

        return image, mask

# Example Usage (for testing the dataset class directly)
if __name__ == '__main__':
    logger.info("Testing MicroscopyDataset...")
    
    # Create dummy directories and files for testing
    test_data_dir = Path("temp_test_dataset")
    test_img_dir = test_data_dir / "images"
    test_mask_dir = test_data_dir / "masks"
    
    os.makedirs(test_img_dir, exist_ok=True)
    os.makedirs(test_mask_dir, exist_ok=True)
    
    # Create dummy images (simple NumPy arrays saved as PNG)
    for i in range(3):
        dummy_img_arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        dummy_mask_arr = np.random.randint(0, 2, (100, 100), dtype=np.uint8) * 255 # Binary 0 or 255
        
        cv2.imwrite(str(test_img_dir / f"sample_{i+1}.png"), dummy_img_arr)
        cv2.imwrite(str(test_mask_dir / f"sample_{i+1}_mask.png"), dummy_mask_arr)

    # Define example transforms (similar to what ModelHandler would use)
    input_size = (128, 128)
    # This transform chain should expect a PIL image as input, same as MicroscopyDataset provides.
    # ModelHandler's transforms start with ToTensor(), so this test should too.
    image_transform_test = transforms.Compose([
        transforms.ToTensor(), # Converts PIL to Tensor [C, H, W] and scales to [0,1]
        transforms.Resize(input_size, antialias=True), # Operates on Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # image_transform_test_pil_first was an alternative, removing for clarity as ModelHandler sends ToTensor-first transforms
    # image_transform_test_pil_first = transforms.Compose([
    #     transforms.PILToTensor(),
    #     transforms.ToDtype(torch.float32, scale=True),
    #     transforms.Resize(input_size, antialias=True),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    mask_transform_test = transforms.Compose([
        # Expects PIL image for mask as well
        transforms.ToTensor(), # Converts PIL to Tensor [1, H, W]
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.NEAREST), # Operates on Tensor
    ])

    try:
        dataset = MicroscopyDataset(
            image_dir=test_img_dir,
            mask_dir=test_mask_dir,
            image_transform=image_transform_test, 
            mask_transform=mask_transform_test
        )
        logger.info(f"Dataset created with {len(dataset)} samples.")

        if len(dataset) > 0:
            img_tensor, mask_tensor = dataset[0]
            logger.info(f"Sample 0 image tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
            logger.info(f"Sample 0 mask tensor shape: {mask_tensor.shape}, dtype: {mask_tensor.dtype}")
            logger.info(f"Sample 0 mask unique values: {torch.unique(mask_tensor)}")
            assert mask_tensor.min() >= 0.0 and mask_tensor.max() <= 1.0, "Mask values not in [0,1]"
            assert len(torch.unique(mask_tensor)) <= 2, "Mask is not binary"

    except Exception as e:
        logger.error(f"Error during dataset test: {e}")
        logger.error(traceback.format_exc())
    finally:
        # Clean up dummy files and directory
        import shutil
        if test_data_dir.exists():
            logger.info(f"Cleaning up test directory: {test_data_dir}")
            shutil.rmtree(test_data_dir)
    logger.info("MicroscopyDataset test finished.") 