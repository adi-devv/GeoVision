"""
Configuration settings for land cover segmentation pipeline
"""
import torch


class Config:
    """Global configuration for the segmentation pipeline"""

    # Directory paths
    TRAIN_IMG_DIR = '../data/train/images'
    TRAIN_MASK_DIR = '../data/train/masks'
    VALID_IMG_DIR = '../data/valid/images'
    VALID_MASK_DIR = '../data/valid/masks'
    TARGET_DIR = '../target'
    OUTPUT_DIR = '../outputs'

    # Model parameters
    PATCH_SIZE = 256
    CLASSES = ['urban', 'forest', 'water', 'land']
    NUM_CLASSES = len(CLASSES)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Training parameters
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 6  # 6 with 30 epochs is smoother n more accurate
    EPOCHS = 30

    # DeepGlobe class mappings
    CLASS_MAP = {
        (0, 255, 255): 0,  # Urban - Cyan
        (255, 255, 0): 1,  # Agriculture - Yellow
        (255, 0, 255): 2,  # Rangeland - Magenta
        (0, 255, 0): 3,  # Forest - Green
        (0, 0, 255): 4,  # Water - Blue
        (255, 255, 255): 5,  # Barren - White
        (0, 0, 0): 6  # Unknown - Black
    }

    # Target class mapping (merge similar classes)
    TARGET_MAP = {
        0: 0,  # Urban -> Urban
        1: 3,  # Agriculture -> Land
        2: 3,  # Rangeland -> Land
        3: 1,  # Forest -> Forest
        4: 2,  # Water -> Water
        5: 3,  # Barren -> Land
        6: 4  # Unknown -> Ignore
    }

    # Visualization colors
    COLORMAP = {
        0: [255, 100, 100],  # Urban = Light Red
        1: [100, 255, 100],  # Forest = Light Green
        2: [100, 100, 255],  # Water = Light Blue
        3: [255, 255, 100],  # Land = Light Yellow
        4: [128, 128, 128]  # Other/Ignore = Gray
    }
