"""
Data processing utilities and dataset classes for land cover segmentation
"""
import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import os
from config import Config

config = Config()


class ImageProcessor:
    """Handles image loading and preprocessing operations"""

    @staticmethod
    def load_tiff(path, target_size=(1000, 1200)):
        """Load and preprocess TIFF image with proper normalization"""
        with rasterio.open(path) as src:
            raw_img = src.read()
            raw_img = np.transpose(raw_img, (1, 2, 0))

            # Standardize to RGB (3 channels)
            if raw_img.shape[2] > 3:
                raw_img = raw_img[:, :, :3]
            elif raw_img.shape[2] == 1:
                raw_img = np.repeat(raw_img, 3, axis=2)

        # Resize and apply gaussian smoothing
        raw_img = resize(raw_img, target_size, anti_aliasing=True)
        for c in range(raw_img.shape[2]):
            raw_img[:, :, c] = gaussian(raw_img[:, :, c], sigma=1)

        # Normalize based on data type
        raw_img = ImageProcessor._normalize_image(raw_img)
        return np.clip(raw_img, 0, 1)

    @staticmethod
    def _normalize_image(img):
        """Normalize image based on its data type"""
        if img.dtype == np.uint8:
            return img.astype(np.float32) / 255.0
        elif img.dtype == np.uint16:
            return img.astype(np.float32) / 65535.0
        else:
            # For float data, normalize to [0, 1]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                return (img - img_min) / (img_max - img_min)
            return np.zeros_like(img)

    @staticmethod
    def create_patches(img, patch_size):
        """Split image into patches with padding for edge cases"""
        patches, positions = [], []
        h, w = img.shape[:2]

        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                end_i = min(i + patch_size, h)
                end_j = min(j + patch_size, w)
                patch = img[i:end_i, j:end_j]

                # Pad patch if smaller than required size
                if patch.shape[:2] != (patch_size, patch_size):
                    padded_patch = np.zeros((patch_size, patch_size, img.shape[2]), dtype=patch.dtype)
                    padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                    patch = padded_patch

                patches.append(patch)
                positions.append((i, j, end_i, end_j))

        return patches, positions


class DeepGlobeDataset(Dataset):
    """Dataset class for DeepGlobe land cover segmentation"""

    def __init__(self, img_dir, mask_dir, patch_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size

        self.images = self._find_image_mask_pairs(img_dir, mask_dir)
        print(f'Found {len(self.images)} image-mask pairs')

    @staticmethod
    def _find_image_mask_pairs(img_dir, mask_dir):
        """Find matching image-mask pairs"""
        img_files = os.listdir(img_dir)
        mask_files = os.listdir(mask_dir)
        pairs = []

        for img_file in img_files:
            base_name = os.path.splitext(img_file)[0]
            mask_name = base_name.replace('_sat', '_mask') + '.png'
            if mask_name in mask_files:
                pairs.append((img_file, mask_name))

        return pairs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file, mask_file = self.images[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Load and process image
        img = ImageProcessor.load_tiff(img_path, (self.patch_size, self.patch_size))

        # Load and process mask with fallback methods
        try:
            mask_single = self._load_mask_rasterio(mask_path)
        except Exception:
            try:
                mask_single = self._load_mask_matplotlib(mask_path)
            except Exception as e:
                print(f'Failed to load mask {mask_path}: {e}')
                mask_single = np.full((self.patch_size, self.patch_size), 4, dtype=np.uint8)

        return torch.from_numpy(img.transpose(2, 0, 1)).float(), torch.from_numpy(mask_single).long()

    def _load_mask_rasterio(self, mask_path):
        """Load mask using rasterio"""
        with rasterio.open(mask_path) as src:
            mask = src.read()

            # Standardize mask format
            if len(mask.shape) == 3 and mask.shape[0] <= 4:
                mask = np.transpose(mask, (1, 2, 0))
            elif len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=2)

            mask = self._process_mask(mask)
            return self._convert_rgb_to_classes(mask)

    def _load_mask_matplotlib(self, mask_path):
        """Fallback mask loading using matplotlib"""
        mask = plt.imread(mask_path)
        mask = self._process_mask(mask)
        return self._convert_rgb_to_classes(mask)

    def _process_mask(self, mask):
        """Process mask to standard format"""
        # Resize if needed
        if mask.shape[:2] != (self.patch_size, self.patch_size):
            if len(mask.shape) == 3:
                mask = resize(mask, (self.patch_size, self.patch_size, mask.shape[2]),
                              anti_aliasing=False, preserve_range=True)
            else:
                mask = resize(mask, (self.patch_size, self.patch_size),
                              anti_aliasing=False, preserve_range=True)

        # Convert to uint8
        if mask.dtype != np.uint8:
            if mask.max() <= 1.0:
                mask = (mask * 255).astype(np.uint8)
            else:
                mask = mask.astype(np.uint8)

        # Ensure RGB format
        if len(mask.shape) == 2:
            mask = np.stack([mask] * 3, axis=2)
        elif mask.shape[2] == 1:
            mask = np.repeat(mask, 3, axis=2)
        elif mask.shape[2] > 3:
            mask = mask[:, :, :3]

        return mask

    def _convert_rgb_to_classes(self, mask):
        """Convert RGB mask to single-channel class mask"""
        mask_single = np.full((self.patch_size, self.patch_size), 4, dtype=np.uint8)

        for rgb, cls in config.CLASS_MAP.items():
            rgb_array = np.array(rgb, dtype=np.uint8)
            # Allow some tolerance in RGB matching
            mask_pixels = np.all(np.abs(mask.astype(np.int16) - rgb_array.astype(np.int16)) <= 10, axis=2)
            target_cls = config.TARGET_MAP[cls]
            mask_single[mask_pixels] = target_cls

        return mask_single
