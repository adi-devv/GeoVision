# src/preprocess.py
import rasterio
import numpy as np
import os
from pathlib import Path


def load_tiff(file_path):
    """Load a single TIFF file."""
    with rasterio.open(file_path) as src:
        img = src.read()  # Shape: (bands, height, width)
        img = np.transpose(img, (1, 2, 0))  # Shape: (height, width, bands)
    return img


def normalize_image(img):
    """Normalize image to [0, 1] per band, handle NaN/invalid values."""
    img = img.astype(np.float32)
    for band in range(img.shape[-1]):
        band_data = img[:, :, band]
        valid_mask = ~np.isnan(band_data)
        if valid_mask.any():
            band_min, band_max = band_data[valid_mask].min(), band_data[valid_mask].max()
            if band_max > band_min:
                img[:, :, band] = (band_data - band_min) / (band_max - band_min)
        img[:, :, band] = np.nan_to_num(img[:, :, band], nan=0.0)
    return img


def tile_image(img, tile_size=512):
    """Split image into tiles of specified size."""
    h, w, bands = img.shape
    tiles = []
    tile_coords = []
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = img[i:i + tile_size, j:j + tile_size, :]
            if tile.shape[:2] == (tile_size, tile_size):
                tiles.append(tile)
                tile_coords.append((i, j))
    return tiles, tile_coords


def preprocess_dataset(dataset_dir, output_dir, tile_size=512):
    """Preprocess TIFF files: load, normalize, tile, save as numpy."""
    print("Starting preprocessing...")
    dataset_path = Path(dataset_dir)
    tiff_files = list(dataset_path.glob("*.tif")) + list(dataset_path.glob("*.tiff"))
    os.makedirs(output_dir, exist_ok=True)
    all_tiles = []
    all_coords = []
    all_filenames = []

    for file in tiff_files:
        print(f"Processing {file.name}...")
        img = load_tiff(file)
        img_norm = normalize_image(img)
        tiles, coords = tile_image(img_norm, tile_size)
        for idx, tile in enumerate(tiles):
            tile_filename = f"{Path(file).stem}_tile_{idx}.npy"
            np.save(os.path.join(output_dir, tile_filename), tile)
            all_tiles.append(tile)
            all_coords.append(coords[idx])
            all_filenames.append(tile_filename)

    print(f"Preprocessing done: {len(all_tiles)} tiles created.")
    return all_tiles, all_coords, all_filenames, tiff_files


if __name__ == "__main__":
    dataset_dir = "../dataset"
    output_dir = "../outputs/processed"
    tiles, coords, tile_filenames, tiff_files = preprocess_dataset(dataset_dir, output_dir)