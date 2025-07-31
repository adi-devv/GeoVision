import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os
from pathlib import Path
import rasterio
from skimage.transform import resize


def evaluate_clustering(features, labels):
    """Evaluate clustering using silhouette score."""
    print("Evaluating clustering...")
    if len(np.unique(labels)) > 1:
        score = silhouette_score(features, labels)
        print(f"Silhouette Score: {score:.4f}")
    else:
        print("Silhouette Score: N/A (single cluster)")
    print("Evaluation done.")


def visualize_clusters(labels, tile_files, tiff_files, output_dir, tile_size=512):
    """Visualize clusters as map overlay with downsampling."""
    print("Starting visualization...")
    os.makedirs(output_dir, exist_ok=True)
    colors = {0: [1, 0, 0], 1: [0, 1, 0], 2: [0, 0, 1]}  # Red, Green, Blue

    for tiff_file in tiff_files:
        print(f"Visualizing {Path(tiff_file).stem}...")
        with rasterio.open(tiff_file) as src:
            h, w = src.height, src.width
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        for label, tile_file in zip(labels, tile_files):
            tile_idx = int(Path(tile_file).stem.split("_tile_")[-1])
            row = (tile_idx // (w // tile_size)) * tile_size
            col = (tile_idx % (w // tile_size)) * tile_size
            overlay[row:row + tile_size, col:col + tile_size, :] = colors[label % len(colors)]

        # Downsample the overlay to reduce memory usage (e.g., to 1/4 the resolution)
        target_size = (h // 4, w // 4)  # Adjust divisor as needed
        overlay = resize(overlay, target_size, anti_aliasing=True)

        plt.figure(figsize=(10, 10))
        plt.imshow(overlay)
        plt.title(f"Terrain Classification: {Path(tiff_file).stem}")
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f"{Path(tiff_file).stem}_overlay.png"), dpi=300, bbox_inches='tight')
        plt.close()
    print("Visualization done.")