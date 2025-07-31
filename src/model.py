# src/model.py
import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.cluster import KMeans
import numpy as np
import os
from pathlib import Path


def load_resnet18(device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Load pre-trained ResNet18, remove final layer for feature extraction."""
    model = models.resnet18(pretrained=True)
    model = nn.Sequential(*list(model.children())[:-1])
    model.eval()
    model.to(device)
    return model, device


def extract_features(tiles, model, device, batch_size=32):
    """Extract features from tiles using ResNet18."""
    print("Starting feature extraction...")
    features = []
    with torch.no_grad():
        for i in range(0, len(tiles), batch_size):
            batch = tiles[i:i + batch_size]
            batch = np.stack(batch)
            batch = torch.tensor(batch).permute(0, 3, 1, 2).float().to(device)
            batch_features = model(batch).cpu().numpy().squeeze()
            features.append(batch_features)
    features = np.vstack(features)
    print("Feature extraction done.")
    return features


def cluster_features(features, n_clusters=3):
    """Apply K-means clustering to features."""
    print("Starting clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    print("Clustering done.")
    return labels, kmeans


def process_tiles(tile_dir, n_clusters=3):
    """Process tiles: extract features and cluster."""
    print("Processing tiles...")
    tile_files = list(Path(tile_dir).glob("*.npy"))
    tiles = [np.load(f) for f in tile_files]
    model, device = load_resnet18()
    features = extract_features(tiles, model, device)
    labels, kmeans = cluster_features(features, n_clusters)
    print("Tile processing done.")
    return labels, tile_files
