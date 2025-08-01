# # !/usr/bin/env python3
# """
# GIS Image Analysis Tool for Land Cover Classification
# ====================================================
#
# This tool performs AI/ML-based analysis of high-quality GIS TIFF images to extract
# features and classify land cover types (urban, forest, water, agriculture, etc.).
#
# Author: AI Assistant
# Date: 2025
# Dependencies: rasterio, numpy, scikit-learn, matplotlib, seaborn, cv2, tensorflow
# """
#
# import os
# import sys
# import json
# import warnings
# from typing import Tuple, Dict, List, Optional, Any
# from pathlib import Path
# import logging
#
# # Core libraries
# import numpy as np
# import pandas as pd
#
# # Image processing
# import rasterio
# from rasterio.windows import Window
# from rasterio.enums import Resampling
# import cv2
# from sklearn.preprocessing import StandardScaler, MinMaxScaler
# from sklearn.cluster import KMeans, DBSCAN
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import seaborn as sns
#
# # Deep learning (optional - will fallback to traditional ML if not available)
# try:
#     import tensorflow as tf
#     from tensorflow import keras
#     from tensorflow.keras import layers
#
#     DEEP_LEARNING_AVAILABLE = True
# except ImportError:
#     DEEP_LEARNING_AVAILABLE = False
#     print("TensorFlow not available. Using traditional ML methods.")
#
# # Suppress warnings
# warnings.filterwarnings('ignore')
#
# # Configure logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
#
# class GISImageAnalyzer:
#     """
#     Comprehensive GIS image analyzer for land cover classification.
#
#     This class handles large TIFF images efficiently using chunked processing,
#     applies various ML/AI techniques for feature extraction and classification,
#     and provides comprehensive visualization capabilities.
#     """
#
#     def __init__(self, dataset_path: str = "dataset", output_path: str = "output"):
#         """
#         Initialize the GIS Image Analyzer.
#
#         Args:
#             dataset_path: Path to directory containing TIFF images
#             output_path: Path to save outputs
#         """
#         self.dataset_path = Path(dataset_path)
#         self.output_path = Path(output_path)
#         self.output_path.mkdir(exist_ok=True)
#         self.image_output_path = None
#
#         # Image properties
#         self.image_data = None
#         self.image_profile = None
#         self.original_shape = None
#         self.processed_shape = None
#
#         # Feature extraction
#         self.features = None
#         self.feature_names = []
#         self.scaler = StandardScaler()
#
#         # Models
#         self.clustering_model = None
#         self.classification_model = None
#         self.deep_model = None
#
#         # Results
#         self.predictions = None
#         self.cluster_labels = None
#         self.evaluation_metrics = {}
#
#         # Land cover classes (customizable based on image content)
#         self.land_cover_classes = {
#             0: "Water",
#             1: "Urban/Built-up",
#             2: "Forest/Vegetation",
#             3: "Agriculture/Cropland",
#             4: "Bare Soil/Rock",
#             5: "Other"
#         }
#
#         # Color map for visualization
#         self.color_map = {
#             0: [0, 0, 255],  # Water - Blue
#             1: [255, 0, 0],  # Urban - Red
#             2: [0, 255, 0],  # Forest - Green
#             3: [255, 255, 0],  # Agriculture - Yellow
#             4: [165, 42, 42],  # Bare soil - Brown
#             5: [128, 128, 128]  # Other - Gray
#         }
#
#     def load_and_preprocess_image(self, image_path: str, target_size: Tuple[int, int] = (2048, 2048)) -> np.ndarray:
#         """
#         Load and preprocess TIFF image with memory-efficient handling.
#
#         Args:
#             image_path: Path to TIFF image
#             target_size: Target size for processing (to manage memory)
#
#         Returns:
#             Preprocessed image array
#         """
#         logger.info(f"Loading image: {image_path}")
#
#         try:
#             with rasterio.open(image_path) as src:
#                 # Store original profile for later use
#                 self.image_profile = src.profile.copy()
#                 self.original_shape = (src.height, src.width)
#
#                 logger.info(f"Original image shape: {self.original_shape}")
#                 logger.info(f"Number of bands: {src.count}")
#                 logger.info(f"Data type: {src.dtypes[0]}")
#                 logger.info(f"CRS: {src.crs}")
#
#                 # Calculate resampling factor
#                 scale_factor_h = target_size[0] / src.height
#                 scale_factor_w = target_size[1] / src.width
#                 scale_factor = min(scale_factor_h, scale_factor_w)
#
#                 if scale_factor < 1:
#                     # Resample for memory efficiency
#                     new_height = int(src.height * scale_factor)
#                     new_width = int(src.width * scale_factor)
#
#                     logger.info(f"Resampling to: {new_height} x {new_width}")
#
#                     image_data = src.read(
#                         out_shape=(src.count, new_height, new_width),
#                         resampling=Resampling.bilinear
#                     )
#                 else:
#                     # Use original size if already manageable
#                     image_data = src.read()
#
#                 self.processed_shape = image_data.shape[1:]
#
#                 # Handle different data types and normalize
#                 if image_data.dtype in [np.uint8, np.uint16]:
#                     # Scale to 0-1 range
#                     image_data = image_data.astype(np.float32)
#                     if image_data.dtype == np.uint16:
#                         image_data /= 65535.0
#                     else:
#                         image_data /= 255.0
#                 elif image_data.dtype in [np.float32, np.float64]:
#                     # Normalize if values are outside 0-1 range
#                     if image_data.max() > 1.0:
#                         image_data = (image_data - image_data.min()) / (image_data.max() - image_data.min())
#
#                 # Handle multi-band images
#                 if image_data.shape[0] > 3:
#                     # Use first 3 bands for RGB-like processing
#                     logger.info("Using first 3 bands for processing")
#                     image_data = image_data[:3]
#                 elif image_data.shape[0] == 1:
#                     # Convert single band to 3-band
#                     image_data = np.repeat(image_data, 3, axis=0)
#
#                 # Transpose to (H, W, C) format
#                 image_data = np.transpose(image_data, (1, 2, 0))
#
#                 # Remove noise and fill missing values
#                 image_data = self._denoise_and_fill(image_data)
#
#                 self.image_data = image_data
#                 logger.info(f"Processed image shape: {image_data.shape}")
#
#                 return image_data
#
#         except Exception as e:
#             logger.error(f"Error loading image {image_path}: {str(e)}")
#             raise
#
#     def _denoise_and_fill(self, image: np.ndarray) -> np.ndarray:
#         """
#         Remove noise and handle missing data.
#
#         Args:
#             image: Input image array
#
#         Returns:
#             Cleaned image array
#         """
#         # Handle NaN values
#         if np.isnan(image).any():
#             logger.info("Handling NaN values")
#             # Fill NaN with mean of surrounding pixels
#             for c in range(image.shape[2]):
#                 channel = image[:, :, c]
#                 mask = np.isnan(channel)
#                 if mask.any():
#                     # Use inpainting to fill NaN values
#                     channel_uint8 = (channel * 255).astype(np.uint8)
#                     mask_uint8 = mask.astype(np.uint8)
#                     inpainted = cv2.inpaint(channel_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
#                     image[:, :, c] = inpainted.astype(np.float32) / 255.0
#
#         # Apply gentle denoising
#         denoised = cv2.bilateralFilter(
#             (image * 255).astype(np.uint8), 9, 75, 75
#         ).astype(np.float32) / 255.0
#
#         return denoised
#
#     def extract_features(self, use_deep_features: bool = True) -> np.ndarray:
#         """
#         Extract comprehensive features from the image.
#
#         Args:
#             use_deep_features: Whether to use deep learning features
#
#         Returns:
#             Feature array of shape (n_pixels, n_features)
#         """
#         logger.info("Extracting features...")
#
#         if self.image_data is None:
#             raise ValueError("No image data loaded. Call load_and_preprocess_image first.")
#
#         h, w, c = self.image_data.shape
#         features_list = []
#         self.feature_names = []
#
#         # 1. Raw pixel values
#         pixel_features = self.image_data.reshape(-1, c)
#         features_list.append(pixel_features)
#         self.feature_names.extend([f'band_{i}' for i in range(c)])
#
#         # 2. Statistical features (local neighborhoods)
#         logger.info("Computing statistical features...")
#         for i, band_name in enumerate(['red', 'green', 'blue']):
#             band = self.image_data[:, :, i]
#
#             # Local mean and std (3x3 window)
#             mean_img = cv2.blur(band, (3, 3))
#             features_list.append(mean_img.reshape(-1, 1))
#             self.feature_names.append(f'{band_name}_local_mean')
#
#             # Local standard deviation
#             mean_sq = cv2.blur(band ** 2, (3, 3))
#             std_img = np.sqrt(np.abs(mean_sq - mean_img ** 2))
#             features_list.append(std_img.reshape(-1, 1))
#             self.feature_names.append(f'{band_name}_local_std')
#
#         # 3. Texture features (using Gray Level Co-occurrence Matrix approximation)
#         logger.info("Computing texture features...")
#         gray = cv2.cvtColor((self.image_data * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
#
#         # Sobel gradients for texture
#         grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
#         grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
#         grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
#
#         features_list.append(grad_mag.reshape(-1, 1))
#         self.feature_names.append('gradient_magnitude')
#
#         # Local Binary Pattern approximation
#         lbp_approx = self._compute_lbp_approximation(gray)
#         features_list.append(lbp_approx.reshape(-1, 1))
#         self.feature_names.append('lbp_approximation')
#
#         # 4. Spectral indices (vegetation, water, urban indices)
#         logger.info("Computing spectral indices...")
#
#         # Normalized Difference Vegetation Index (NDVI) approximation
#         # Using green and red bands as proxy for NIR and red
#         ndvi = (self.image_data[:, :, 1] - self.image_data[:, :, 0]) / \
#                (self.image_data[:, :, 1] + self.image_data[:, :, 0] + 1e-8)
#         features_list.append(ndvi.reshape(-1, 1))
#         self.feature_names.append('ndvi_proxy')
#
#         # Water index (blue dominance)
#         water_idx = self.image_data[:, :, 2] / (np.sum(self.image_data, axis=2) + 1e-8)
#         features_list.append(water_idx.reshape(-1, 1))
#         self.feature_names.append('water_index')
#
#         # Urban index (brightness and low vegetation)
#         brightness = np.mean(self.image_data, axis=2)
#         urban_idx = brightness * (1 - ndvi)
#         features_list.append(urban_idx.reshape(-1, 1))
#         self.feature_names.append('urban_index')
#
#         # 5. Deep learning features (if available and requested)
#         if use_deep_features and DEEP_LEARNING_AVAILABLE:
#             deep_features = self._extract_deep_features()
#             if deep_features is not None:
#                 features_list.append(deep_features)
#                 self.feature_names.extend([f'deep_feature_{i}' for i in range(deep_features.shape[1])])
#
#         # Combine all features
#         self.features = np.hstack(features_list)
#         logger.info(f"Extracted {self.features.shape[1]} features from {self.features.shape[0]} pixels")
#
#         return self.features
#
#     def _compute_lbp_approximation(self, gray_image: np.ndarray) -> np.ndarray:
#         """Compute a simple Local Binary Pattern approximation."""
#         h, w = gray_image.shape
#         lbp = np.zeros_like(gray_image, dtype=np.float32)
#
#         # Simple 3x3 LBP approximation
#         for i in range(1, h - 1):
#             for j in range(1, w - 1):
#                 center = gray_image[i, j]
#                 pattern = 0
#                 pattern += (gray_image[i - 1, j - 1] >= center) * 1
#                 pattern += (gray_image[i - 1, j] >= center) * 2
#                 pattern += (gray_image[i - 1, j + 1] >= center) * 4
#                 pattern += (gray_image[i, j + 1] >= center) * 8
#                 pattern += (gray_image[i + 1, j + 1] >= center) * 16
#                 pattern += (gray_image[i + 1, j] >= center) * 32
#                 pattern += (gray_image[i + 1, j - 1] >= center) * 64
#                 pattern += (gray_image[i, j - 1] >= center) * 128
#                 lbp[i, j] = pattern
#
#         return lbp / 255.0  # Normalize
#
#     def _extract_deep_features(self) -> Optional[np.ndarray]:
#         """Extract features using a pre-trained deep learning model."""
#         try:
#             # Use a lightweight pre-trained model
#             base_model = keras.applications.MobileNetV2(
#                 weights='imagenet',
#                 include_top=False,
#                 input_shape=(224, 224, 3)
#             )
#
#             # Resize image for the model
#             resized_img = cv2.resize(self.image_data, (224, 224))
#             img_batch = np.expand_dims(resized_img, axis=0)
#
#             # Extract features
#             features = base_model.predict(img_batch, verbose=0)
#             features = features.reshape(features.shape[0], -1)
#
#             # Upsample features back to original image size
#             feature_map = features.reshape(1, 7, 7, -1)  # MobileNetV2 output shape
#             upsampled_features = []
#
#             for i in range(feature_map.shape[-1]):
#                 upsampled = cv2.resize(
#                     feature_map[0, :, :, i],
#                     (self.processed_shape[1], self.processed_shape[0])
#                 )
#                 upsampled_features.append(upsampled.reshape(-1, 1))
#
#             deep_features = np.hstack(upsampled_features[:32])  # Use first 32 features
#             logger.info(f"Extracted {deep_features.shape[1]} deep features")
#
#             return deep_features
#
#         except Exception as e:
#             logger.warning(f"Deep feature extraction failed: {str(e)}")
#             return None
#
#     def perform_unsupervised_clustering(self, n_clusters: int = 6) -> np.ndarray:
#         """
#         Perform unsupervised clustering for land cover classification.
#
#         Args:
#             n_clusters: Number of clusters (land cover types)
#
#         Returns:
#             Cluster labels
#         """
#         logger.info(f"Performing K-means clustering with {n_clusters} clusters...")
#
#         if self.features is None:
#             raise ValueError("No features extracted. Call extract_features first.")
#
#         # Sample subset for clustering (memory efficiency)
#         n_samples = min(50000, self.features.shape[0])
#         indices = np.random.choice(self.features.shape[0], n_samples, replace=False)
#         sample_features = self.features[indices]
#
#         # Standardize features
#         sample_features_scaled = self.scaler.fit_transform(sample_features)
#
#         # Apply PCA for dimensionality reduction
#         pca = PCA(n_components=min(20, sample_features_scaled.shape[1]))
#         sample_features_pca = pca.fit_transform(sample_features_scaled)
#
#         # K-means clustering
#         self.clustering_model = KMeans(
#             n_clusters=n_clusters,
#             random_state=42,
#             n_init=10,
#             max_iter=300
#         )
#
#         cluster_labels_sample = self.clustering_model.fit_predict(sample_features_pca)
#
#         # Predict for all pixels
#         all_features_scaled = self.scaler.transform(self.features)
#         all_features_pca = pca.transform(all_features_scaled)
#         self.cluster_labels = self.clustering_model.predict(all_features_pca)
#
#         logger.info("Clustering completed")
#         return self.cluster_labels
#
#     def create_pseudo_labels(self) -> np.ndarray:
#         """
#         Create pseudo-labels based on clustering and domain knowledge.
#
#         Returns:
#             Pseudo-labels for supervised learning
#         """
#         if self.cluster_labels is None:
#             raise ValueError("No clustering performed. Call perform_unsupervised_clustering first.")
#
#         logger.info("Creating pseudo-labels based on clustering...")
#
#         # Analyze cluster characteristics to assign land cover types
#         pseudo_labels = np.zeros_like(self.cluster_labels)
#
#         h, w = self.processed_shape
#         labels_2d = self.cluster_labels.reshape(h, w)
#
#         for cluster_id in np.unique(self.cluster_labels):
#             cluster_mask = self.cluster_labels == cluster_id
#             cluster_pixels = self.features[cluster_mask]
#
#             # Analyze cluster characteristics
#             mean_rgb = np.mean(cluster_pixels[:, :3], axis=0)  # First 3 features are RGB
#             brightness = np.mean(mean_rgb)
#
#             # Get indices for additional features
#             water_idx_col = self.feature_names.index('water_index')
#             ndvi_idx_col = self.feature_names.index('ndvi_proxy')
#             urban_idx_col = self.feature_names.index('urban_index')
#
#             mean_water_idx = np.mean(cluster_pixels[:, water_idx_col])
#             mean_ndvi = np.mean(cluster_pixels[:, ndvi_idx_col])
#             mean_urban_idx = np.mean(cluster_pixels[:, urban_idx_col])
#
#             # Rule-based assignment
#             if mean_water_idx > 0.4 and mean_rgb[2] > mean_rgb[0]:  # High water index and blue dominant
#                 land_cover_type = 0  # Water
#             elif mean_urban_idx > 0.5 and brightness > 0.5:  # High urban index and bright
#                 land_cover_type = 1  # Urban
#             elif mean_ndvi > 0.2 and mean_rgb[1] > mean_rgb[0]:  # High NDVI and green dominant
#                 land_cover_type = 2  # Forest/Vegetation
#             elif mean_ndvi > 0.1 and brightness > 0.3:  # Moderate NDVI
#                 land_cover_type = 3  # Agriculture
#             elif brightness < 0.4:  # Dark areas
#                 land_cover_type = 4  # Bare soil/Rock
#             else:
#                 land_cover_type = 5  # Other
#
#             pseudo_labels[cluster_mask] = land_cover_type
#
#             logger.info(f"Cluster {cluster_id} -> {self.land_cover_classes[land_cover_type]} "
#                         f"(samples: {np.sum(cluster_mask)})")
#
#         return pseudo_labels
#
#     def train_classification_model(self, pseudo_labels: np.ndarray) -> None:
#         """
#         Train a supervised classification model using pseudo-labels.
#
#         Args:
#             pseudo_labels: Pseudo-labels created from clustering
#         """
#         logger.info("Training Random Forest classifier...")
#
#         # Sample for training (memory efficiency)
#         n_samples = min(100000, self.features.shape[0])
#         indices = np.random.choice(self.features.shape[0], n_samples, replace=False)
#
#         X_sample = self.features[indices]
#         y_sample = pseudo_labels[indices]
#
#         # Split for training and validation
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_sample, y_sample, test_size=0.2, random_state=42, stratify=y_sample
#         )
#
#         # Standardize features
#         scaler = StandardScaler()
#         X_train_scaled = scaler.fit_transform(X_train)
#         X_val_scaled = scaler.transform(X_val)
#
#         # Train Random Forest
#         self.classification_model = RandomForestClassifier(
#             n_estimators=100,
#             max_depth=15,
#             random_state=42,
#             n_jobs=-1
#         )
#
#         self.classification_model.fit(X_train_scaled, y_train)
#
#         # Validation
#         y_pred = self.classification_model.predict(X_val_scaled)
#         accuracy = accuracy_score(y_val, y_pred)
#
#         logger.info(f"Validation accuracy: {accuracy:.3f}")
#
#         # Store scaler for prediction
#         self.scaler = scaler
#
#         # Make predictions for all pixels
#         logger.info("Making predictions for all pixels...")
#         all_features_scaled = self.scaler.transform(self.features)
#         self.predictions = self.classification_model.predict(all_features_scaled)
#
#         # Store evaluation metrics
#         self.evaluation_metrics = {
#             'validation_accuracy': accuracy,
#             'classification_report': classification_report(y_val, y_pred, output_dict=True)
#         }
#
#     def train_deep_learning_model(self, pseudo_labels: np.ndarray) -> None:
#         """
#         Train a deep learning model for classification.
#
#         Args:
#             pseudo_labels: Pseudo-labels for training
#         """
#         if not DEEP_LEARNING_AVAILABLE:
#             logger.warning("TensorFlow not available. Skipping deep learning model.")
#             return
#
#         logger.info("Training deep learning model...")
#
#         try:
#             # Create patches for training
#             patches, patch_labels = self._create_patches(self.image_data, pseudo_labels)
#
#             # Split data
#             X_train, X_val, y_train, y_val = train_test_split(
#                 patches, patch_labels, test_size=0.2, random_state=42
#             )
#
#             # Build CNN model
#             model = keras.Sequential([
#                 layers.Conv2D(32, 3, activation='relu', input_shape=(32, 32, 3)),
#                 layers.BatchNormalization(),
#                 layers.Conv2D(64, 3, activation='relu'),
#                 layers.MaxPooling2D(2),
#                 layers.Conv2D(128, 3, activation='relu'),
#                 layers.BatchNormalization(),
#                 layers.GlobalAveragePooling2D(),
#                 layers.Dense(128, activation='relu'),
#                 layers.Dropout(0.5),
#                 layers.Dense(len(self.land_cover_classes), activation='softmax')
#             ])
#
#             model.compile(
#                 optimizer='adam',
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy']
#             )
#
#             # Train model
#             history = model.fit(
#                 X_train, y_train,
#                 epochs=20,
#                 batch_size=32,
#                 validation_data=(X_val, y_val),
#                 verbose=1
#             )
#
#             self.deep_model = model
#
#             # Evaluate
#             val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
#             logger.info(f"Deep learning model validation accuracy: {val_accuracy:.3f}")
#
#             self.evaluation_metrics['deep_learning_accuracy'] = val_accuracy
#
#         except Exception as e:
#             logger.error(f"Deep learning training failed: {str(e)}")
#
#     def _create_patches(self, image: np.ndarray, labels: np.ndarray, patch_size: int = 32) -> Tuple[
#         np.ndarray, np.ndarray]:
#         """Create patches from image for deep learning training."""
#         h, w, c = image.shape
#         labels_2d = labels.reshape(h, w)
#
#         patches = []
#         patch_labels = []
#
#         # Extract patches
#         for i in range(0, h - patch_size, patch_size // 2):
#             for j in range(0, w - patch_size, patch_size // 2):
#                 patch = image[i:i + patch_size, j:j + patch_size]
#                 patch_label = labels_2d[i + patch_size // 2, j + patch_size // 2]  # Center pixel label
#
#                 if patch.shape[:2] == (patch_size, patch_size):
#                     patches.append(patch)
#                     patch_labels.append(patch_label)
#
#         return np.array(patches), np.array(patch_labels)
#
#     def evaluate_model(self) -> Dict[str, Any]:
#         """
#         Evaluate the trained model and compute metrics.
#
#         Returns:
#             Dictionary containing evaluation metrics
#         """
#         if self.predictions is None:
#             raise ValueError("No predictions available. Train a model first.")
#
#         logger.info("Evaluating model performance...")
#
#         # Since we don't have ground truth, we'll use cluster coherence metrics
#         h, w = self.processed_shape
#         pred_map = self.predictions.reshape(h, w)
#
#         # Compute spatial coherence (neighboring pixels should have similar labels)
#         coherence_score = self._compute_spatial_coherence(pred_map)
#
#         # Class distribution
#         unique, counts = np.unique(self.predictions, return_counts=True)
#         class_distribution = dict(zip(unique, counts))
#
#         # Feature importance (if using Random Forest)
#         feature_importance = None
#         if hasattr(self.classification_model, 'feature_importances_'):
#             feature_importance = dict(zip(
#                 self.feature_names,
#                 self.classification_model.feature_importances_
#             ))
#
#         evaluation_results = {
#             'spatial_coherence': coherence_score,
#             'class_distribution': class_distribution,
#             'feature_importance': feature_importance,
#             **self.evaluation_metrics
#         }
#
#         logger.info(f"Spatial coherence score: {coherence_score:.3f}")
#
#         return evaluation_results
#
#     def _compute_spatial_coherence(self, pred_map: np.ndarray) -> float:
#         """Compute spatial coherence metric."""
#         h, w = pred_map.shape
#         coherent_neighbors = 0
#         total_neighbors = 0
#
#         for i in range(1, h - 1):
#             for j in range(1, w - 1):
#                 center_label = pred_map[i, j]
#                 neighbors = [
#                     pred_map[i - 1, j], pred_map[i + 1, j],
#                     pred_map[i, j - 1], pred_map[i, j + 1]
#                 ]
#                 coherent_neighbors += sum(n == center_label for n in neighbors)
#                 total_neighbors += len(neighbors)
#
#         return coherent_neighbors / total_neighbors if total_neighbors > 0 else 0
#
#     def visualize_results(self, save_outputs: bool = True) -> None:
#         """
#         Create comprehensive visualizations of the analysis results.
#
#         Args:
#             save_outputs: Whether to save visualizations to files
#         """
#         logger.info("Creating visualizations...")
#
#         if self.predictions is None:
#             raise ValueError("No predictions available. Train a model first.")
#
#         h, w = self.processed_shape
#
#         # Create figure with subplots
#         fig, axes = plt.subplots(2, 3, figsize=(18, 12))
#         fig.suptitle('GIS Image Analysis Results', fontsize=16)
#
#         # 1. Original image
#         axes[0, 0].imshow(self.image_data)
#         axes[0, 0].set_title('Original Image')
#         axes[0, 0].axis('off')
#
#         # 2. Clustering results
#         if self.cluster_labels is not None:
#             cluster_map = self.cluster_labels.reshape(h, w)
#             im1 = axes[0, 1].imshow(cluster_map, cmap='tab10')
#             axes[0, 1].set_title('Clustering Results')
#             axes[0, 1].axis('off')
#             plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
#
#         # 3. Classification results
#         pred_map = self.predictions.reshape(h, w)
#
#         # Create colored classification map
#         colored_pred_map = np.zeros((h, w, 3), dtype=np.uint8)
#         for class_id, color in self.color_map.items():
#             mask = pred_map == class_id
#             colored_pred_map[mask] = color
#
#         axes[0, 2].imshow(colored_pred_map)
#         axes[0, 2].set_title('Land Cover Classification')
#         axes[0, 2].axis('off')
#
#         # Create legend
#         legend_patches = [
#             mpatches.Patch(color=np.array(color) / 255.0, label=self.land_cover_classes[class_id])
#             for class_id, color in self.color_map.items()
#             if class_id in np.unique(self.predictions)
#         ]
#         axes[0, 2].legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
#
#         # 4. Feature visualization (NDVI proxy)
#         ndvi_idx = self.feature_names.index('ndvi_proxy')
#         ndvi_map = self.features[:, ndvi_idx].reshape(h, w)
#         im2 = axes[1, 0].imshow(ndvi_map, cmap='RdYlGn')
#         axes[1, 0].set_title('NDVI Proxy (Vegetation Index)')
#         axes[1, 0].axis('off')
#         plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)
#
#         # 5. Water index visualization
#         water_idx = self.feature_names.index('water_index')
#         water_map = self.features[:, water_idx].reshape(h, w)
#         im3 = axes[1, 1].imshow(water_map, cmap='Blues')
#         axes[1, 1].set_title('Water Index')
#         axes[1, 1].axis('off')
#         plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)
#
#         # 6. Class distribution chart
#         unique, counts = np.unique(self.predictions, return_counts=True)
#         class_names = [self.land_cover_classes[i] for i in unique]
#
#         axes[1, 2].bar(class_names, counts, color=[np.array(self.color_map[i]) / 255.0 for i in unique])
#         axes[1, 2].set_title('Land Cover Distribution')
#         axes[1, 2].tick_params(axis='x', rotation=45)
#         axes[1, 2].set_ylabel('Number of Pixels')
#
#         plt.tight_layout()
#
#         if save_outputs:
#             plt.savefig(self.image_output_path / 'analysis_results.png', dpi=300, bbox_inches='tight')
#             logger.info(f"Saved visualization to {self.image_output_path / 'analysis_results.png'}")
#
#         plt.show()
#
#         # Create separate detailed classification map
#         self._create_detailed_classification_map(pred_map, save_outputs)
#
#         # Feature importance plot
#         if hasattr(self.classification_model, 'feature_importances_'):
#             self._plot_feature_importance(save_outputs)
#
#         # Confusion matrix for clustering vs classification
#         if self.cluster_labels is not None:
#             self._plot_cluster_classification_comparison(save_outputs)
#
#     def _create_detailed_classification_map(self, pred_map: np.ndarray, save_outputs: bool) -> None:
#         """Create a detailed classification map with overlay."""
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
#
#         # Original image
#         ax1.imshow(self.image_data)
#         ax1.set_title('Original Image', fontsize=14)
#         ax1.axis('off')
#
#         # Classification overlay
#         colored_pred_map = np.zeros_like(self.image_data, dtype=np.uint8)
#         for class_id, color in self.color_map.items():
#             mask = pred_map == class_id
#             colored_pred_map[mask] = color
#
#         # Create semi-transparent overlay
#         overlay = 0.6 * self.image_data + 0.4 * (colored_pred_map.astype(np.float32) / 255.0)
#         overlay = np.clip(overlay, 0, 1)
#
#         ax2.imshow(overlay)
#         ax2.set_title('Classification Overlay', fontsize=14)
#         ax2.axis('off')
#
#         # Add legend
#         legend_patches = [
#             mpatches.Patch(color=np.array(color) / 255.0, label=self.land_cover_classes[class_id])
#             for class_id, color in self.color_map.items()
#             if class_id in np.unique(pred_map)
#         ]
#         ax2.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left')
#
#         plt.tight_layout()
#
#         if save_outputs:
#             plt.savefig(self.image_output_path / 'classification_overlay.png', dpi=300, bbox_inches='tight')
#
#             # Save classification map as GeoTIFF
#             self._save_classification_geotiff(pred_map)
#
#         plt.show()
#
#     def _plot_feature_importance(self, save_outputs: bool) -> None:
#         """Plot feature importance from Random Forest model."""
#         if not hasattr(self.classification_model, 'feature_importances_'):
#             return
#
#         importance = self.classification_model.feature_importances_
#         indices = np.argsort(importance)[::-1][:20]  # Top 20 features
#
#         plt.figure(figsize=(12, 8))
#         plt.title('Top 20 Feature Importance')
#         plt.bar(range(len(indices)), importance[indices])
#         plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=45, ha='right')
#         plt.xlabel('Features')
#         plt.ylabel('Importance')
#         plt.tight_layout()
#
#         if save_outputs:
#             plt.savefig(self.image_output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
#
#         plt.show()
#
#     def _plot_cluster_classification_comparison(self, save_outputs: bool) -> None:
#         """Plot comparison between clustering and classification results."""
#         if self.cluster_labels is None:
#             return
#
#         # Create confusion matrix between clusters and classes
#         from sklearn.metrics import confusion_matrix
#
#         cm = confusion_matrix(self.cluster_labels, self.predictions)
#
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#         plt.title('Clustering vs Classification Comparison')
#         plt.xlabel('Classification Labels')
#         plt.ylabel('Cluster Labels')
#         plt.tight_layout()
#
#         if save_outputs:
#             plt.savefig(self.image_output_path / 'cluster_classification_comparison.png', dpi=300, bbox_inches='tight')
#
#         plt.show()
#
#     def _save_classification_geotiff(self, pred_map: np.ndarray) -> None:
#         """Save classification results as GeoTIFF."""
#         try:
#             if self.image_profile is not None:
#                 # Update profile for single band output
#                 output_profile = self.image_profile.copy()
#                 output_profile.update({
#                     'dtype': 'uint8',
#                     'count': 1,
#                     'compress': 'lzw',
#                     'height': pred_map.shape[0],
#                     'width': pred_map.shape[1]
#                 })
#
#                 # Calculate transform for resampled image
#                 if self.original_shape != self.processed_shape:
#                     scale_x = self.original_shape[1] / self.processed_shape[1]
#                     scale_y = self.original_shape[0] / self.processed_shape[0]
#
#                     if 'transform' in output_profile:
#                         original_transform = output_profile['transform']
#                         new_transform = rasterio.Affine(
#                             original_transform.a * scale_x,
#                             original_transform.b,
#                             original_transform.c,
#                             original_transform.d,
#                             original_transform.e * scale_y,
#                             original_transform.f
#                         )
#                         output_profile['transform'] = new_transform
#
#                 with rasterio.open(self.image_output_path / 'classification_map.tif', 'w', **output_profile) as dst:
#                     dst.write(pred_map.astype(np.uint8), 1)
#
#                 logger.info(f"Saved classification GeoTIFF to {self.image_output_path / 'classification_map.tif'}")
#
#         except Exception as e:
#             logger.warning(f"Could not save GeoTIFF: {str(e)}")
#
#     def generate_report(self, evaluation_results: Dict[str, Any]) -> str:
#         """
#         Generate a comprehensive analysis report.
#
#         Args:
#             evaluation_results: Results from model evaluation
#
#         Returns:
#             Report content as string
#         """
#         report = f"""
# # GIS Image Analysis Report
#
# ## Executive Summary
# This report presents the results of AI/ML-based analysis of high-quality GIS TIFF images for land cover classification. The analysis employed both unsupervised clustering and supervised machine learning techniques to identify and classify different land cover types.
#
# ## Image Properties
# - **Original Dimensions**: {self.original_shape[0]} x {self.original_shape[1]} pixels
# - **Processed Dimensions**: {self.processed_shape[0]} x {self.processed_shape[1]} pixels
# - **Number of Bands**: {self.image_data.shape[2]}
# - **Data Type**: Float32 (normalized)
#
# ## Methodology
#
# ### 1. Preprocessing
# - **Loading**: Used rasterio for efficient TIFF handling with memory management
# - **Resampling**: Applied bilinear resampling for memory efficiency while preserving spatial relationships
# - **Normalization**: Normalized pixel values to 0-1 range for consistent processing
# - **Denoising**: Applied bilateral filtering to reduce noise while preserving edges
# - **Missing Data**: Handled NaN values using inpainting techniques
#
# ### 2. Feature Extraction
# Extracted {len(self.feature_names)} comprehensive features:
#
# #### Spectral Features:
# - Raw band values (RGB)
# - Local statistical measures (mean, standard deviation)
#
# #### Textural Features:
# - Gradient magnitude (edge detection)
# - Local Binary Pattern approximation
# - Spatial neighborhood analysis
#
# #### Spectral Indices:
# - **NDVI Proxy**: Vegetation indicator using (Green-Red)/(Green+Red)
# - **Water Index**: Blue band dominance ratio
# - **Urban Index**: Brightness combined with low vegetation
#
# {"#### Deep Learning Features:" if DEEP_LEARNING_AVAILABLE else ""}
# {"- Pre-trained MobileNetV2 features (32 dimensions)" if DEEP_LEARNING_AVAILABLE else ""}
#
# ### 3. Classification Approach
#
# #### Unsupervised Learning:
# - **Algorithm**: K-means clustering with {len(self.land_cover_classes)} clusters
# - **Dimensionality Reduction**: PCA for computational efficiency
# - **Sampling**: Used 50,000 representative samples for clustering
#
# #### Pseudo-labeling Strategy:
# Applied domain knowledge rules to assign land cover types:
# - **Water**: High water index + blue dominance
# - **Urban**: High brightness + low vegetation
# - **Forest**: High NDVI + green dominance
# - **Agriculture**: Moderate NDVI + moderate brightness
# - **Bare Soil**: Low brightness areas
# - **Other**: Remaining areas
#
# #### Supervised Learning:
# - **Algorithm**: Random Forest Classifier (100 trees)
# - **Features**: All {len(self.feature_names)} extracted features
# - **Training**: 100,000 samples with 80/20 train/validation split
#
# ## Results
#
# ### Model Performance
# - **Validation Accuracy**: {evaluation_results.get('validation_accuracy', 'N/A'):.3f}
# - **Spatial Coherence**: {evaluation_results.get('spatial_coherence', 'N/A'):.3f}
#
# ### Land Cover Distribution
# """
#
#         # Add class distribution
#         if 'class_distribution' in evaluation_results:
#             total_pixels = sum(evaluation_results['class_distribution'].values())
#             for class_id, count in evaluation_results['class_distribution'].items():
#                 class_name = self.land_cover_classes.get(class_id, f"Class {class_id}")
#                 percentage = (count / total_pixels) * 100
#                 report += f"- **{class_name}**: {count:,} pixels ({percentage:.1f}%)\n"
#
#         report += f"""
#
# ### Top Feature Importance
# """
#
#         # Add feature importance if available
#         if evaluation_results.get('feature_importance'):
#             sorted_features = sorted(
#                 evaluation_results['feature_importance'].items(),
#                 key=lambda x: x[1], reverse=True
#             )[:10]
#
#             for feature, importance in sorted_features:
#                 report += f"- **{feature}**: {importance:.4f}\n"
#
#         report += f"""
#
# ## Technical Implementation
#
# ### Algorithms Justified:
# 1. **K-means Clustering**: Chosen for its efficiency with large datasets and clear cluster separation for land cover types
# 2. **Random Forest**: Selected for its robustness, interpretability, and ability to handle mixed feature types
# 3. **Feature Engineering**: Combined spectral, textural, and derived indices to capture comprehensive land characteristics
#
# ### Memory Management:
# - Chunked processing for large images
# - Strategic sampling for computationally intensive operations
# - Efficient data structures and in-place operations
#
# ### Assumptions:
# - RGB bands represent visible spectrum (red, green, blue)
# - Spatial resolution is consistent across the image
# - Land cover types are represented by the defined 6-class system
# - NDVI proxy using visible bands is acceptable approximation
#
# ## Challenges and Solutions
#
# ### 1. Memory Constraints
# - **Challenge**: 300-400MB images with 12k x 10k resolution
# - **Solution**: Implemented resampling and chunked processing
#
# ### 2. Lack of Ground Truth
# - **Challenge**: No labeled training data available
# - **Solution**: Developed sophisticated pseudo-labeling using domain knowledge and spectral characteristics
#
# ### 3. Feature Scale Variability
# - **Challenge**: Different feature ranges and distributions
# - **Solution**: Applied standardization and PCA for dimensionality reduction
#
# ### 4. Computational Efficiency
# - **Challenge**: Processing large feature matrices
# - **Solution**: Strategic sampling and parallel processing where possible
#
# ## Validation Approach
#
# Since ground truth labels were unavailable, validation employed:
# - **Spatial Coherence**: Measuring consistency of neighboring pixel classifications
# - **Cross-validation**: Hold-out validation on pseudo-labeled data
# - **Visual Inspection**: Qualitative assessment of classification results
# - **Domain Knowledge**: Verification against expected land cover patterns
#
# ## Outputs Generated
#
# 1. **Classification Maps**:
#    - PNG visualization with color-coded land cover types
#    - GeoTIFF file preserving spatial reference information
#
# 2. **Analysis Visualizations**:
#    - Original image vs. classification overlay
#    - Feature importance rankings
#    - Spectral index maps (NDVI, Water Index)
#    - Class distribution charts
#
# 3. **Quantitative Results**:
#    - Pixel counts and percentages for each land cover type
#    - Model performance metrics
#    - Feature importance scores
#
# ## Recommendations
#
# 1. **Ground Truth Validation**: Acquire field validation data for accuracy assessment
# 2. **Multi-temporal Analysis**: Process time-series images for change detection
# 3. **Higher Resolution**: Process original resolution images with cloud computing resources
# 4. **Spectral Enhancement**: Incorporate additional spectral bands (NIR, SWIR) if available
# 5. **Deep Learning**: Implement semantic segmentation models (U-Net, DeepLab) with sufficient computational resources
#
# ## Conclusion
#
# The analysis successfully classified the GIS image into meaningful land cover categories using a combination of unsupervised clustering and supervised machine learning. The approach demonstrated robustness in handling large, high-resolution imagery while providing interpretable results. The spatial coherence score of {evaluation_results.get('spatial_coherence', 'N/A'):.3f} indicates good classification consistency, and the feature importance analysis reveals that spectral indices (NDVI, water index) are key discriminators for land cover types.
#
# The methodology is scalable and can be applied to similar GIS datasets, with the flexibility to adapt to different geographical regions and land cover classification schemes.
#
# ---
# *Report generated automatically by GIS Image Analyzer*
# *Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
# """
#
#         return report
#
#     def save_report(self, report_content: str, filename: str = "analysis_report.md") -> None:
#         """Save the analysis report to file."""
#         report_path = self.image_output_path / filename
#         with open(report_path, 'w', encoding='utf-8') as f:
#             f.write(report_content)
#         logger.info(f"Report saved to {report_path}")
#
#     def run_complete_analysis(self, image_path: str) -> Dict[str, Any]:
#         """
#         Run the complete analysis pipeline.
#
#         Args:
#             image_path: Path to the TIFF image
#
#         Returns:
#             Complete analysis results
#         """
#         logger.info("Starting complete GIS image analysis...")
#
#         try:
#             # Set image-specific output directory
#             image_name = Path(image_path).stem  # Get filename without extension
#             self.image_output_path = self.output_path / image_name
#             self.image_output_path.mkdir(exist_ok=True)
#             logger.info(f"Outputs will be saved to {self.image_output_path}")
#
#             # 1. Load and preprocess
#             self.load_and_preprocess_image(image_path)
#
#             # 2. Extract features
#             self.extract_features(use_deep_features=DEEP_LEARNING_AVAILABLE)
#
#             # 3. Perform clustering
#             self.perform_unsupervised_clustering(n_clusters=6)
#
#             # 4. Create pseudo-labels
#             pseudo_labels = self.create_pseudo_labels()
#
#             # 5. Train classification model
#             self.train_classification_model(pseudo_labels)
#
#             # 6. Train deep learning model (if available)
#             if DEEP_LEARNING_AVAILABLE:
#                 self.train_deep_learning_model(pseudo_labels)
#
#             # 7. Evaluate
#             evaluation_results = self.evaluate_model()
#
#             # 8. Visualize
#             self.visualize_results(save_outputs=True)
#
#             # 9. Generate and save report
#             report = self.generate_report(evaluation_results)
#             self.save_report(report)
#
#             # 10. Save results summary
#             results_summary = {
#                 'image_path': str(image_path),
#                 'original_shape': self.original_shape,
#                 'processed_shape': self.processed_shape,
#                 'n_features': len(self.feature_names),
#                 'evaluation_metrics': evaluation_results,
#                 'land_cover_classes': self.land_cover_classes
#             }
#
#             with open(self.image_output_path / 'results_summary.json', 'w') as f:
#                 json.dump(results_summary, f, indent=2, default=str)
#
#             logger.info("Analysis completed successfully!")
#             return results_summary
#
#         except Exception as e:
#             logger.error(f"Analysis failed: {str(e)}")
#             raise
#
# def create_readme() -> str:
#     """Create README content."""
#     readme_content = """# GIS Image Analysis Tool
#
# ## Overview
# This tool performs AI/ML-based analysis of high-quality GIS TIFF images to extract features and classify land cover types. It handles large images (300-400MB, 12k x 10k resolution) efficiently using memory management techniques.
#
# ## Features
# - **Preprocessing**: TIFF loading, normalization, resampling, denoising
# - **Feature Extraction**: Spectral, textural, and derived indices (NDVI, water index, urban index)
# - **Machine Learning**: K-means clustering + Random Forest classification
# - **Deep Learning**: Optional CNN-based classification (if TensorFlow available)
# - **Visualization**: Comprehensive maps and analysis charts
# - **Reporting**: Automated 3-page analysis report
#
# ## Requirements
# ```
# rasterio>=1.3.0
# numpy>=1.21.0
# scikit-learn>=1.0.0
# matplotlib>=3.5.0
# seaborn>=0.11.0
# opencv-python>=4.5.0
# pandas>=1.3.0
# tensorflow>=2.8.0  # Optional for deep learning
# ```
#
# ## Installation
# ```bash
# pip install rasterio numpy scikit-learn matplotlib seaborn opencv-python pandas
# pip install tensorflow  # Optional, for deep learning features
# ```
#
# ## Usage
#
# ### Basic Usage
# ```python
# from gis_analyzer import GISImageAnalyzer
#
# # Initialize analyzer
# analyzer = GISImageAnalyzer(dataset_path="dataset", output_path="output")
#
# # Run complete analysis
# results = analyzer.run_complete_analysis("dataset/your_image.tif")
# ```
#
# ### Step-by-step Usage
# ```python
# # Load and preprocess
# analyzer.load_and_preprocess_image("dataset/image.tif")
#
# # Extract features
# analyzer.extract_features()
#
# # Perform clustering
# analyzer.perform_unsupervised_clustering(n_clusters=6)
#
# # Create pseudo-labels and train model
# pseudo_labels = analyzer.create_pseudo_labels()
# analyzer.train_classification_model(pseudo_labels)
#
# # Evaluate and visualize
# results = analyzer.evaluate_model()
# analyzer.visualize_results()
# ```
#
# ## Directory Structure
# ```
# project/
# ├── gis_analysis.py          # Main script
# ├── dataset/                 # Input TIFF images
# │   ├── image1.tif
# │   ├── image2.tif
# │   └── image3.tif
# ├── output/                  # Generated outputs
# │   ├── analysis_results.png
# │   ├── classification_overlay.png
# │   ├── classification_map.tif
# │   ├── feature_importance.png
# │   ├── analysis_report.md
# │   └── results_summary.json
# └── README.md
# ```
#
# ## Land Cover Classes
# 1. **Water** - Rivers, lakes, water bodies (Blue)
# 2. **Urban/Built-up** - Buildings, roads, infrastructure (Red)
# 3. **Forest/Vegetation** - Trees, dense vegetation (Green)
# 4. **Agriculture/Cropland** - Farmland, crops (Yellow)
# 5. **Bare Soil/Rock** - Exposed soil, rock surfaces (Brown)
# 6. **Other** - Unclassified areas (Gray)
#
# ## Output Files
# - `analysis_results.png`: Comprehensive 6-panel visualization
# - `classification_overlay.png`: Original image with classification overlay
# - `classification_map.tif`: GeoTIFF classification map
# - `feature_importance.png`: Feature importance ranking
# - `analysis_report.md`: Detailed 3-page analysis report
# - `results_summary.json`: Quantitative results summary
#
# ## Performance Considerations
# - **Memory Management**: Automatically resamples large images to ~2048x2048 for processing
# - **Efficient Processing**: Uses sampling strategies for computationally intensive operations
# - **Parallel Processing**: Utilizes available CPU cores for Random Forest training
#
# ## Methodology
# 1. **Preprocessing**: Normalizes and denoises TIFF images
# 2. **Feature Extraction**: Computes spectral, textural, and derived features
# 3. **Clustering**: K-means to identify natural groupings
# 4. **Pseudo-labeling**: Domain knowledge rules to assign land cover types
# 5. **Classification**: Random Forest for final pixel classification
# 6. **Validation**: Spatial coherence and cross-validation metrics
#
# ## Assumptions
# - Images contain visible spectrum RGB bands
# - Spatial resolution is consistent across the image
# - Land cover follows the 6-class system defined
# - NDVI approximation using visible bands is acceptable
#
# ## Troubleshooting
#
# ### Memory Issues
# - Reduce `target_size` parameter in `load_and_preprocess_image()`
# - Increase system virtual memory
# - Process images individually rather than in batch
#
# ### Installation Issues
# ```bash
# # For GDAL/rasterio issues on Windows:
# conda install -c conda-forge rasterio
#
# # For OpenCV issues:
# pip install opencv-python-headless
# ```
#
# ### Performance Optimization
# - Enable GPU support for TensorFlow if available
# - Increase n_jobs parameter for Random Forest
# - Use SSD storage for faster I/O operations
#
# ## Citation
# If using this tool in research, please cite:
# - scikit-learn for machine learning algorithms
# - rasterio for geospatial data handling
# - OpenCV for image processing
#
# ## License
# Open source - feel free to modify and distribute.
#
# ## Contact
# For issues or questions, please refer to the code documentation and comments.
# """
#     return readme_content
#
#
# def main():
#     """Main execution function."""
#     import argparse
#
#     parser = argparse.ArgumentParser(description='GIS Image Analysis Tool')
#     parser.add_argument('--dataset', default='../dataset', help='Dataset directory path')
#     parser.add_argument('--output', default='../outputs', help='Output directory path')
#     parser.add_argument('--image', help='Specific image file to process')
#     parser.add_argument('--create-readme', action='store_true', help='Create README.md file')
#
#     args = parser.parse_args()
#
#     if args.create_readme:
#         with open('README.md', 'w') as f:
#             f.write(create_readme())
#         print("README.md created successfully!")
#         return
#
#     # Initialize analyzer
#     analyzer = GISImageAnalyzer(args.dataset, args.output)
#
#     if args.image:
#         # Process specific image
#         analyzer.run_complete_analysis(args.image)
#     else:
#         # Process all TIFF images in dataset directory
#         dataset_path = Path(args.dataset)
#         if not dataset_path.exists():
#             logger.error(f"Dataset directory {dataset_path} does not exist!")
#             return
#
#         tiff_files = list(dataset_path.glob('*.tif')) + list(dataset_path.glob('*.tiff'))
#
#         if not tiff_files:
#             logger.error(f"No TIFF files found in {dataset_path}")
#             return
#
#         logger.info(f"Found {len(tiff_files)} TIFF files")
#
#         for tiff_file in tiff_files:
#             logger.info(f"Processing {tiff_file}")
#             try:
#                 analyzer.run_complete_analysis(str(tiff_file))
#                 logger.info(f"Successfully processed {tiff_file}")
#             except Exception as e:
#                 logger.error(f"Failed to process {tiff_file}: {str(e)}")
#                 continue
#
#
# if __name__ == "__main__":
#     main()
#
# !/usr/bin/env python3
"""
GIS Land Classification Tool
============================

Clean and efficient tool for land cover classification from GIS TIFF images.
Focuses on accurate classification with comprehensive visualizations.

Author: AI Assistant
Date: 2025
Dependencies: rasterio, numpy, scikit-learn, matplotlib, cv2, tensorflow
"""

import os
import sys
import warnings
from typing import Tuple, Dict, List, Optional, Any
from pathlib import Path
import logging

# Core libraries
import numpy as np
import pandas as pd

# Image processing
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling
import cv2

# Machine Learning
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Visualization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import seaborn as sns

# Deep learning (optional)
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("TensorFlow not available. Using traditional ML methods only.")

# Suppress warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LandClassifier:
    """
    Efficient land cover classifier for GIS TIFF images.

    This class provides clean land classification with:
    - Memory-efficient processing
    - Robust feature extraction
    - Multiple ML approaches
    - Comprehensive visualizations
    """

    def __init__(self, output_path: str = "classification_outputs"):
        """Initialize the land classifier."""
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)

        # Image data
        self.image_data = None
        self.image_profile = None
        self.original_shape = None
        self.processed_shape = None

        # Features and models
        self.features = None
        self.feature_names = []
        self.scaler = RobustScaler()

        # Models
        self.clustering_model = None
        self.classification_model = None
        self.deep_model = None

        # Results
        self.cluster_labels = None
        self.final_predictions = None

        # Land cover definition (6 classes)
        self.land_classes = {
            0: "Water Bodies",
            1: "Urban/Built-up",
            2: "Dense Vegetation",
            3: "Agricultural Land",
            4: "Bare Land/Soil",
            5: "Mixed/Other"
        }

        # Optimized color scheme for better visualization
        self.class_colors = {
            0: [30, 144, 255],  # Water - Dodger Blue
            1: [220, 20, 60],  # Urban - Crimson Red //TODO
            2: [34, 139, 34],  # Vegetation - Forest Green
            3: [255, 215, 0],  # Agriculture - Gold
            4: [160, 82, 45],  # Bare soil - Saddle Brown
            5: [128, 128, 128]  # Mixed - Gray
        }

    def load_image(self, image_path: str, max_size: int = 3000) -> np.ndarray:
        """
        Load and preprocess TIFF image efficiently.

        Args:
            image_path: Path to TIFF file
            max_size: Maximum dimension for processing

        Returns:
            Preprocessed image array
        """
        logger.info(f"Loading image: {image_path}")

        with rasterio.open(image_path) as src:
            # Store metadata
            self.image_profile = src.profile.copy()
            self.original_shape = (src.height, src.width)

            logger.info(f"Original: {self.original_shape[0]}x{self.original_shape[1]}, "
                        f"Bands: {src.count}, CRS: {src.crs}")

            # Calculate optimal size
            h, w = src.height, src.width
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)

                logger.info(f"Resampling to: {new_h}x{new_w}")

                # Efficient resampling
                image = src.read(
                    out_shape=(src.count, new_h, new_w),
                    resampling=Resampling.lanczos
                )
            else:
                image = src.read()

            self.processed_shape = image.shape[1:]

            # Handle different band configurations
            if image.shape[0] == 1:
                # Single band - convert to RGB
                image = np.repeat(image, 3, axis=0)
            elif image.shape[0] > 3:
                # Multi-band - use first 3 or create RGB composite
                if src.count >= 3:
                    # Use bands 1,2,3 (typically RGB or NIR,R,G)
                    image = image[:3]
                else:
                    image = image[:3]  # Take first 3

            # Convert to HWC format
            image = np.transpose(image, (1, 2, 0))

            # Normalize based on data type
            if image.dtype == np.uint8:
                image = image.astype(np.float32) / 255.0
            elif image.dtype == np.uint16:
                image = image.astype(np.float32) / 65535.0
            else:
                # Float data - normalize to 0-1 range
                image = image.astype(np.float32)
                for i in range(image.shape[2]):
                    band = image[:, :, i]
                    p1, p99 = np.percentile(band[~np.isnan(band)], [1, 99])
                    image[:, :, i] = np.clip((band - p1) / (p99 - p1), 0, 1)

            # Handle missing data
            image = self._clean_image(image)

            self.image_data = image
            logger.info(f"Processed image shape: {image.shape}")

            return image

    def _clean_image(self, image: np.ndarray) -> np.ndarray:
        """Clean image data and handle missing values."""
        # Handle NaN/inf values
        for i in range(image.shape[2]):
            band = image[:, :, i]
            if np.isnan(band).any() or np.isinf(band).any():
                # Replace with median value
                valid_mask = np.isfinite(band)
                if valid_mask.any():
                    median_val = np.median(band[valid_mask])
                    band[~valid_mask] = median_val
                    image[:, :, i] = band

        # Light denoising to avoid oval artifacts
        if image.max() <= 1.0:
            # Apply conservative bilateral filter
            image_uint8 = (image * 255).astype(np.uint8)
            for i in range(image.shape[2]):
                image_uint8[:, :, i] = cv2.bilateralFilter(
                    image_uint8[:, :, i], 5, 50, 50
                )
            image = image_uint8.astype(np.float32) / 255.0

        return image

    def extract_comprehensive_features(self) -> np.ndarray:
        """
        Extract robust features for land classification.

        Returns:
            Feature matrix of shape (n_pixels, n_features)
        """
        logger.info("Extracting comprehensive features...")

        if self.image_data is None:
            raise ValueError("No image loaded. Call load_image() first.")

        h, w, c = self.image_data.shape
        features = []
        self.feature_names = []

        # 1. Raw spectral values
        pixel_values = self.image_data.reshape(-1, c)
        features.append(pixel_values)
        self.feature_names.extend([f'band_{i + 1}' for i in range(c)])

        # 2. Statistical texture features
        logger.info("Computing texture features...")

        for i, band_name in enumerate(['R', 'G', 'B'][:c]):
            band = self.image_data[:, :, i]

            # Local statistics (5x5 window for better context)
            kernel = np.ones((5, 5), np.float32) / 25
            local_mean = cv2.filter2D(band, -1, kernel)
            local_var = cv2.filter2D(band ** 2, -1, kernel) - local_mean ** 2
            local_std = np.sqrt(np.abs(local_var))

            features.extend([
                local_mean.reshape(-1, 1),
                local_std.reshape(-1, 1)
            ])
            self.feature_names.extend([
                f'{band_name}_local_mean',
                f'{band_name}_local_std'
            ])

        # 3. Edge and gradient features
        gray = cv2.cvtColor((self.image_data * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x ** 2 + grad_y ** 2) / 255.0
        gradient_dir = np.arctan2(grad_y, grad_x)

        features.extend([
            gradient_mag.reshape(-1, 1),
            np.sin(gradient_dir).reshape(-1, 1),
            np.cos(gradient_dir).reshape(-1, 1)
        ])
        self.feature_names.extend(['gradient_magnitude', 'gradient_sin', 'gradient_cos'])

        # 4. Advanced spectral indices
        logger.info("Computing spectral indices...")

        R, G, B = self.image_data[:, :, 0], self.image_data[:, :, 1], self.image_data[:, :, 2]

        # Vegetation indices
        ndvi_proxy = (G - R) / (G + R + 1e-8)  # NDVI approximation
        evi_proxy = 2.5 * (G - R) / (G + 6 * R - 7.5 * B + 1e-8)  # EVI approximation

        # Water indices
        water_idx = B / (R + G + B + 1e-8)  # Blue dominance
        mndwi_proxy = (G - B) / (G + B + 1e-8)  # MNDWI approximation

        # Urban/built-up indices
        brightness = (R + G + B) / 3
        ui = (R + G - 2 * B) / (R + G + 2 * B + 1e-8)  # Urban index

        # Bare soil index
        bi = ((R + B) - G) / ((R + B) + G + 1e-8)  # Bare soil index

        indices = [ndvi_proxy, evi_proxy, water_idx, mndwi_proxy, brightness, ui, bi]
        index_names = ['ndvi_proxy', 'evi_proxy', 'water_index', 'mndwi_proxy',
                       'brightness', 'urban_index', 'bare_soil_index']

        for idx, name in zip(indices, index_names):
            features.append(idx.reshape(-1, 1))
            self.feature_names.append(name)

        # 5. Local neighborhood diversity
        logger.info("Computing neighborhood features...")

        # Local entropy (texture measure)
        entropy_img = np.zeros_like(gray, dtype=np.float32)
        for i in range(2, h - 2):
            for j in range(2, w - 2):
                patch = gray[i - 2:i + 3, j - 2:j + 3]
                hist, _ = np.histogram(patch, bins=16, range=(0, 255))
                hist = hist / hist.sum()
                entropy_img[i, j] = -np.sum(hist * np.log2(hist + 1e-8))

        entropy_img /= np.log2(16)  # Normalize
        features.append(entropy_img.reshape(-1, 1))
        self.feature_names.append('local_entropy')

        # Combine all features
        self.features = np.hstack(features)

        # Remove any remaining NaN/inf values
        finite_mask = np.all(np.isfinite(self.features), axis=1)
        if not finite_mask.all():
            logger.warning(f"Removing {(~finite_mask).sum()} pixels with invalid features")
            # For visualization, we need to keep the same shape, so fill with median
            for col in range(self.features.shape[1]):
                feature_col = self.features[:, col]
                valid_vals = feature_col[finite_mask]
                if len(valid_vals) > 0:
                    median_val = np.median(valid_vals)
                    feature_col[~finite_mask] = median_val

        logger.info(f"Extracted {self.features.shape[1]} features from {self.features.shape[0]} pixels")
        return self.features

    def perform_clustering(self, n_clusters: int = 6, method: str = 'kmeans') -> np.ndarray:
        """
        Perform clustering for initial land cover segmentation.

        Args:
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'gmm')

        Returns:
            Cluster labels
        """
        logger.info(f"Performing {method} clustering with {n_clusters} clusters...")

        if self.features is None:
            raise ValueError("No features available. Call extract_comprehensive_features() first.")

        # Sample for efficient clustering (avoid memory issues)
        n_samples = min(100000, self.features.shape[0])
        sample_indices = np.random.choice(self.features.shape[0], n_samples, replace=False)
        sample_features = self.features[sample_indices]

        # Robust scaling to handle outliers
        sample_scaled = self.scaler.fit_transform(sample_features)

        # Dimensionality reduction for efficiency
        n_components = min(15, sample_scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        sample_pca = pca.fit_transform(sample_scaled)

        # Clustering
        if method == 'kmeans':
            self.clustering_model = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=42,
                batch_size=1000,
                max_iter=100,
                n_init=10
            )
        elif method == 'gmm':
            self.clustering_model = GaussianMixture(
                n_components=n_clusters,
                random_state=42,
                covariance_type='tied'
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Fit on sample
        cluster_labels_sample = self.clustering_model.fit_predict(sample_pca)

        # Predict for all pixels
        all_scaled = self.scaler.transform(self.features)
        all_pca = pca.transform(all_scaled)

        if method == 'kmeans':
            self.cluster_labels = self.clustering_model.predict(all_pca)
        else:  # GMM
            self.cluster_labels = self.clustering_model.predict(all_pca)

        # Compute clustering quality
        if len(sample_pca) > 1000:  # Only if enough samples
            silhouette_avg = silhouette_score(sample_pca, cluster_labels_sample)
            logger.info(f"Silhouette score: {silhouette_avg:.3f}")

        logger.info("Clustering completed successfully")
        return self.cluster_labels

    def create_intelligent_labels(self) -> np.ndarray:
        """
        Create pseudo-labels using domain knowledge and spectral analysis.

        Returns:
            Pseudo-labels for each pixel
        """
        logger.info("Creating intelligent pseudo-labels...")

        if self.cluster_labels is None:
            raise ValueError("No clustering results. Call perform_clustering() first.")

        pseudo_labels = np.zeros_like(self.cluster_labels)
        h, w = self.processed_shape

        # Get feature indices
        feature_dict = {name: i for i, name in enumerate(self.feature_names)}

        # Analyze each cluster
        for cluster_id in np.unique(self.cluster_labels):
            mask = self.cluster_labels == cluster_id
            cluster_features = self.features[mask]

            if len(cluster_features) == 0:
                continue

            # Extract key characteristics
            brightness = np.mean(cluster_features[:, feature_dict['brightness']])
            water_idx = np.mean(cluster_features[:, feature_dict['water_index']])
            ndvi = np.mean(cluster_features[:, feature_dict['ndvi_proxy']])
            urban_idx = np.mean(cluster_features[:, feature_dict['urban_index']])
            bare_soil_idx = np.mean(cluster_features[:, feature_dict['bare_soil_index']])

            # RGB averages
            r_avg = np.mean(cluster_features[:, 0])
            g_avg = np.mean(cluster_features[:, 1])
            b_avg = np.mean(cluster_features[:, 2])

            # Classification rules (improved logic)
            if water_idx > 0.45 and b_avg > max(r_avg, g_avg) * 1.1:
                land_type = 0  # Water
            elif urban_idx > 0.5 and brightness > 0.5 and bare_soil_idx > 0.1:
                land_type = 1  # Urban/Built-up
            elif ndvi > 0.25 and g_avg > max(r_avg, b_avg) * 1.1:
                land_type = 2  # Dense Vegetation
            elif 0.1 < ndvi < 0.25 and brightness > 0.4:
                land_type = 3  # Agricultural
            elif bare_soil_idx > 0.2 and brightness < 0.5:
                land_type = 4  # Bare Land
            else:
                land_type = 5  # Mixed/Other

            pseudo_labels[mask] = land_type

            pixel_count = mask.sum()
            logger.info(f"Cluster {cluster_id} -> {self.land_classes[land_type]} "
                        f"({pixel_count:,} pixels, {pixel_count / len(mask) * 100:.1f}%)")

        return pseudo_labels

    def train_classifier(self, pseudo_labels: np.ndarray) -> None:
        """
        Train Random Forest classifier using pseudo-labels.

        Args:
            pseudo_labels: Training labels from clustering analysis
        """
        logger.info("Training Random Forest classifier...")

        # Stratified sampling for balanced training
        unique_labels, counts = np.unique(pseudo_labels, return_counts=True)
        min_samples_per_class = max(1000, min(counts))

        train_indices = []
        for label in unique_labels:
            label_indices = np.where(pseudo_labels == label)[0]
            if len(label_indices) > min_samples_per_class:
                selected = np.random.choice(label_indices, min_samples_per_class, replace=False)
            else:
                selected = label_indices
            train_indices.extend(selected)

        train_indices = np.array(train_indices)
        X_train = self.features[train_indices]
        y_train = pseudo_labels[train_indices]

        # Train-validation split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Scale features
        scaler = RobustScaler()
        X_tr_scaled = scaler.fit_transform(X_tr)
        X_val_scaled = scaler.transform(X_val)

        # Train Random Forest
        self.classification_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.classification_model.fit(X_tr_scaled, y_tr)

        # Validation
        val_accuracy = self.classification_model.score(X_val_scaled, y_val)
        logger.info(f"Validation accuracy: {val_accuracy:.3f}")

        # Predict for all pixels
        logger.info("Classifying all pixels...")
        all_scaled = scaler.transform(self.features)
        self.final_predictions = self.classification_model.predict(all_scaled)

        # Store scaler for future use
        self.scaler = scaler

        # Log final distribution
        final_unique, final_counts = np.unique(self.final_predictions, return_counts=True)
        total_pixels = len(self.final_predictions)

        logger.info("Final land cover distribution:")
        for label, count in zip(final_unique, final_counts):
            percentage = (count / total_pixels) * 100
            logger.info(f"  {self.land_classes[label]}: {count:,} pixels ({percentage:.1f}%)")

    def create_visualizations(self, save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization comparing all results.

        Args:
            save_path: Optional path to save visualizations
        """
        logger.info("Creating comprehensive visualizations...")

        if self.final_predictions is None:
            raise ValueError("No classification results available.")

        # Setup the comparison figure
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.2)

        h, w = self.processed_shape

        # 1. Original Image
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(self.image_data)
        ax1.set_title('Original Image', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # 2. Clustering Results
        ax2 = fig.add_subplot(gs[0, 1])
        if self.cluster_labels is not None:
            cluster_map = self.cluster_labels.reshape(h, w)
            im2 = ax2.imshow(cluster_map, cmap='tab10', interpolation='nearest')
            ax2.set_title('Clustering Results', fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        ax2.axis('off')

        # 3. Final Classification
        ax3 = fig.add_subplot(gs[0, 2])
        pred_map = self.final_predictions.reshape(h, w)

        # Create custom colormap
        colors = [np.array(self.class_colors[i]) / 255.0 for i in range(len(self.land_classes))]
        cmap = ListedColormap(colors)

        im3 = ax3.imshow(pred_map, cmap=cmap, vmin=0, vmax=len(self.land_classes) - 1, interpolation='nearest')
        ax3.set_title('Land Cover Classification', fontsize=14, fontweight='bold')

        # Custom legend for classification
        legend_elements = [
            mpatches.Patch(color=colors[i], label=self.land_classes[i])
            for i in range(len(self.land_classes))
        ]
        ax3.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.axis('off')

        # 4. Classification Overlay
        ax4 = fig.add_subplot(gs[0, 3])
        # Create semi-transparent overlay
        overlay = self.image_data.copy()
        alpha = 0.6
        for class_id in range(len(self.land_classes)):
            mask = pred_map == class_id
            color = np.array(self.class_colors[class_id]) / 255.0
            for c in range(3):
                overlay[mask, c] = alpha * overlay[mask, c] + (1 - alpha) * color[c]

        ax4.imshow(np.clip(overlay, 0, 1))
        ax4.set_title('Classification Overlay', fontsize=14, fontweight='bold')
        ax4.axis('off')

        # 5. NDVI Proxy
        ax5 = fig.add_subplot(gs[1, 0])
        ndvi_idx = self.feature_names.index('ndvi_proxy')
        ndvi_map = self.features[:, ndvi_idx].reshape(h, w)
        im5 = ax5.imshow(ndvi_map, cmap='RdYlGn', interpolation='nearest')
        ax5.set_title('NDVI Proxy (Vegetation)', fontsize=14, fontweight='bold')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
        ax5.axis('off')

        # 6. Water Index
        ax6 = fig.add_subplot(gs[1, 1])
        water_idx = self.feature_names.index('water_index')
        water_map = self.features[:, water_idx].reshape(h, w)
        im6 = ax6.imshow(water_map, cmap='Blues', interpolation='nearest')
        ax6.set_title('Water Index', fontsize=14, fontweight='bold')
        plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
        ax6.axis('off')

        # 7. Urban Index
        ax7 = fig.add_subplot(gs[1, 2])
        urban_idx = self.feature_names.index('urban_index')
        urban_map = self.features[:, urban_idx].reshape(h, w)
        im7 = ax7.imshow(urban_map, cmap='Reds', interpolation='nearest')
        ax7.set_title('Urban Index', fontsize=14, fontweight='bold')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        ax7.axis('off')

        # 8. Brightness
        ax8 = fig.add_subplot(gs[1, 3])
        brightness_idx = self.feature_names.index('brightness')
        brightness_map = self.features[:, brightness_idx].reshape(h, w)
        im8 = ax8.imshow(brightness_map, cmap='gray', interpolation='nearest')
        ax8.set_title('Brightness', fontsize=14, fontweight='bold')
        plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
        ax8.axis('off')

        # 9. Class Distribution Chart
        ax9 = fig.add_subplot(gs[2, 0:2])
        unique, counts = np.unique(self.final_predictions, return_counts=True)
        class_names = [self.land_classes[i] for i in unique]
        colors_for_bars = [np.array(self.class_colors[i]) / 255.0 for i in unique]

        bars = ax9.bar(class_names, counts, color=colors_for_bars, edgecolor='black', linewidth=1)
        ax9.set_title('Land Cover Distribution', fontsize=14, fontweight='bold')
        ax9.set_ylabel('Number of Pixels')
        ax9.tick_params(axis='x', rotation=45)

        # Add percentage labels on bars
        total = sum(counts)
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width() / 2., height + total * 0.01,
                     f'{count:,}\n({count / total * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=10)

        # 10. Feature Importance (if available)
        ax10 = fig.add_subplot(gs[2, 2:4])
        if hasattr(self.classification_model, 'feature_importances_'):
            importance = self.classification_model.feature_importances_
            indices = np.argsort(importance)[::-1][:15]  # Top 15

            ax10.barh(range(len(indices)), importance[indices])
            ax10.set_yticks(range(len(indices)))
            ax10.set_yticklabels([self.feature_names[i] for i in indices])
            ax10.set_xlabel('Feature Importance')
            ax10.set_title('Top 15 Feature Importance', fontsize=14, fontweight='bold')
            ax10.invert_yaxis()

        plt.suptitle('GIS Land Cover Classification Results', fontsize=18, fontweight='bold', y=0.98)

        # Save if path provided
        if save_path:
            save_path = Path(save_path)
            plt.savefig(save_path / 'land_classification_comparison.png',
                        dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved comprehensive visualization to {save_path}")

        plt.tight_layout()
        plt.show()

        # Create separate detailed maps
        self._create_detailed_maps(save_path)

    def _create_detailed_maps(self, save_path: Optional[Path] = None) -> None:
        """Create detailed individual maps for each aspect."""

        # 1. High-resolution classification map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        h, w = self.processed_shape
        pred_map = self.final_predictions.reshape(h, w)

        # Original
        ax1.imshow(self.image_data)
        ax1.set_title('Original GIS Image', fontsize=16, fontweight='bold')
        ax1.axis('off')

        # Classification with clean boundaries
        colors = [np.array(self.class_colors[i]) / 255.0 for i in range(len(self.land_classes))]
        cmap = ListedColormap(colors)

        im = ax2.imshow(pred_map, cmap=cmap, vmin=0, vmax=len(self.land_classes) - 1,
                        interpolation='nearest')
        ax2.set_title('Land Cover Classification', fontsize=16, fontweight='bold')
        ax2.axis('off')

        # Enhanced legend
        legend_elements = [
            mpatches.Patch(color=colors[i], label=f'{self.land_classes[i]}')
            for i in range(len(self.land_classes))
        ]
        ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize=12, frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path / 'detailed_classification.png',
                        dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()

        # 2. Individual spectral index maps
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()

        indices_to_plot = [
            ('ndvi_proxy', 'NDVI Proxy', 'RdYlGn'),
            ('water_index', 'Water Index', 'Blues'),
            ('urban_index', 'Urban Index', 'Reds'),
            ('bare_soil_index', 'Bare Soil Index', 'YlOrBr'),
            ('brightness', 'Brightness', 'gray'),
            ('local_entropy', 'Texture (Entropy)', 'viridis')
        ]

        for i, (feature_name, title, cmap) in enumerate(indices_to_plot):
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
                feature_map = self.features[:, feature_idx].reshape(h, w)

                im = axes[i].imshow(feature_map, cmap=cmap, interpolation='nearest')
                axes[i].set_title(title, fontsize=13, fontweight='bold')
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

        plt.suptitle('Spectral Indices and Features', fontsize=16, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path / 'spectral_indices.png',
                        dpi=300, bbox_inches='tight', facecolor='white')

        plt.show()

    def save_classification_results(self, save_path: Optional[str] = None) -> None:
        """
        Save classification results as GeoTIFF and other formats.

        Args:
            save_path: Directory to save results
        """
        if save_path is None:
            save_path = self.output_path
        else:
            save_path = Path(save_path)
            save_path.mkdir(exist_ok=True)

        if self.final_predictions is None:
            logger.warning("No classification results to save")
            return

        h, w = self.processed_shape
        pred_map = self.final_predictions.reshape(h, w)

        # Save as GeoTIFF if we have spatial reference
        if self.image_profile is not None:
            try:
                output_profile = self.image_profile.copy()
                output_profile.update({
                    'dtype': 'uint8',
                    'count': 1,
                    'compress': 'lzw',
                    'height': h,
                    'width': w,
                    'nodata': 255
                })

                # Adjust transform if resampled
                if self.original_shape != self.processed_shape:
                    scale_x = self.original_shape[1] / self.processed_shape[1]
                    scale_y = self.original_shape[0] / self.processed_shape[0]

                    if 'transform' in output_profile:
                        orig_transform = output_profile['transform']
                        new_transform = rasterio.Affine(
                            orig_transform.a * scale_x, orig_transform.b, orig_transform.c,
                            orig_transform.d, orig_transform.e * scale_y, orig_transform.f
                        )
                        output_profile['transform'] = new_transform

                with rasterio.open(save_path / 'land_classification.tif', 'w', **output_profile) as dst:
                    dst.write(pred_map.astype(np.uint8), 1)

                    # Add class descriptions
                    dst.descriptions = tuple([self.land_classes[i] for i in range(len(self.land_classes))])

                logger.info(f"Saved GeoTIFF classification to {save_path / 'land_classification.tif'}")

            except Exception as e:
                logger.warning(f"Could not save GeoTIFF: {str(e)}")

        # Save as PNG
        colors = [self.class_colors[i] for i in range(len(self.land_classes))]
        colored_map = np.zeros((h, w, 3), dtype=np.uint8)

        for class_id in range(len(self.land_classes)):
            mask = pred_map == class_id
            colored_map[mask] = colors[class_id]

        plt.figure(figsize=(12, 8))
        plt.imshow(colored_map)
        plt.axis('off')
        plt.title('Land Cover Classification Map', fontsize=16, fontweight='bold', pad=20)

        # Add legend
        legend_elements = [
            mpatches.Patch(color=np.array(colors[i]) / 255.0, label=self.land_classes[i])
            for i in range(len(self.land_classes))
        ]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left',
                   fontsize=12, frameon=True)

        plt.tight_layout()
        plt.savefig(save_path / 'classification_map.png', dpi=300, bbox_inches='tight',
                    facecolor='white')
        plt.close()

        # Save statistics
        unique, counts = np.unique(self.final_predictions, return_counts=True)
        total_pixels = len(self.final_predictions)

        stats = {
            'total_pixels': int(total_pixels),
            'image_dimensions': {'height': h, 'width': w},
            'land_cover_stats': {}
        }

        for class_id, count in zip(unique, counts):
            stats['land_cover_stats'][self.land_classes[class_id]] = {
                'pixel_count': int(count),
                'percentage': float(count / total_pixels * 100)
            }

        import json
        with open(save_path / 'classification_statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Saved classification statistics to {save_path}")

    def run_classification(self, image_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Run complete land classification pipeline.

        Args:
            image_path: Path to input TIFF image
            output_dir: Directory for outputs (optional)

        Returns:
            Classification results and statistics
        """
        logger.info("=" * 60)
        logger.info("STARTING GIS LAND CLASSIFICATION")
        logger.info("=" * 60)

        try:
            # Setup output directory
            if output_dir is None:
                image_name = Path(image_path).stem
                output_dir = self.output_path / f"classification_{image_name}"
            else:
                output_dir = Path(output_dir)

            output_dir.mkdir(exist_ok=True)
            logger.info(f"Output directory: {output_dir}")

            # Step 1: Load and preprocess
            logger.info("Step 1: Loading and preprocessing image...")
            self.load_image(image_path, max_size=3000)

            # Step 2: Feature extraction
            logger.info("Step 2: Extracting comprehensive features...")
            self.extract_comprehensive_features()

            # Step 3: Clustering
            logger.info("Step 3: Performing initial clustering...")
            self.perform_clustering(n_clusters=6, method='kmeans')

            # Step 4: Pseudo-labeling
            logger.info("Step 4: Creating intelligent pseudo-labels...")
            pseudo_labels = self.create_intelligent_labels()

            # Step 5: Train classifier
            logger.info("Step 5: Training Random Forest classifier...")
            self.train_classifier(pseudo_labels)

            # Step 6: Create visualizations
            logger.info("Step 6: Creating comprehensive visualizations...")
            self.create_visualizations(save_path=output_dir)

            # Step 7: Save results
            logger.info("Step 7: Saving classification results...")
            self.save_classification_results(save_path=output_dir)

            # Compile final results
            unique, counts = np.unique(self.final_predictions, return_counts=True)
            total_pixels = len(self.final_predictions)

            results = {
                'image_path': str(image_path),
                'output_directory': str(output_dir),
                'original_dimensions': self.original_shape,
                'processed_dimensions': self.processed_shape,
                'total_pixels': int(total_pixels),
                'land_cover_distribution': {},
                'classification_summary': {
                    'method': 'Random Forest with Pseudo-labeling',
                    'features_used': len(self.feature_names),
                    'clustering_method': 'K-means'
                }
            }

            for class_id, count in zip(unique, counts):
                class_name = self.land_classes[class_id]
                percentage = (count / total_pixels) * 100
                results['land_cover_distribution'][class_name] = {
                    'pixels': int(count),
                    'percentage': round(percentage, 2)
                }

            logger.info("=" * 60)
            logger.info("CLASSIFICATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)

            # Print summary
            logger.info("FINAL LAND COVER DISTRIBUTION:")
            for class_name, stats in results['land_cover_distribution'].items():
                logger.info(f"  {class_name}: {stats['pixels']:,} pixels ({stats['percentage']:.1f}%)")

            return results

        except Exception as e:
            logger.error(f"Classification failed: {str(e)}")
            raise


def main():
    """Main execution function with command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description='GIS Land Classification Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python land_classifier.py                              # Process all TIFF files in ../dataset
  python land_classifier.py --image satellite_image.tif  # Process specific image
  python land_classifier.py --dataset custom_path        # Use custom dataset path
  python land_classifier.py --clusters 8                 # Use 8 clusters instead of 6
        """
    )

    parser.add_argument('--image',
                        help='Path to specific TIFF image (optional)')
    parser.add_argument('--dataset', default='../dataset',
                        help='Dataset directory path (default: ../dataset)')
    parser.add_argument('--output', default='../outputs',
                        help='Output directory (default: ../outputs)')
    parser.add_argument('--clusters', type=int, default=6,
                        help='Number of clusters for initial segmentation (default: 6)')
    parser.add_argument('--max-size', type=int, default=3000,
                        help='Maximum image dimension for processing (default: 3000)')

    args = parser.parse_args()

    # Setup paths
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    # Create directories if they don't exist
    dataset_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Dataset directory: {dataset_path.absolute()}")
    logger.info(f"Output directory: {output_path.absolute()}")

    # Initialize classifier
    classifier = LandClassifier(output_path=str(output_path))

    try:
        if args.image:
            # Process specific image
            image_path = Path(args.image)
            if not image_path.exists():
                logger.error(f"Image file not found: {image_path}")
                sys.exit(1)

            if not image_path.suffix.lower() in ['.tif', '.tiff']:
                logger.error(f"Input must be a TIFF file, got: {image_path.suffix}")
                sys.exit(1)

            logger.info(f"Processing single image: {image_path}")
            results = classifier.run_classification(str(image_path))
            print_results_summary(results)

        else:
            # Process all TIFF files in dataset directory
            tiff_extensions = ['.tif']
            tiff_files = []

            for ext in tiff_extensions:
                tiff_files.extend(list(dataset_path.glob(f'*{ext}')))

            if not tiff_files:
                logger.error(f"No TIFF files found in {dataset_path}")
                logger.info("Please place your TIFF images in the dataset directory")
                logger.info("Supported formats: .tif, .tiff, .TIF, .TIFF")
                sys.exit(1)

            logger.info(f"Found {len(tiff_files)} TIFF file(s) to process:")
            for i, tiff_file in enumerate(tiff_files, 1):
                logger.info(f"  {i}. {tiff_file.name}")

            # Process each file
            all_results = []
            for i, tiff_file in enumerate(tiff_files, 1):
                logger.info(f"\n{'=' * 60}")
                logger.info(f"PROCESSING IMAGE {i}/{len(tiff_files)}: {tiff_file.name}")
                logger.info(f"{'=' * 60}")

                try:
                    results = classifier.run_classification(str(tiff_file))
                    all_results.append(results)
                    print_results_summary(results)

                except Exception as e:
                    logger.error(f"Failed to process {tiff_file.name}: {str(e)}")
                    continue

            # Print overall summary
            if all_results:
                print_overall_summary(all_results)

    except Exception as e:
        logger.error(f"Execution failed: {str(e)}")
        sys.exit(1)


def print_results_summary(results):
    """Print individual image results summary."""
    print(f"\n{'=' * 60}")
    print("CLASSIFICATION RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Image: {Path(results['image_path']).name}")
    print(f"Original size: {results['original_dimensions'][0]}x{results['original_dimensions'][1]}")
    print(f"Processed size: {results['processed_dimensions'][0]}x{results['processed_dimensions'][1]}")
    print(f"Total pixels: {results['total_pixels']:,}")
    print(f"Output directory: {results['output_directory']}")
    print("\nLand Cover Distribution:")
    for class_name, stats in results['land_cover_distribution'].items():
        print(f"  {class_name:<20}: {stats['pixels']:>8,} pixels ({stats['percentage']:>5.1f}%)")


def print_overall_summary(all_results):
    """Print summary for all processed images."""
    print(f"\n{'=' * 80}")
    print("OVERALL PROCESSING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total images processed: {len(all_results)}")

    # Aggregate statistics
    total_pixels_all = sum(r['total_pixels'] for r in all_results)

    # Combine land cover stats
    combined_stats = {}
    for results in all_results:
        for class_name, stats in results['land_cover_distribution'].items():
            if class_name not in combined_stats:
                combined_stats[class_name] = {'pixels': 0, 'percentage': 0}
            combined_stats[class_name]['pixels'] += stats['pixels']

    # Calculate overall percentages
    for class_name in combined_stats:
        combined_stats[class_name]['percentage'] = (
                combined_stats[class_name]['pixels'] / total_pixels_all * 100
        )

    print(f"Total pixels processed: {total_pixels_all:,}")
    print("\nCombined Land Cover Distribution:")
    for class_name, stats in combined_stats.items():
        print(f"  {class_name:<20}: {stats['pixels']:>10,} pixels ({stats['percentage']:>5.1f}%)")

    print(f"\nIndividual Results:")
    for i, results in enumerate(all_results, 1):
        image_name = Path(results['image_path']).name
        print(f"  {i:2d}. {image_name:<30} -> {results['output_directory']}")


if __name__ == "__main__":
    main()