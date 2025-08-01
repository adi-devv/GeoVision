import logging
import numpy as np
import rasterio
from pathlib import Path
from typing import Optional, Tuple
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class LandClassifier:
    """Simplified land cover classifier for GIS TIFF images with 4 classes: Urban, Forest, Water, Bare Land."""

    def __init__(self, output_path: str = "gem_outputs"):
        """Initialize the classifier."""
        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True)
        self.image_data = None
        self.image_profile = None
        self.original_shape = None
        self.processed_shape = None
        self.features = None
        self.feature_names = []
        self.scaler = RobustScaler()
        self.clustering_model = None
        self.classification_model = None
        self.cluster_labels = None
        self.final_predictions = None
        self.land_classes = {
            0: "Water",
            1: "Urban",
            2: "Forest",
            3: "Bare Land"
        }
        self.class_colors = {
            0: [30, 144, 255],  # Water - Blue
            1: [255, 255, 255],  # Urban - Red
            2: [34, 139, 34],  # Forest - Green
            3: [160, 82, 45]  # Bare Land - Brown
        }

    def load_image(self, image_path: str, max_size: int = 2000) -> np.ndarray:
        """Load and preprocess TIFF image."""
        logger.info(f"Loading image: {image_path}")
        with rasterio.open(image_path) as src:
            self.image_profile = src.profile.copy()
            self.original_shape = (src.height, src.width)
            h, w = src.height, src.width
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_h, new_w = int(h * scale), int(w * scale)
                image = src.read(out_shape=(src.count, new_h, new_w))
            else:
                image = src.read()
            self.processed_shape = image.shape[1:]
            if image.shape[0] == 1:
                image = np.repeat(image, 3, axis=0)
            elif image.shape[0] > 3:
                image = image[:3]
            image = np.transpose(image, (1, 2, 0)).astype(np.float32) / 255.0
            self.image_data = np.nan_to_num(image, nan=np.nanmean(image))
        logger.info(f"Processed shape: {self.image_data.shape}")
        return self.image_data

    def extract_features(self) -> np.ndarray:
        """Extract 6 features for classification."""
        logger.info("Extracting features...")
        if self.image_data is None:
            raise ValueError("No image loaded.")
        h, w, c = self.image_data.shape
        features = []
        self.feature_names = []

        # Raw spectral values (3 features)
        pixel_values = self.image_data.reshape(-1, c)
        features.append(pixel_values)
        self.feature_names.extend(['band_1', 'band_2', 'band_3'])

        # Spectral indices (3 features)
        R, G, B = self.image_data[:, :, 0], self.image_data[:, :, 1], self.image_data[:, :, 2]
        ndvi = (G - R) / (G + R + 1e-8)  # Forest detection
        water_idx = B / (R + G + B + 1e-8)  # Water detection
        brightness = (R + G + B) / 3  # Urban vs. Bare Land
        features.extend([ndvi.reshape(-1, 1), water_idx.reshape(-1, 1), brightness.reshape(-1, 1)])
        self.feature_names.extend(['ndvi', 'water_index', 'brightness'])

        self.features = np.hstack(features)
        self.features = np.nan_to_num(self.features, nan=np.nanmean(self.features, axis=0))
        logger.info(f"Extracted {self.features.shape[1]} features")
        return self.features

    def perform_clustering(self, n_clusters: int = 4) -> np.ndarray:
        """Perform KMeans clustering."""
        logger.info(f"Clustering with {n_clusters} clusters...")
        if self.features is None:
            raise ValueError("No features available.")
        n_samples = min(50000, self.features.shape[0])
        sample_indices = np.random.choice(self.features.shape[0], n_samples, replace=False)
        sample_features = self.features[sample_indices]
        sample_scaled = self.scaler.fit_transform(sample_features)
        pca = PCA(n_components=5, random_state=42)
        sample_pca = pca.fit_transform(sample_scaled)
        self.clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.clustering_model.fit(sample_pca)
        all_scaled = self.scaler.transform(self.features)
        all_pca = pca.transform(all_scaled)
        self.cluster_labels = self.clustering_model.predict(all_pca)
        return self.cluster_labels

    def create_pseudo_labels(self) -> np.ndarray:
        """Assign pseudo-labels based on spectral indices with refined rules."""
        logger.info("Creating pseudo-labels...")
        if self.cluster_labels is None:
            raise ValueError("No clustering results.")
        pseudo_labels = np.zeros_like(self.cluster_labels)
        feature_dict = {name: i for i, name in enumerate(self.feature_names)}
        for cluster_id in np.unique(self.cluster_labels):
            mask = self.cluster_labels == cluster_id
            cluster_features = self.features[mask]
            ndvi = np.mean(cluster_features[:, feature_dict['ndvi']])
            water_idx = np.mean(cluster_features[:, feature_dict['water_index']])
            brightness = np.mean(cluster_features[:, feature_dict['brightness']])

            if water_idx > 0.4:
                land_type = 0  # Water
            elif brightness > 0.6:  # Increased threshold for Urban
                land_type = 1  # Urban
            elif ndvi > 0.1:  # Forest
                land_type = 2  # Forest
            elif brightness < 0.3 or (0.3 <= brightness <= 0.6 and ndvi <= 0.1):  # Bare Land (including agri fields)
                land_type = 3  # Bare Land
            else:
                land_type = 3  # Default to Bare Land
            pseudo_labels[mask] = land_type
        return pseudo_labels

    def train_classifier(self, pseudo_labels: np.ndarray) -> Tuple[float, float]:
        """Train Random Forest classifier and compute IoU."""
        logger.info("Training classifier...")
        train_indices = []
        unique_labels, counts = np.unique(pseudo_labels, return_counts=True)
        min_samples = max(500, min(counts))
        for label in unique_labels:
            label_indices = np.where(pseudo_labels == label)[0]
            selected = np.random.choice(label_indices, min(min_samples, len(label_indices)), replace=False)
            train_indices.extend(selected)
        X_train = self.features[train_indices]
        y_train = pseudo_labels[train_indices]
        X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        X_val_scaled = self.scaler.transform(X_val)
        self.classification_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        self.classification_model.fit(X_tr_scaled, y_tr)
        val_accuracy = self.classification_model.score(X_val_scaled, y_val)
        logger.info(f"Validation accuracy: {val_accuracy:.3f}")

        # Compute IoU on validation set
        y_val_pred = self.classification_model.predict(X_val_scaled)
        iou_per_class = {}
        for class_id in unique_labels:
            true_mask = (y_val == class_id)
            pred_mask = (y_val_pred == class_id)
            intersection = np.sum(true_mask & pred_mask)
            union = np.sum(true_mask | pred_mask)
            iou = intersection / union if union > 0 else 0.0
            iou_per_class[self.land_classes[class_id]] = iou
        mean_iou = np.mean(list(iou_per_class.values())) if iou_per_class else 0.0
        logger.info("IoU per class: " + ", ".join(f"{k}: {v:.3f}" for k, v in iou_per_class.items()))
        logger.info(f"Mean IoU: {mean_iou:.3f}")

        all_scaled = self.scaler.transform(self.features)
        self.final_predictions = self.classification_model.predict(all_scaled)
        return val_accuracy, mean_iou

    def create_visualization(self, save_path: Optional[str] = None) -> None:
        """Create classification map and distribution chart."""
        logger.info("Creating visualization...")
        if self.final_predictions is None:
            raise ValueError("No classification results.")
        h, w = self.processed_shape
        pred_map = self.final_predictions.reshape(h, w)
        colors = [np.array(self.class_colors[i]) / 255.0 for i in range(len(self.land_classes))]
        cmap = ListedColormap(colors)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        ax1.imshow(self.image_data)
        ax1.set_title('Original Image', fontsize=12)
        ax1.axis('off')

        ax2.imshow(pred_map, cmap=cmap, vmin=0, vmax=len(self.land_classes) - 1)
        ax2.set_title('Land Cover Classification', fontsize=12)
        ax2.axis('off')
        legend_elements = [mpatches.Patch(color=colors[i], label=self.land_classes[i]) for i in
                           range(len(self.land_classes))]
        ax2.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

        unique, counts = np.unique(self.final_predictions, return_counts=True)
        class_names = [self.land_classes[i] for i in unique]
        colors_bars = [colors[i] for i in unique]
        bars = ax3.bar(class_names, counts, color=colors_bars, edgecolor='black')
        ax3.set_title('Class Distribution', fontsize=12)
        ax3.set_ylabel('Pixel Count')
        ax3.tick_params(axis='x', rotation=45)
        total = sum(counts)
        for bar, count in zip(bars, counts):
            ax3.text(bar.get_x() + bar.get_width() / 2, count, f'{count:,}\n({count / total * 100:.1f}%)',
                     ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        if save_path:
            plt.savefig(Path(save_path) / 'classification_map.png', dpi=200, bbox_inches='tight')
        plt.show()

    def save_results(self, save_path: Optional[str] = None) -> None:
        """Save classification results as PNG and statistics (no GeoTIFF)."""
        if save_path is None:
            save_path = self.output_path
        else:
            save_path = Path(save_path)
        save_path.mkdir(exist_ok=True)
        if self.final_predictions is None:
            logger.warning("No results to save")
            return
        h, w = self.processed_shape
        pred_map = self.final_predictions.reshape(h, w)

        # Save PNG
        colored_map = np.zeros((h, w, 3), dtype=np.uint8)
        for class_id in range(len(self.land_classes)):
            mask = pred_map == class_id
            colored_map[mask] = self.class_colors[class_id]
        plt.figure(figsize=(10, 8))
        plt.imshow(colored_map)
        plt.axis('off')
        plt.title('Land Cover Classification')
        legend_elements = [mpatches.Patch(color=np.array(self.class_colors[i]) / 255.0, label=self.land_classes[i])
                           for i in range(len(self.land_classes))]
        plt.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.savefig(save_path / 'classification_map.png', dpi=200, bbox_inches='tight')
        plt.close()

        # Save statistics
        unique, counts = np.unique(self.final_predictions, return_counts=True)
        total_pixels = len(self.final_predictions)
        stats = {
            'total_pixels': int(total_pixels),
            'dimensions': {'height': h, 'width': w},
            'land_cover_stats': {self.land_classes[i]: {'pixels': int(c), 'percentage': float(c / total_pixels * 100)}
                                 for i, c in zip(unique, counts)}
        }
        with open(save_path / 'statistics.json', 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to {save_path}")

    def run_classification(self, image_path: str, output_dir: Optional[str] = None) -> dict:
        """Run the complete classification pipeline."""
        logger.info("Starting land classification...")
        if output_dir is None:
            image_name = Path(image_path).stem
            output_dir = self.output_path / f"classification_{image_name}"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        self.load_image(image_path)
        self.extract_features()
        self.perform_clustering(n_clusters=4)
        pseudo_labels = self.create_pseudo_labels()
        val_accuracy, mean_iou = self.train_classifier(pseudo_labels)
        self.create_visualization(save_path=output_dir)
        self.save_results(save_path=output_dir)

        unique, counts = np.unique(self.final_predictions, return_counts=True)
        total_pixels = len(self.final_predictions)
        results = {
            'image_path': str(image_path),
            'output_directory': str(output_dir),
            'total_pixels': int(total_pixels),
            'land_cover_distribution': {
                self.land_classes[i]: {'pixels': int(c), 'percentage': round(c / total_pixels * 100, 2)}
                for i, c in zip(unique, counts)},
            'validation_accuracy': val_accuracy,
            'mean_iou': mean_iou
        }
        logger.info("Classification completed!")
        for class_name, stats in results['land_cover_distribution'].items():
            logger.info(f"{class_name}: {stats['pixels']:,} pixels ({stats['percentage']:.1f}%)")
        logger.info(f"Validation Accuracy: {val_accuracy:.3f}, Mean IoU: {mean_iou:.3f}")
        return results


def main():
    """Main execution function to process all TIFF images in ../dataset and save to ../gem_outputs."""
    dataset_path = Path("../dataset")
    output_path = Path("../gem_outputs")
    output_path.mkdir(exist_ok=True)  # Create ../gem_outputs if it doesn't exist
    logger.info(f"Dataset directory: {dataset_path.absolute()}")
    logger.info(f"Output directory: {output_path.absolute()}")

    classifier = LandClassifier(output_path=str(output_path))

    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return

    tiff_extensions = ['.tif', '.tiff']
    tiff_files = [f for ext in tiff_extensions for f in dataset_path.glob(f'*{ext}')]

    if not tiff_files:
        logger.error(f"No TIFF files found in {dataset_path}")
        logger.info("Please place your TIFF images in the dataset directory")
        logger.info("Supported formats: .tif, .tiff, .TIF, .TIFF")
        return

    logger.info(f"Found {len(tiff_files)} TIFF file(s) to process:")
    for i, tiff_file in enumerate(tiff_files, 1):
        logger.info(f" {i}. {tiff_file.name}")

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

    if all_results:
        print_overall_summary(all_results)


def print_results_summary(results):
    """Print individual image results summary."""
    print(f"\n{'=' * 60}")
    print("CLASSIFICATION RESULTS SUMMARY")
    print(f"{'=' * 60}")
    print(f"Image: {Path(results['image_path']).name}")
    print(f"Total pixels: {results['total_pixels']:,}")
    print(f"Output directory: {results['output_directory']}")
    print("\nLand Cover Distribution:")
    for class_name, stats in results['land_cover_distribution'].items():
        print(f" {class_name:<10}: {stats['pixels']:>8,} pixels ({stats['percentage']:>5.1f}%)")
    print(f"Validation Accuracy: {results['validation_accuracy']:.3f}")
    print(f"Mean IoU: {results['mean_iou']:.3f}")


def print_overall_summary(all_results):
    """Print summary for all processed images."""
    print(f"\n{'=' * 80}")
    print("OVERALL PROCESSING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total images processed: {len(all_results)}")
    total_pixels_all = sum(r['total_pixels'] for r in all_results)
    combined_stats = {}
    for results in all_results:
        for class_name, stats in results['land_cover_distribution'].items():
            if class_name not in combined_stats:
                combined_stats[class_name] = {'pixels': 0, 'percentage': 0}
            combined_stats[class_name]['pixels'] += stats['pixels']
    for class_name in combined_stats:
        combined_stats[class_name]['percentage'] = combined_stats[class_name]['pixels'] / total_pixels_all * 100
    print(f"Total pixels processed: {total_pixels_all:,}")
    print("\nCombined Land Cover Distribution:")
    for class_name, stats in combined_stats.items():
        print(f" {class_name:<10}: {stats['pixels']:>10,} pixels ({stats['percentage']:>5.1f}%)")
    print(f"\nIndividual Results:")
    for i, results in enumerate(all_results, 1):
        image_name = Path(results['image_path']).name
        print(f" {i:2d}. {image_name:<30} -> {results['output_directory']}")


if __name__ == "__main__":
    main()
