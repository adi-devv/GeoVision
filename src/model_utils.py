"""
Model training, evaluation, and prediction utilities
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import os
from config import Config
from data_utils import ImageProcessor

config = Config()


class ModelTrainer:
    """Handles model training and evaluation"""

    def __init__(self, model, device=config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=4)  # Ignore unknown class

    def train(self, train_loader, val_loader=None, epochs=config.EPOCHS):
        """Train the model with optional validation"""
        for epoch in range(epochs):
            train_loss = self._train_epoch(train_loader)
            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}')

            if val_loader:
                val_loss = self._validate_epoch(val_loader)
                print(f'Epoch {epoch + 1}/{epochs}, Val Loss: {val_loss:.4f}')

    def _train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for img, mask in train_loader:
            img, mask = img.to(self.device), mask.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(img)
            loss = self.criterion(pred, mask)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(self.device), mask.to(self.device)
                pred = self.model(img)
                loss = self.criterion(pred, mask)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0

    def evaluate(self, val_loader):
        """Evaluate model performance with accuracy and IoU metrics"""
        self.model.eval()
        total_correct = 0
        total_pixels = 0
        iou_per_class = np.zeros(config.NUM_CLASSES)
        class_counts = np.zeros(config.NUM_CLASSES)

        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(self.device), mask.to(self.device)
                pred = torch.argmax(self.model(img), dim=1)

                # Only evaluate on valid pixels (ignore class 4)
                valid_mask = mask < 4
                pred_valid = pred[valid_mask]
                mask_valid = mask[valid_mask]

                total_correct += (pred_valid == mask_valid).sum().item()
                total_pixels += mask_valid.numel()

                # Calculate IoU per class
                for cls in range(config.NUM_CLASSES):
                    pred_cls = (pred_valid == cls)
                    mask_cls = (mask_valid == cls)
                    intersection = (pred_cls & mask_cls).sum().item()
                    union = (pred_cls | mask_cls).sum().item()

                    if union > 0:
                        iou_per_class[cls] += intersection / union
                        class_counts[cls] += 1

        accuracy = total_correct / total_pixels if total_pixels > 0 else 0
        valid_classes = class_counts > 0
        mean_iou = np.mean(iou_per_class[valid_classes] / class_counts[valid_classes]) if np.any(valid_classes) else 0

        return accuracy, mean_iou


class SegmentationPredictor:
    """Handles prediction and visualization of segmentation results"""

    def __init__(self, model, device=config.DEVICE):
        self.model = model.to(device)
        self.device = device

    def predict_image(self, img_path, output_path):
        """Predict segmentation for a single image and save visualization"""
        # Load image maintaining original dimensions
        img = ImageProcessor.load_tiff(img_path)
        original_shape = img.shape[:2]

        # Process image through model
        self.model.eval()
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(img_tensor)
            seg_map = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        # Resize prediction to match original image if needed
        if seg_map.shape != original_shape:
            seg_map = resize(seg_map, original_shape, anti_aliasing=False, preserve_range=True).astype(np.uint8)

        # Create visualization
        self._save_visualization(img, seg_map, output_path)
        self._print_class_distribution(seg_map, os.path.basename(img_path))

        return seg_map

    def predict_with_patches(self, img_path, output_path, patch_size=config.PATCH_SIZE):
        """Predict using patch-based approach for large images"""
        img = ImageProcessor.load_tiff(img_path)
        patches, positions = ImageProcessor.create_patches(img, patch_size)
        h, w = img.shape[:2]
        seg_map = np.zeros((h, w), dtype=np.uint8)

        self.model.eval()
        for patch, (i, j, end_i, end_j) in zip(patches, positions):
            patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred = self.model(patch_tensor)
                pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

            # Only use the valid portion of the prediction
            valid_h = end_i - i
            valid_w = end_j - j
            seg_map[i:end_i, j:end_j] = pred[:valid_h, :valid_w]

        # Create visualization
        self._save_visualization(img, seg_map, output_path)
        self._print_class_distribution(seg_map, os.path.basename(img_path))

        return seg_map

    def _save_visualization(self, img, seg_map, output_path):
        """Create and save side-by-side visualization"""
        colored_map = self._create_colored_segmentation(seg_map)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

        ax1.imshow(img)
        ax1.set_title('Original Image')
        ax1.axis('off')

        ax2.imshow(colored_map)
        ax2.set_title('Segmentation Map\n(Red=Urban, Green=Forest, Blue=Water, Yellow=Land)')
        ax2.axis('off')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f'Generated {output_path}')

    @staticmethod
    def _create_colored_segmentation(seg_map):
        """Convert class map to colored visualization"""
        h, w = seg_map.shape
        colored_map = np.zeros((h, w, 3), dtype=np.uint8)

        for cls in range(config.NUM_CLASSES + 1):
            colored_map[seg_map == cls] = config.COLORMAP.get(cls, [0, 0, 0])

        return colored_map

    @staticmethod
    def _print_class_distribution(seg_map, image_name):
        """Print class distribution statistics"""
        print(f'Class distribution for {image_name}:')
        unique, counts = np.unique(seg_map, return_counts=True)

        for cls, count in zip(unique, counts):
            class_name = config.CLASSES[cls] if cls < len(config.CLASSES) else 'Other'
            percentage = (count / seg_map.size) * 100
            print(f'  {class_name}: {percentage:.1f}%')
        print()