import rasterio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from skimage.transform import resize
from skimage.filters import gaussian
import matplotlib.pyplot as plt
from segmentation_models_pytorch import Unet
import os

# Config
TRAIN_IMG_DIR = "../data/train/images"
TRAIN_MASK_DIR = "../data/train/masks"
VALID_IMG_DIR = "../data/valid/images"
VALID_MASK_DIR = "../data/valid/masks"
TARGET_DIR = "../target"
PATCH_SIZE = 256
CLASSES = ["urban", "forest", "water", "land"]
NUM_CLASSES = len(CLASSES)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# Preprocessing
def load_tiff(path, target_size=(1000, 1200)):
    """Load and preprocess TIFF image."""
    with rasterio.open(path) as src:
        raw_img = src.read()  # Shape: (bands, height, width)
        raw_img = np.transpose(raw_img, (1, 2, 0))  # Shape: (height, width, bands)

        # Handle different numbers of bands
        if raw_img.shape[2] > 3:
            raw_img = raw_img[:, :, :3]  # Take only first 3 bands (RGB)
        elif raw_img.shape[2] == 1:
            raw_img = np.repeat(raw_img, 3, axis=2)  # Convert grayscale to RGB

    raw_img = resize(raw_img, target_size, anti_aliasing=True)  # Downsample

    # Apply gaussian filter to each channel separately
    for c in range(raw_img.shape[2]):
        raw_img[:, :, c] = gaussian(raw_img[:, :, c], sigma=1)

    # Proper normalization - handle different data types
    if raw_img.dtype == np.uint8:
        raw_img = raw_img.astype(np.float32) / 255.0
    elif raw_img.dtype == np.uint16:
        raw_img = raw_img.astype(np.float32) / 65535.0
    else:
        # For float data, normalize to [0, 1]
        img_min, img_max = raw_img.min(), raw_img.max()
        if img_max > img_min:
            raw_img = (raw_img - img_min) / (img_max - img_min)
        else:
            raw_img = np.zeros_like(raw_img)

    raw_img = np.clip(raw_img, 0, 1)

    return raw_img


def create_patches(img, patch_size):
    """Split image into patches with proper handling."""
    patches = []
    patch_positions = []
    h, w = img.shape[:2]

    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            end_i = min(i + patch_size, h)
            end_j = min(j + patch_size, w)

            patch = img[i:end_i, j:end_j]

            # Pad patch if it"s smaller than patch_size
            if patch.shape[:2] != (patch_size, patch_size):
                padded_patch = np.zeros((patch_size, patch_size, img.shape[2]), dtype=patch.dtype)
                padded_patch[:patch.shape[0], :patch.shape[1]] = patch
                patch = padded_patch

            patches.append(patch)
            patch_positions.append((i, j, end_i, end_j))

    return patches, patch_positions


# DeepGlobe Dataset
class DeepGlobeDataset(Dataset):
    """Dataset for DeepGlobe images and masks."""

    def __init__(self, img_dir, mask_dir, patch_size=256):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.patch_size = patch_size

        # Get all image files
        img_files = [f for f in os.listdir(img_dir)]
        mask_files = [f for f in os.listdir(mask_dir)]

        # Match images with masks
        self.images = []
        for img_file in img_files:
            # Try different mask naming conventions
            base_name = os.path.splitext(img_file)[0]

            mask_name = base_name.replace("_sat", "_mask") + ".png"
            if mask_name in mask_files:
                self.images.append((img_file, mask_name))
                break

        print(f"Found {len(self.images)} image-mask pairs")

        # Dataset Classes
        self.class_map = {
            (0, 255, 255): 0,  # Urban - Cyan
            (255, 255, 0): 1,  # Agriculture - Yellow
            (255, 0, 255): 2,  # Rangeland - Magenta
            (0, 255, 0): 3,  # Forest - Green
            (0, 0, 255): 4,  # Water - Blue
            (255, 255, 255): 5,  # Barren - White
            (0, 0, 0): 6  # Unknown - Black
        }

        # Intended Classes
        self.target_map = {
            0: 0,  # Urban -> Urban
            1: 3,  # Agriculture -> Land
            2: 3,  # Rangeland -> Land
            3: 1,  # Forest -> Forest
            4: 2,  # Water -> Water
            5: 3,  # Barren -> Land
            6: 4  # Unknown -> Ignore
        }

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file, mask_file = self.images[idx]
        img_path = os.path.join(self.img_dir, img_file)
        mask_path = os.path.join(self.mask_dir, mask_file)

        # Load image
        img = load_tiff(img_path, (self.patch_size, self.patch_size))

        try:
            # Load mask using rasterio for consistency
            with rasterio.open(mask_path) as src:
                mask = src.read()

                # Handle different mask formats
                if len(mask.shape) == 3 and mask.shape[0] <= 4:  # (bands, height, width)
                    mask = np.transpose(mask, (1, 2, 0))  # (height, width, bands)
                elif len(mask.shape) == 2:  # (height, width)
                    mask = np.expand_dims(mask, axis=2)  # (height, width, 1)

                # Resize mask to match target size FIRST
                target_shape = (self.patch_size, self.patch_size)
                if mask.shape[:2] != target_shape:
                    if len(mask.shape) == 3:
                        mask = resize(mask, target_shape + (mask.shape[2],),
                                      anti_aliasing=False, preserve_range=True)
                    else:
                        mask = resize(mask, target_shape,
                                      anti_aliasing=False, preserve_range=True)

                # Convert to uint8 if needed
                if mask.dtype != np.uint8:
                    if mask.max() <= 1.0:
                        mask = (mask * 255).astype(np.uint8)
                    else:
                        mask = mask.astype(np.uint8)

                # Ensure 3 channels for RGB processing
                if len(mask.shape) == 2:
                    mask = np.stack([mask] * 3, axis=2)
                elif mask.shape[2] == 1:
                    mask = np.repeat(mask, 3, axis=2)
                elif mask.shape[2] == 4:  # RGBA
                    mask = mask[:, :, :3]  # Drop alpha
                elif mask.shape[2] > 3:
                    mask = mask[:, :, :3]  # Take first 3 channels

                # Convert RGB mask to single-channel with new land class
                mask_single = np.full((self.patch_size, self.patch_size), 4, dtype=np.uint8)  # Default to ignore

                for rgb, cls in self.class_map.items():
                    # Create boolean mask for this class with tolerance
                    rgb_array = np.array(rgb, dtype=np.uint8)
                    mask_pixels = np.all(np.abs(mask.astype(np.int16) - rgb_array.astype(np.int16)) <= 10, axis=2)
                    target_cls = self.target_map[cls]
                    mask_single[mask_pixels] = target_cls

                return torch.from_numpy(img.transpose(2, 0, 1)).float(), torch.from_numpy(mask_single).long()

        except Exception as e:
            print(f"Error loading mask {mask_path}: {str(e)}")
            print(f"Trying alternative loading method...")

            # Fallback: try loading with matplotlib
            try:
                mask = plt.imread(mask_path)

                # Resize if needed
                if mask.shape[:2] != (self.patch_size, self.patch_size):
                    if len(mask.shape) == 3:
                        mask = resize(mask, (self.patch_size, self.patch_size, mask.shape[2]),
                                      anti_aliasing=False, preserve_range=True)
                    else:
                        mask = resize(mask, (self.patch_size, self.patch_size),
                                      anti_aliasing=False, preserve_range=True)

                # Convert to uint8
                if mask.dtype == np.float32 or mask.dtype == np.float64:
                    mask = (mask * 255).astype(np.uint8)

                # Ensure 3 channels
                if len(mask.shape) == 2:
                    mask = np.stack([mask] * 3, axis=2)
                elif mask.shape[2] > 3:
                    mask = mask[:, :, :3]

                # Convert to single channel with land class
                mask_single = np.full((self.patch_size, self.patch_size), 4, dtype=np.uint8)  # Default to ignore

                for rgb, cls in self.class_map.items():
                    rgb_array = np.array(rgb, dtype=np.uint8)
                    mask_pixels = np.all(np.abs(mask.astype(np.int16) - rgb_array.astype(np.int16)) <= 10, axis=2)
                    target_cls = self.target_map[cls]
                    mask_single[mask_pixels] = target_cls

                return torch.from_numpy(img.transpose(2, 0, 1)).float(), torch.from_numpy(mask_single).long()

            except Exception as e2:
                print(f"Fallback loading also failed: {str(e2)}")
                # Return dummy data
                dummy_mask = np.full((self.patch_size, self.patch_size), 4, dtype=np.uint8)  # Default to ignore
                return torch.from_numpy(img.transpose(2, 0, 1)).float(), torch.from_numpy(dummy_mask).long()


# Training
def train_unet(model, train_loader, val_loader, epochs=10, device=DEVICE):
    """Fine-tune U-Net on DeepGlobe."""
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Lower learning rate
    criterion = torch.nn.CrossEntropyLoss(ignore_index=4)  # Ignore "unknown" class

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)
            optimizer.zero_grad()
            pred = model(img)
            loss = criterion(pred, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
        print(f"Training - Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for img, mask in val_loader:
                img, mask = img.to(device), mask.to(device)
                pred = model(img)
                loss = criterion(pred, mask)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0
        print(f"Validation - Epoch {epoch + 1}/{epochs}, Val Loss: {avg_val_loss:.4f}")


# Inference and Visualization
def predict_and_visualize(img, model, output_path, device=DEVICE):
    """Predict and visualize segmentation map for a single image."""
    model.eval()
    patches, positions = create_patches(img, PATCH_SIZE)
    h, w = img.shape[:2]
    seg_map = np.zeros((h, w), dtype=np.uint8)

    for patch, (i, j, end_i, end_j) in zip(patches, positions):
        patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(patch_tensor)
            pred = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

        # Only use the valid portion of the prediction
        valid_h = end_i - i
        valid_w = end_j - j
        seg_map[i:end_i, j:end_j] = pred[:valid_h, :valid_w]

    # Updated color-coded map with land class
    colormap = {
        0: [255, 100, 100],  # Urban = Light Red
        1: [100, 255, 100],  # Forest = Light Green
        2: [100, 100, 255],  # Water = Light Blue
        3: [255, 255, 100],  # Land = Light Yellow
        4: [128, 128, 128]  # Other/Ignore = Gray
    }

    colored_map = np.zeros((h, w, 3), dtype=np.uint8)
    for cls in range(NUM_CLASSES + 1):
        colored_map[seg_map == cls] = colormap.get(cls, [0, 0, 0])

    # Create visualization with original image and segmentation
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.imshow(img)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(colored_map)
    ax2.set_title("Segmentation Map\n(Red=Urban, Green=Forest, Blue=Water, Yellow=Land)")
    ax2.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Segmentation classes distribution:")
    unique, counts = np.unique(seg_map, return_counts=True)
    for cls, count in zip(unique, counts):
        class_name = CLASSES[cls] if cls < len(CLASSES) else "Other"
        percentage = (count / seg_map.size) * 100
        print(f"  {class_name}: {percentage:.1f}%")

    return seg_map


# Evaluation
def evaluate_model(model, val_loader, device=DEVICE):
    """Evaluate with accuracy and IoU."""
    model.eval()
    total_correct = 0
    total_pixels = 0
    iou_per_class = np.zeros(NUM_CLASSES)
    class_counts = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            pred = model(img)
            pred = torch.argmax(pred, dim=1)

            valid_mask = mask < 4  # Ignore "unknown" class (index 4)
            pred_valid = pred[valid_mask]
            mask_valid = mask[valid_mask]

            total_correct += (pred_valid == mask_valid).sum().item()
            total_pixels += mask_valid.numel()

            for cls in range(NUM_CLASSES):
                pred_cls = (pred_valid == cls)
                mask_cls = (mask_valid == cls)
                intersection = (pred_cls & mask_cls).sum().item()
                union = (pred_cls | mask_cls).sum().item()

                if union > 0:
                    iou_per_class[cls] += intersection / union
                    class_counts[cls] += 1

    accuracy = total_correct / total_pixels if total_pixels > 0 else 0

    # Calculate mean IoU only for classes that appeared
    valid_classes = class_counts > 0
    mean_iou = np.mean(iou_per_class[valid_classes] / class_counts[valid_classes]) if np.any(valid_classes) else 0

    return accuracy, mean_iou


# Main
if __name__ == "__main__":
    print("Step: Initializing U-Net model")
    # Initialize model with proper number of classes (4 main classes + 1 ignore class)
    model = Unet(
        encoder_name="resnet34",
        in_channels=3,
        classes=NUM_CLASSES + 1,  # 4 classes + 1 ignore class
        encoder_weights="imagenet"
    )

    print("Step: Setting up DeepGlobe datasets and data loaders")
    # Check if training data exists
    if os.path.exists(TRAIN_IMG_DIR) and os.path.exists(TRAIN_MASK_DIR):
        train_dataset = DeepGlobeDataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)

        if len(train_dataset) > 0:
            train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)  # Smaller batch size

            valid_loader = None
            if os.path.exists(VALID_IMG_DIR) and os.path.exists(VALID_MASK_DIR):
                valid_dataset = DeepGlobeDataset(VALID_IMG_DIR, VALID_MASK_DIR)
                if len(valid_dataset) > 0:
                    valid_loader = DataLoader(valid_dataset, batch_size=4)

            print("Step: Training U-Net model")
            train_unet(model, train_loader, valid_loader, epochs=30)  # More epochs

            if valid_loader:
                print("Step: Evaluating model on validation set")
                accuracy, mean_iou = evaluate_model(model, valid_loader)
                print(f"Validation Accuracy: {accuracy:.4f}, Mean IoU: {mean_iou:.4f}")
        else:
            print("No training data found, using pre-trained model for inference only")
    else:
        print("Training directories not found, using pre-trained model for inference only")

    print("Step: Preparing to process target images")

    if os.path.exists(TARGET_DIR):
        target_images = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith((".tif", ".tiff"))]
        if not target_images:
            print(f"No TIFF images found in {TARGET_DIR}")
        else:
            print(f"Step: Processing {len(target_images)} target images from {TARGET_DIR}")
            os.makedirs("../outputs", exist_ok=True)

            for target_image_file in target_images:
                img_path = os.path.join(TARGET_DIR, target_image_file)
                print(f"Loading {target_image_file}...")
                img = load_tiff(img_path)  # Load without target_size initially
                img = load_tiff(img_path, target_size=(img.shape[0], img.shape[1]))  # Use original image size

                output_name = target_image_file.replace(".tif", ".png").replace(".tiff", ".png")
                output_path = os.path.join("../outputs", f"segmentation_{output_name}")

                print(f"Processing {target_image_file}...")
                model.eval()
                img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float().unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pred = model(img_tensor)
                    seg_map = torch.argmax(pred, dim=1).squeeze().cpu().numpy()

                if seg_map.shape != img.shape[:2]:
                    seg_map = resize(seg_map, img.shape[:2], anti_aliasing=False, preserve_range=True).astype(np.uint8)

                # Updated colormap with land class
                colormap = {
                    0: [255, 100, 100],  # Urban = Light Red
                    1: [100, 255, 100],  # Forest = Light Green
                    2: [100, 100, 255],  # Water = Light Blue
                    3: [255, 255, 100],  # Land = Light Yellow
                    4: [128, 128, 128]  # Other/Ignore = Gray
                }

                h, w = img.shape[:2]
                colored_map = np.zeros((h, w, 3), dtype=np.uint8)
                for cls in range(NUM_CLASSES + 1):
                    colored_map[seg_map == cls] = colormap.get(cls, [0, 0, 0])

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                ax1.imshow(img)
                ax1.set_title("Original Image")
                ax1.axis("off")
                ax2.imshow(colored_map)
                ax2.set_title("Segmentation Map\n(Red=Urban, Green=Forest, Blue=Water, Yellow=Land)")
                ax2.axis("off")
                plt.tight_layout()
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"Generated {output_path}")

                # Print class distribution for this image
                print(f"Class distribution for {target_image_file}:")
                unique, counts = np.unique(seg_map, return_counts=True)
                for cls, count in zip(unique, counts):
                    class_name = CLASSES[cls] if cls < len(CLASSES) else "Other"
                    percentage = (count / seg_map.size) * 100
                    print(f"  {class_name}: {percentage:.1f}%")
                print()
    else:
        print(f"Target directory {TARGET_DIR} not found")
