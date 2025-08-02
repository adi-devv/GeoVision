"""
Main execution pipeline for land cover segmentation
"""
import os
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet

from config import Config
from data_utils import DeepGlobeDataset
from model_utils import ModelTrainer, SegmentationPredictor

config = Config()


def setup_data_loaders():
    """Set up training and validation data loaders"""
    train_loader = val_loader = None

    if os.path.exists(config.TRAIN_IMG_DIR) and os.path.exists(config.TRAIN_MASK_DIR):
        train_dataset = DeepGlobeDataset(config.TRAIN_IMG_DIR, config.TRAIN_MASK_DIR)
        if len(train_dataset) > 0:
            train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    if os.path.exists(config.VALID_IMG_DIR) and os.path.exists(config.VALID_MASK_DIR):
        valid_dataset = DeepGlobeDataset(config.VALID_IMG_DIR, config.VALID_MASK_DIR)
        if len(valid_dataset) > 0:
            val_loader = DataLoader(valid_dataset, batch_size=config.BATCH_SIZE)

    return train_loader, val_loader


def train_model(model, train_loader, val_loader):
    """Train the segmentation model"""
    print('Training model')
    trainer = ModelTrainer(model)
    trainer.train(train_loader, val_loader)

    # Evaluate if validation data is available
    if val_loader:
        print('Evaluating model')
        accuracy, mean_iou = trainer.evaluate(val_loader)
        print(f'Validation Accuracy: {accuracy:.4f}, Mean IoU: {mean_iou:.4f}')

    return model


def process_target_images(model):
    """Process all target images for inference"""
    if not os.path.exists(config.TARGET_DIR):
        print(f'Target directory {config.TARGET_DIR} not found')
        return

    target_images = [f for f in os.listdir(config.TARGET_DIR)
                     if f.lower().endswith(('.tif', '.tiff'))]

    if not target_images:
        print(f'No TIFF images found in {config.TARGET_DIR}')
        return

    print(f'Processing {len(target_images)} target images')
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    predictor = SegmentationPredictor(model)

    for img_file in target_images:
        img_path = os.path.join(config.TARGET_DIR, img_file)
        output_name = img_file.replace('.tif', '.png')
        output_path = os.path.join(config.OUTPUT_DIR, f'segmentation_{output_name}')

        print(f'Processing {img_file}...')

        # Use patch-based prediction for large images
        try:
            predictor.predict_image(img_path, output_path)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f'GPU memory error, switching to patch-based prediction...')
                predictor.predict_with_patches(img_path, output_path)
            else:
                raise e


def main():
    """Main execution pipeline"""
    print(f'Using device: {config.DEVICE}')
    print('=' * 50)

    # Initialize model
    print('Initializing U-Net model')
    model = Unet(
        encoder_name='resnet34',
        in_channels=3,
        classes=config.NUM_CLASSES + 1,  # +1 for ignore class
        encoder_weights='imagenet'
    )

    # Setup data
    print('Setting up data loaders')
    train_loader, val_loader = setup_data_loaders()

    # Train model if training data is available
    if train_loader and len(train_loader.dataset) > 0:
        model = train_model(model, train_loader, val_loader)
    else:
        print('No training data found, using pre-trained model for inference')

    # Process target images
    print('Processing target images')
    process_target_images(model)

    print('Pipeline completed successfully!')


if __name__ == '__main__':
    main()
