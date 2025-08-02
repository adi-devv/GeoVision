# Land Cover Segmentation Pipeline

A deep learning pipeline for satellite image land cover classification using U-Net with ResNet34 encoder, trained on the DeepGlobe dataset.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended, 4GB+ VRAM)
- 8GB+ RAM for large image processing

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/adi-devv/GeoVision
cd land-cover-segmentation
```

2. **Install dependencies:**
```bash
pip install torch torchvision torchaudio
pip install segmentation-models-pytorch
pip install rasterio scikit-image matplotlib numpy
```

Or use requirements.txt:
```bash
pip install -r requirements.txt
```

### Directory Structure Setup

Create the following directory structure:
```
project/
├── config.py
├── data_utils.py  
├── model_utils.py
├── main.py
├── data/
│   ├── train/
│   │   ├── images/    # Training satellite images (.tif)
│   │   └── masks/     # Training masks (.png)
│   └── valid/
│       ├── images/    # Validation satellite images (.tif)  
│       └── masks/     # Validation masks (.png)
├── target/            # Images to segment (.tif/.tiff)
└── outputs/           # Generated segmentation results
```

## 📊 Dataset Requirements

### DeepGlobe Dataset
Download from: [DeepGlobe 2018 Challenge](http://deepglobe.org/)

**Training Data:**
- Place satellite images in `data/train/images/`
- Place corresponding masks in `data/train/masks/`
- Image naming: `*_sat.jpg` 
- Mask naming: `*_mask.png`

**Expected Data Format:**
- **Images:** TIFF format, RGB or multispectral
- **Masks:** PNG format with RGB color encoding
- **Resolution:** Any resolution (auto-resized to 256×256 for training)

### Image Assumptions
- **Bands:** 1-3 channels supported (auto-converted to RGB)
- **Data Types:** uint8, uint16, float32
- **Formats:** TIFF for satellite images, PNG for masks
- **Size:** Minimum 256×256 pixels recommended

## 🎯 Usage

### Basic Usage
```bash
python main.py
```

This will:
1. Load and preprocess training data
2. Train U-Net model (if training data available)
3. Process all images in `target/` directory
4. Save segmentation results to `outputs/`

### Configuration

Edit `config.py` to modify:
```python
class Config:
    # Directories
    TRAIN_IMG_DIR = '../data/train/images'
    TARGET_DIR = '../target'
    OUTPUT_DIR = '../outputs'
    
    # Training parameters  
    EPOCHS = 30
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-4
    PATCH_SIZE = 256
```

### Custom Usage

**Training only:**
```python
from main import setup_data_loaders, train_model
from segmentation_models_pytorch import Unet

model = Unet(encoder_name='resnet34', in_channels=3, classes=5)
train_loader, val_loader = setup_data_loaders()
trained_model = train_model(model, train_loader, val_loader)
```

**Inference only:**
```python
from model_utils import SegmentationPredictor
from segmentation_models_pytorch import Unet

model = Unet(encoder_name='resnet34', in_channels=3, classes=5)
predictor = SegmentationPredictor(model)
seg_map = predictor.predict_image('path/to/image.tif', 'output.png')
```

## 📈 Output

### Segmentation Results
Each processed image generates:
- **Visualization:** Side-by-side original and segmentation images
- **Color Coding:**
  - 🔴 Red: Urban areas
  - 🟢 Green: Forest/vegetation  
  - 🔵 Blue: Water bodies
  - 🟡 Yellow: General land (agriculture, barren)
- **Statistics:** Class distribution percentages

### Example Output
```
Class distribution for satellite_image.tif:
  Urban: 15.2%
  Forest: 45.8% 
  Water: 8.1%
  Land: 30.9%
```

## 🔧 Troubleshooting

### Common Issues

**GPU Out of Memory:**
- Reduce `BATCH_SIZE` in config.py
- The system automatically falls back to patch-based processing for large images

**Missing Dependencies:**
```bash
# GDAL issues (for rasterio)
conda install -c conda-forge rasterio

# PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Data Loading Errors:**
- Ensure image-mask naming convention: `image_sat.png` ↔ `image_mask.png`
- Check file permissions and formats
- Verify directory structure matches expected layout

**Performance Optimization:**
- Use SSD storage for faster I/O
- Increase `num_workers` in DataLoader for multi-core systems
- Monitor GPU memory usage with `nvidia-smi`

## 📋 Model Details

### Architecture
- **Base Model:** U-Net with ResNet34 encoder
- **Input:** RGB satellite images (3 channels)
- **Output:** 5-class segmentation (4 land cover + 1 ignore)
- **Loss:** CrossEntropyLoss with ignore_index=4

### Class Mapping
```python
Original → Target Classes:
Urban (Cyan) → Urban
Agriculture (Yellow) → Land  
Rangeland (Magenta) → Land
Forest (Green) → Forest
Water (Blue) → Water
Barren (White) → Land
Unknown (Black) → Ignored
```

## 📚 References

**Dataset:**
```bibtex
@InProceedings{DeepGlobe18,
 author = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
 title = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
 booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 month = {June},
 year = {2018}
}
```

**Model Architecture:**
- [Segmentation Models PyTorch](https://github.com/qubvel/segmentation_models.pytorch)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/enhancement`)
3. Commit changes (`git commit -am 'Add enhancement'`)
4. Push to branch (`git push origin feature/enhancement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check existing issues in the repository
2. Review troubleshooting section above
3. Create new issue with:
   - System specifications
   - Error messages
   - Sample data (if possible)

---

**Last Updated:** August 2025  
**Python Version:** 3.8+  

**PyTorch Version:** 1.9.0+


