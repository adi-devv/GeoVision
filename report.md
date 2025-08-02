# GeoVision: Technical Report

**Project:** Satellite Image Land Cover Classification  
**Repository:** [https://github.com/adi-devv/GeoVision](https://github.com/adi-devv/GeoVision)  
**Model:** U-Net with ResNet34 Encoder  
**Dataset:** DeepGlobe Land Cover Challenge  
**Date:** August 2025

---

## 1. Introduction and Approach

### 1.1 Problem Statement
Land cover classification from satellite imagery is crucial for environmental monitoring, urban planning, and climate research. This project implements an automated semantic segmentation pipeline to classify satellite images into four primary land cover types: urban areas, forests, water bodies, and general land.

### 1.2 Technical Approach
We employed a U-Net architecture with a pre-trained ResNet34 encoder for pixel-wise land cover classification. The approach consists of:

- **Architecture:** U-Net with ResNet34 backbone pre-trained on ImageNet
- **Class Mapping:** Simplified 6-class DeepGlobe labels into 4 target classes
- **Training Strategy:** Transfer learning with fine-tuning on satellite imagery
- **Inference:** Patch-based processing for large satellite images

### 1.3 Dataset and Preprocessing
**Dataset Source:** DeepGlobe 2018 Land Cover Classification Challenge¹

The DeepGlobe dataset provides high-resolution satellite imagery with pixel-level annotations for land cover classification. Original dataset contains 6 classes which we consolidated into 4 meaningful categories:

| Original Classes | Target Classes |
|-----------------|----------------|
| Urban (Cyan) | Urban |
| Agriculture (Yellow) | Land |
| Rangeland (Magenta) | Land |
| Forest (Green) | Forest |
| Water (Blue) | Water |
| Barren (White) | Land |
| Unknown (Black) | Ignored |

**Preprocessing Pipeline:**
1. **Normalization:** Data-type aware normalization (uint8/uint16/float)
2. **Resizing:** Standardized to 256×256 patches for training
3. **Gaussian Filtering:** Applied with σ=1 for noise reduction
4. **Channel Standardization:** Ensured RGB format (3 channels)

---

## 2. Implementation Details and Challenges

### 2.1 Model Architecture
```
U-Net Configuration:
- Encoder: ResNet34 (ImageNet pre-trained)
- Input Channels: 3 (RGB)
- Output Classes: 5 (4 target + 1 ignore)
- Loss Function: CrossEntropyLoss (ignore_index=4)
- Optimizer: Adam (lr=1e-4)
```

### 2.2 Training Configuration
- **Training Images:** ~100 satellite images
- **Epochs:** 30
- **Batch Size:** 4 (GPU memory constraints)
- **Patch Size:** 256×256 pixels
- **Device:** CUDA-enabled GPU

### 2.3 Key Technical Challenges

#### 2.3.1 Memory Management
**Challenge:** Large satellite images (>1000×1000 pixels) caused GPU memory overflow.  
**Solution:** Implemented patch-based inference with automatic fallback for memory errors.

#### 2.3.2 Class Imbalance
**Challenge:** Uneven distribution of land cover types in training data.  
**Solution:** Used CrossEntropyLoss with ignore_index for unknown regions, allowing the model to focus on confident predictions.

#### 2.3.3 Multi-format Data Loading
**Challenge:** Inconsistent TIFF formats and mask encoding across samples.  
**Solution:** Robust loading pipeline with rasterio primary method and matplotlib fallback, plus tolerance-based RGB matching (±10 pixel values).

#### 2.3.4 Edge Case Handling
**Challenge:** Images smaller than patch size and irregular dimensions.  
**Solution:** Padding strategy to ensure consistent patch dimensions while preserving spatial relationships.

### 2.4 Data Pipeline Robustness
The implementation includes comprehensive error handling:
- Multiple mask loading methods (rasterio → matplotlib → dummy fallback)
- Automatic data type detection and conversion
- Graceful handling of corrupted or missing files

---

## 3. Results and Analysis

### 3.1 Model Performance
Training was conducted over 30 epochs with the following observations:

**Training Convergence:**
- Stable loss reduction over epochs
- No significant overfitting observed
- Validation loss tracked training loss closely

**Evaluation Metrics:**
- **Accuracy:** Pixel-wise classification accuracy on validation set
- **Mean IoU:** Intersection over Union averaged across classes
- **Class Distribution Analysis:** Per-image land cover statistics

### 3.2 Qualitative Results
The model successfully generates interpretable segmentation maps with:
- **Urban Areas:** Clearly identified as red regions
- **Forest Coverage:** Green regions showing vegetation
- **Water Bodies:** Blue regions for lakes, rivers, and coastal areas
- **General Land:** Yellow regions for agriculture, barren land, and mixed use

### 3.3 Output Visualization
Each prediction generates:
1. Side-by-side comparison (original vs. segmentation)
2. Color-coded classification map
3. Quantitative class distribution statistics
4. High-resolution output (300 DPI) for analysis

### 3.4 Limitations and Future Work

**Current Limitations:**
- Limited training data (~100 images) may affect generalization
- Fixed patch size may not capture large-scale spatial patterns
- No temporal analysis for land cover change detection

**Potential Improvements:**
- Data augmentation strategies (rotation, color adjustment)
- Multi-scale training with variable patch sizes
- Ensemble methods for improved accuracy
- Integration of spectral indices (NDVI, NDWI) if multispectral data available

---

## 4. Assumptions and Technical Specifications

### 4.1 Image Assumptions
**Resolution Requirements:**
- Input images processed at native resolution, then resized as needed
- Optimal performance on images ≥256×256 pixels
- Patch-based processing handles images up to several thousand pixels

**Spectral Assumptions:**
- **Band Requirements:** RGB (3-band) input expected
- **Automatic Conversion:** Single-band images converted to RGB via replication
- **Band Selection:** For >3 bands, first 3 channels used (typically RGB)
- **Data Types:** Supports uint8, uint16, and float32 formats

**Spatial Assumptions:**
- Images assumed to be orthorectified (geometrically corrected)
- Consistent spatial resolution within training/inference datasets
- Geographic projection assumed consistent (no coordinate system handling)

### 4.2 Computational Requirements
- **GPU Memory:** Minimum 4GB VRAM for training
- **CPU Memory:** 8GB+ RAM recommended for large image processing
- **Storage:** Sufficient space for patch generation and output visualization

### 4.3 Software Dependencies
```
Core Dependencies:
- PyTorch ≥1.9.0
- segmentation_models_pytorch
- rasterio ≥1.2.0
- scikit-image ≥0.18.0
- matplotlib ≥3.3.0
- numpy ≥1.19.0
```

---

## 5. References and Citations

¹ **DeepGlobe Dataset:**
```bibtex
@InProceedings{DeepGlobe18,
 author = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
 title = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
 booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 month = {June},
 year = {2018}
}
```

**Additional References:**
- **U-Net Architecture:** Ronneberger, O., Fischer, P., & Brox, T. (2015). U-net: Convolutional networks for biomedical image segmentation.
- **ResNet Backbone:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
- **Segmentation Models PyTorch:** Yakubovskiy, P. (2020). Segmentation Models Pytorch. https://github.com/qubvel/segmentation_models.pytorch

---

## 6. Conclusion

This project successfully demonstrates automated land cover classification using deep learning on satellite imagery. The U-Net architecture with ResNet34 encoder provides robust feature extraction and accurate pixel-wise classification. The modular implementation ensures maintainability and extensibility for future enhancements.

The approach effectively handles real-world challenges including memory constraints, data format variations, and class imbalance. Results show promise for operational deployment in environmental monitoring applications.

**Key Achievements:**
- ✅ Robust satellite image processing pipeline
- ✅ Memory-efficient inference for large images  
- ✅ Comprehensive error handling and fallback mechanisms
- ✅ Interpretable visualization outputs
- ✅ Modular, maintainable codebase

**Project Repository Structure:**
```
GeoVision/
├── src/
│   ├── config.py
│   ├── data_utils.py  
│   ├── model_utils.py
│   └── main.py
├── data/
│   ├── train/
│   │   ├── images/    # Training satellite images (.jpg)
│   │   └── masks/     # Training masks (.png)
│   └── valid/
│       ├── images/    # Validation satellite images (.jpg)  
│       └── masks/     # Validation masks (.png)
├── target/            # Images to segment (.tif/.tiff)
├── outputs/           # Generated segmentation results
├── requirements.txt
└── README.md
```

