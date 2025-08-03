# Land Cover Segmentation: Technical Report

**Project:** Satellite Image Classification  
**Model:** U-Net with ResNet34 Encoder  
**Dataset:** DeepGlobe (50 and 100–120 images tested)  
**Best Configuration:** 256×256 patches, batch size 6, 30 epochs  
**Repository:** [github.com/adi-devv/GeoVision](https://github.com/adi-devv/GeoVision)  
**Date:** August 2025

---

## 1. Approach

### 1.1 Architecture
- **Model:** U-Net with ResNet34 encoder (pretrained on ImageNet)
- **Input:** 256×256 RGB patches
- **Output:** 4 classes (Urban, Forest, Water, Land) + 1 ignore class
- **Loss:** CrossEntropyLoss with class weights (ignore_index=4)
- **Optimizer:** Adam (lr=1e-4)

### 1.2 Training
- **Dataset:** DeepGlobe Land Cover Challenge (tested with 50 and 100–120 images)
- **Parameters Tested:**
  - Batch sizes: 4, 6, 8
  - Epochs: 20, 30, 40, 50
  - Patch sizes: 128, 256, 512
- **Best Configuration:** 256×256 patches, batch size 6, 30 epochs
- **Device:** NVIDIA GTX1650 (4GB VRAM)

### 1.3 Preprocessing
- **Class Consolidation:** Mapped 6 DeepGlobe classes to 4 target classes:
  | Original Classes | Target Classes |
  |------------------|----------------|
  | Urban (Cyan)     | Urban         |
  | Agriculture, Rangeland, Barren | Land |
  | Forest (Green)   | Forest        |
  | Water (Blue)     | Water         |
  | Unknown (Black)  | Ignored       |
- **Steps:**
  - Normalization (uint8/uint16/float)
  - Resizing to 256×256 patches
  - Gaussian filtering (σ=1) for noise reduction
  - RGB channel standardization

---

## 2. Challenges

1. **Water-Forest Color Overlap**: Similar RGB values for water and forest led to misclassification. Mitigated with class-weighted loss and color augmentation.
2. **Class Imbalance**: Dominant land class skewed predictions. Used weighted CrossEntropyLoss.
3. **GPU Memory Constraints**: Large images (>1000×1000 pixels) caused memory issues. Applied patch-based processing with overlap.
4. **Small Dataset Size**: 50 images caused overfitting; switching to 100–120 images improved generalization. Used early stopping and regularization.

---

## 3. Results

### 3.1 Performance
- **Best Configuration:** 256×256 patches, batch size 6, 30 epochs (100–120 images)
- **Metrics:**
  - Validation accuracy: ~85% (100–120 images)
  - Lower accuracy (~78%) with 50 images due to overfitting
  - Stable loss convergence, no overfitting with larger dataset
  - Mean IoU improved with class weighting
- **Training Time:** ~1 hour on NVIDIA GTX1650 (4GB)

### 3.2 Key Observations
- 256×256 patches balanced detail and context.
- Batch size 6 optimized GPU memory.
- 30 epochs sufficient; 50 epochs showed minimal gains.
- 512×512 patches increased computation without accuracy boost.
- Larger dataset (100–120 images) reduced water-forest confusion vs. 50 images.

### 3.3 Visualization
- Color-coded maps (Urban: Red, Forest: Green, Water: Blue, Land: Yellow)
- Side-by-side original vs. predicted images
- High-resolution outputs (300 DPI)

---

## 4. Assumptions

- **Image Properties:**
  - RGB input; single-band images replicated to RGB
  - Images ≥256×256 pixels optimal
  - Orthorectified with consistent resolution
- **Data Consistency:** Uniform geographic projection
- **Computational Resources:**
  - Minimum 4GB VRAM for training
  - 8GB+ RAM for large images
- **Spectral Input:** RGB only; first three bands used if multispectral

---

## 5. Recommendations
- **Enhance Data Augmentation**: Use rotation and color jittering to address water-forest overlap.
- **Multi-Scale Training**: Combine patch sizes (128, 256, 512) for better context.
- **Expand Dataset**: Collect ~500+ images for improved generalization.
- **Incorporate Spectral Indices**: Use NDVI/NDWI if multispectral data available.

---

## 6. References
1. **DeepGlobe Dataset:**
'''
@InProceedings{DeepGlobe18,
  author = {Demir, Ilke and others},
  title = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
  booktitle = {CVPR Workshops},
  year = {2018}
}
'''
3. U-Net: Ronneberger et al. (2015). Convolutional Networks for Biomedical Image Segmentation.
4. ResNet: He et al. (2016). Deep Residual Learning for Image Recognition.


## 7. Conclusion
The U-Net with ResNet34 encoder achieved ~85% validation accuracy with 256×256 patches, batch size 6, and 30 epochs on 100–120 images. Switching from 50 to 100–120 images reduced overfitting and improved performance. Challenges like water-forest color overlap were mitigated with weighted loss. The pipeline is scalable for environmental monitoring.
Repository Structure:

