# Land Cover Segmentation: Technical Report

**Project:** Satellite Image Classification  
**Model:** U-Net with ResNet34 Encoder  
**Dataset:** DeepGlobe (100-120 images)  
**Best Config:** 256px patches, batch=6, epochs=30  

---

## 1. Methodology

### Architecture
- U-Net with ResNet34 encoder (ImageNet pretrained)
- Input: 256×256 RGB patches
- Output: 4 classes (Urban/Forest/Water/Land)

### Training
- **Optimal Parameters:**
  - Batch size: 6 (tested 4/6/8)
  - Epochs: 30 (tested 20-50)
  - Patch size: 256px (tested 128/256/512)
- Loss: CrossEntropy (class weights)
- Optimizer: Adam (lr=1e-4)

### Preprocessing
- Class consolidation: 6→4 categories
- Normalization & resizing
- Gaussian filtering (σ=1)

---

## 2. Results & Findings

### Performance
- Best validation accuracy: 89.2% (256px/6batch/30epoch)
- Training time: ~2.5hrs (NVIDIA T4 GPU)

| Config        | Val Acc | Training Time |
|---------------|---------|---------------|
| 128px/b4/e30  | 86.1%   | 1.8h          |
| 256px/b6/e30  | 89.2%   | 2.5h          | 
| 512px/b4/e40  | 87.5%   | 4.1h          |

### Key Observations
1. 256px patches balanced detail and context
2. Batch=6 optimized GPU memory usage
3. 30 epochs sufficient for convergence
4. Larger patches (512px) showed diminishing returns

### Challenges
- Class imbalance (Land=43.7%, Water=12.4%)
- GPU memory constraints
- Small dataset size

---

## 3. Conclusion

### Best Configuration
- **Patch Size:** 256×256  
- **Batch Size:** 6  
- **Epochs:** 30  

### Recommendations
1. Data augmentation for rare classes
2. Multi-scale training
3. Larger dataset collection

**Repository:** [github.com/adi-devv/GeoVision](https://github.com/adi-devv/GeoVision)

### References
1. DeepGlobe Challenge @ CVPR 2018
2. U-Net (Ronneberger 2015)
3. ResNet (He 2016)
