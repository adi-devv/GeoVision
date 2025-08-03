# Land Cover Segmentation: Technical Report

**Project:** Satellite Image Classification  
**Model:** U-Net with ResNet34 Encoder  
**Dataset:** DeepGlobe (100-120 images)  
**Best Config:** 256px patches, batch=6, epochs=30  


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


## 2. Results & Findings

### Performance
- Best validation accuracy: 85% (256px/6batch/30epoch)
- Training time: ~1hr (NVIDIA GTX1650 4GB GPU)

### Key Observations
1. 256px patches balanced detail and context
2. Batch=6 optimized GPU memory usage
3. 30 epochs sufficient for convergence
4. Larger patches (512px) showed diminishing returns

### Challenges
- Class imbalance
- Water-Forest color overlap
- GPU memory constraints
- Small dataset size

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
```
@InProceedings{DeepGlobe18,
 author = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
 title = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
 booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 month = {June},
 year = {2018}
}
```
2. U-Net (Ronneberger 2015)
3. ResNet (He 2016)



