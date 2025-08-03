Land Cover Segmentation Using Deep Learning: Technical Report
Project: Satellite Image Land Cover ClassificationRepository: https://github.com/adi-devv/GeoVisionModel: U-Net with ResNet34 EncoderDataset: DeepGlobe Land Cover ChallengeDate: August 2025

1. Introduction
Land cover classification from satellite imagery supports environmental monitoring, urban planning, and climate research. This project implements a semantic segmentation pipeline to classify satellite images into four land cover types: urban, forests, water, and general land.
1.1 Technical Approach
We used a U-Net architecture with a pre-trained ResNet34 encoder for pixel-wise classification:

Architecture: U-Net with ResNet34 backbone (ImageNet pre-trained)
Class Mapping: Consolidated 6 DeepGlobe classes into 4
Training: Transfer learning with fine-tuning
Inference: Patch-based processing for large images

1.2 Dataset and Preprocessing
Source: DeepGlobe 2018 Land Cover Classification Challenge¹Classes: Simplified from 6 to 4 categories:



Original Classes
Target Classes



Urban (Cyan)
Urban


Agriculture (Yellow)
Land


Rangeland (Magenta)
Land


Forest (Green)
Forest


Water (Blue)
Water


Barren (White)
Land


Unknown (Black)
Ignored


Preprocessing:

Normalized data (uint8/uint16/float)
Resized to 128×128, 256×256, or 512×512 patches
Applied Gaussian filtering (σ=1)
Ensured RGB format (3 channels)


2. Implementation
2.1 Model Architecture
U-Net Configuration:
- Encoder: ResNet34 (ImageNet pre-trained)
- Input: 3 channels (RGB)
- Output: 5 classes (4 target + 1 ignore)
- Loss: CrossEntropyLoss (ignore_index=4)
- Optimizer: Adam (lr=1e-4)

2.2 Training Configuration

Images: 100–120 satellite images
Epochs: Tested 20, 30, 40, 50
Batch Size: Tested 4, 6, 8
Patch Size: Tested 128×128, 256×256, 512×512
Optimal Setup: 256×256 patches, batch size 6, 30 epochs
Device: CUDA-enabled GPU

2.3 Challenges

Memory: Large images caused GPU overflow; solved with patch-based inference.
Class Imbalance: Handled with CrossEntropyLoss and ignore_index.
Data Formats: Robust pipeline using rasterio with matplotlib fallback.
Edge Cases: Padding ensured consistent patch sizes.

2.4 Hyperparameter Tuning
Tested configurations:

Batch Sizes: 4, 6, 8
Epochs: 20, 30, 40, 50
Patch Sizes: 128×128, 256×256, 512×512
Best Performance: 256×256 patches, batch size 6, 30 epochs, balancing accuracy and efficiency.


3. Results
3.1 Performance

Convergence: Stable loss reduction at 30 epochs, no overfitting
Metrics: High pixel-wise accuracy and Mean IoU with 256×256 patches
Output: Interpretable maps (urban: red, forest: green, water: blue, land: yellow)

3.2 Visualization

Side-by-side original vs. segmentation
Color-coded maps
Class distribution statistics
300 DPI outputs

3.3 Limitations

Limited dataset (100–120 images)
Fixed patch sizes may miss large-scale patterns
No temporal analysis

3.4 Future Work

Data augmentation (rotation, color adjustment)
Multi-scale training
Ensemble methods
Spectral indices (NDVI, NDWI) integration


4. Technical Specifications
4.1 Image Assumptions

Resolution: Optimal at 256×256; handles larger images via patches
Spectral: RGB input; converts single-band to RGB
Spatial: Assumes orthorectified images, consistent resolution

4.2 Computational Requirements

GPU: 4GB+ VRAM (6GB for batch size 6)
CPU: 8GB+ RAM
Storage: Space for patches and outputs

4.3 Dependencies
- PyTorch ≥1.9.0
- segmentation_models_pytorch
- rasterio ≥1.2.0
- scikit-image ≥0.18.0
- matplotlib ≥3.3.0
- numpy ≥1.19.0


5. References
¹ DeepGlobe Dataset:
@InProceedings{DeepGlobe18,
 author = {Demir, Ilke and Koperski, Krzysztof and Lindenbaum, David and Pang, Guan and Huang, Jing and Basu, Saikat and Hughes, Forest and Tuia, Devis and Raskar, Ramesh},
 title = {DeepGlobe 2018: A Challenge to Parse the Earth Through Satellite Images},
 booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
 month = {June},
 year = {2018}
}


U-Net: Ronneberger et al. (2015). U-net: Convolutional networks for biomedical image segmentation.
ResNet: He et al. (2016). Deep residual learning for image recognition.
Segmentation Models: Yakubovskiy (2020). https://github.com/qubvel/segmentation_models.pytorch


6. Conclusion
This project demonstrates robust land cover classification using U-Net with a ResNet34 encoder. The optimal setup (256×256 patches, batch size 6, 30 epochs) achieves high accuracy on 100–120 images. The pipeline handles real-world challenges like memory constraints and data variability, showing promise for environmental monitoring.
Key Achievements:

✅ Robust processing pipeline
✅ Memory-efficient inference
✅ Comprehensive error handling
✅ Interpretable outputs
✅ Optimized hyperparameters

Repository Structure:
GeoVision/
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── model_utils.py
│   └── main.py
├── data/
│   ├── train/
│   │   ├── images/
│   │   └── masks/
│   └── valid/
│       ├── images/
│       └── masks/
├── target/
├── outputs/
├── requirements.txt
└── README.md
