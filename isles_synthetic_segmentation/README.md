# Lesion Segmentation in DWI Images

A comprehensive deep learning framework for training and testing models to segment lesions in Diffusion Weighted Imaging (DWI) images.

## Overview

This repository contains tools and models for automated lesion segmentation in DWI images, which is crucial for stroke diagnosis and treatment planning. The framework supports multiple deep learning architectures and provides comprehensive training, validation, and testing pipelines.

## Directory Structure

```
isles_synthetic_segmentation/
├── models/                 # Model architectures (Python files)
├── datasets/               # Data management utilities
├── train/                  # Training scripts and configurations
├── test/                   # Testing and evaluation scripts
├── utils/                  # Utility functions
├── configs/                # Configuration files
├── logs/                   # Training and validation logs
├── results/                # Results and outputs
└── docs/                   # Documentation
```

## Features

- **Multiple Model Architectures**: U-Net, DeepLab, Attention U-Net, and Transformer-based models
- **Comprehensive Data Pipeline**: Raw data processing, augmentation, and loading utilities
- **Flexible Training**: Configurable training scripts with logging and checkpointing
- **Robust Evaluation**: Multiple evaluation metrics (Dice, IoU, Hausdorff distance, etc.)
- **Visualization Tools**: Tools for visualizing predictions, training progress, and results
- **Modular Design**: Easy to extend and modify for different datasets and models

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 1.9+
- Medical imaging libraries (nibabel, SimpleITK)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd isles_synthetic_segmentation
```

2. Activate the conda environment:
```bash
conda activate glasses
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

1. Place your DWI images and corresponding masks in `datasets/`
2. Run preprocessing scripts to prepare data for training:
```bash
python utils/preprocess.py --config configs/preprocessing.yaml
```

### Training

1. Configure training parameters in `configs/`
2. Start training:
```bash
python train/train.py --config configs/train_config.yaml
```

### Testing

1. Evaluate trained models:
```bash
python test/evaluate.py --model_path path/to/model --data_path datasets/test
```

2. Generate predictions:
```bash
python test/predict.py --model_path path/to/model --input_path path/to/images --output_path results/predictions
```

## Model Architectures

### U-Net
- Classic U-Net architecture with skip connections
- Suitable for medical image segmentation
- Configurable depth and feature channels

### DeepLab
- Atrous spatial pyramid pooling
- Better handling of multi-scale features
- Improved boundary accuracy

### Attention U-Net
- Attention gates for better feature selection
- Focus on relevant regions
- Enhanced segmentation accuracy

### Transformer-based Models
- Vision Transformer (ViT) adaptations
- Self-attention mechanisms
- State-of-the-art performance potential

## Evaluation Metrics

- **Dice Coefficient**: Measures overlap between prediction and ground truth
- **IoU (Intersection over Union)**: Similar to Dice but with different normalization
- **Hausdorff Distance**: Measures boundary accuracy
- **Sensitivity/Specificity**: Clinical relevance metrics
- **Precision/Recall**: Detailed performance analysis

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```

## Contact

For questions and support, please open an issue on GitHub or contact [your-email@domain.com]

## Acknowledgments

- ISLES challenge organizers
- PyTorch community
- Medical imaging research community 