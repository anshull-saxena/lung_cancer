# Lung Histopathology Classification System

A comprehensive deep learning and machine learning pipeline for classifying lung histopathology images into three categories: **Adenocarcinoma (ACA)**, **Normal (N)**, and **Squamous Cell Carcinoma (SCC)**.

## 🚀 Overview

This project implements an advanced ensemble classification system that combines:

- **Multiple CNN Backbones**: DenseNet121, ResNet50, and VGG16
- **Channel Attention Mechanism**: Squeeze-and-Excitation (SE) blocks
- **Genetic Algorithm**: Automated feature selection using DEAP
- **Ensemble Learning**: KNN, SVM, and Random Forest classifiers
- **Majority Voting Fusion**: Final prediction aggregation

## 📋 Features

- **Multi-Architecture Feature Extraction**: Leverages three pre-trained CNN models for robust feature representation
- **Attention Mechanism**: SE blocks enhance feature quality by focusing on important channels
- **Automated Feature Selection**: Genetic Algorithm optimizes feature subset selection
- **Ensemble Classification**: Combines multiple traditional ML algorithms for improved accuracy
- **Reproducible Results**: Fixed random seeds ensure consistent outcomes
- **Comprehensive Evaluation**: Detailed classification reports and accuracy metrics

## 🛠️ Installation

### Prerequisites

- Python 3.7-3.10
- CUDA-compatible GPU (recommended for faster training)
- At least 8GB RAM
- Sufficient storage for dataset and model weights

### Setup

1. **Clone the repository** (or download the notebook):
   ```bash
   git clone <your-repository-url>
   cd lung-histopathology-classification
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv lung_classification_env
   source lung_classification_env/bin/activate  # On Windows: lung_classification_env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Dataset Structure

Organize your lung histopathology dataset as follows:

```
/path/to/lung_colon_image_set/lung_image_sets/
├── lung_aca/          # Adenocarcinoma images
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
├── lung_n/            # Normal tissue images
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
└── lung_scc/          # Squamous Cell Carcinoma images
    ├── image1.jpeg
    ├── image2.jpeg
    └── ...
```

**Note**: Update the `DATA_DIR` variable in the notebook to point to your dataset location.

## 🔧 Configuration

Key parameters that can be adjusted in the notebook:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DATA_DIR` | `/path/to/lung_colon_image_set/lung_image_sets` | Path to dataset |
| `IMG_SIZE` | `(224, 224)` | Input image dimensions |
| `BATCH_SIZE` | `24` | Training batch size |
| `POP_SIZE` | `40` | GA population size |
| `N_GEN` | `10` | GA generations |
| `SEED` | `42` | Random seed for reproducibility |

## 🚀 Usage

1. **Open the Jupyter notebook**:
   ```bash
   jupyter notebook code.ipynb
   ```

2. **Update the configuration**:
   - Set `DATA_DIR` to your dataset path
   - Adjust other parameters if needed

3. **Run all cells sequentially**:
   - The notebook is designed to run from top to bottom
   - Each cell includes progress indicators and status messages

## 📊 Pipeline Architecture

### 1. Feature Extraction
- **DenseNet121**: Dense connectivity patterns
- **ResNet50**: Residual connections for deep networks
- **VGG16**: Classic convolutional architecture
- **SE Blocks**: Channel attention for each backbone

### 2. Genetic Algorithm Feature Selection
- **Population Size**: 40 individuals
- **Generations**: 10 (adjustable)
- **Fitness Function**: Cross-validated KNN accuracy with L0 penalty
- **Operations**: Two-point crossover, bit-flip mutation

### 3. Ensemble Classification
- **KNN**: k=5 with distance weighting
- **SVM**: RBF kernel with probability estimates
- **Random Forest**: 300 trees with parallel processing

### 4. Fusion Strategy
- **Majority Voting**: Final prediction based on classifier consensus

## 📈 Expected Results

The system typically achieves:
- **Individual Classifier Accuracy**: 85-95%
- **Ensemble Accuracy**: 90-98%
- **Feature Reduction**: 30-70% of original features selected

Results may vary based on:
- Dataset quality and size
- Hardware specifications
- Random seed variations

## 🔍 Output Interpretation

The notebook provides several key outputs:

### Classification Reports
- **Precision**: Proportion of positive identifications that were correct
- **Recall**: Proportion of actual positives that were identified correctly
- **F1-Score**: Harmonic mean of precision and recall
- **Support**: Number of actual occurrences of each class

### Feature Selection Results
- **Selected Features**: Number and percentage of features chosen by GA
- **Fitness Score**: GA optimization metric
- **Selection Ratio**: Compression achieved

## ⚡ Performance Optimization

### For Better Performance:
1. **Increase GA generations**: More iterations often improve feature selection
2. **Adjust population size**: Larger populations explore solution space better
3. **Tune classifier hyperparameters**: Grid search for optimal settings
4. **Use GPU acceleration**: Ensure TensorFlow uses GPU for CNN inference

### For Faster Execution:
1. **Reduce batch size**: Lower memory requirements
2. **Decrease GA parameters**: Fewer generations/population size
3. **Use fewer CNN backbones**: Comment out unused models
4. **Reduce image resolution**: Smaller input sizes (with potential accuracy trade-off)

## 🐛 Troubleshooting

### Common Issues:

1. **Memory Errors**:
   - Reduce batch size
   - Use smaller image dimensions
   - Ensure sufficient RAM/VRAM

2. **Dataset Not Found**:
   - Verify `DATA_DIR` path
   - Check folder structure matches expected format
   - Ensure image files are in supported formats

3. **Slow Training**:
   - Enable GPU acceleration
   - Reduce GA parameters
   - Use fewer feature extraction models

4. **Import Errors**:
   - Verify all packages are installed correctly
   - Check TensorFlow GPU installation if using CUDA
   - Update pip and setuptools

## 📝 Requirements

See `requirements.txt` for complete dependency list. Key requirements:

- **TensorFlow**: 2.8.0+ (with GPU support recommended)
- **Scikit-learn**: 1.0.0+
- **DEAP**: 1.3.1+ (for genetic algorithms)
- **NumPy/Pandas**: Standard data science stack

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure you have appropriate permissions for any datasets used.

## 📚 References

- **DenseNet**: Huang, G., et al. "Densely connected convolutional networks."
- **ResNet**: He, K., et al. "Deep residual learning for image recognition."
- **SE-Net**: Hu, J., et al. "Squeeze-and-excitation networks."
- **DEAP**: Fortin, F.A., et al. "DEAP: Evolutionary algorithms made easy."

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review the configuration parameters
3. Ensure your dataset follows the expected structure
4. Verify all dependencies are properly installed

---

**Note**: This system is designed for research and educational purposes. For clinical applications, additional validation and regulatory approval would be required.