
# Aggregation Operators in Neural Networks: Multi-Backbone Attention and Ensemble Methods for Lung Histopathology


A comprehensive deep learning and machine learning pipeline for classifying lung histopathology images into three categories: **Adenocarcinoma (ACA)**, **Normal (N)**, and **Squamous Cell Carcinoma (SCC)**. Developed as part of academic research at BITS Pilani, Hyderabad Campus under the supervision of Prof. Swati Hait.

## 🚀 Overview


This project implements an advanced ensemble classification system that combines:

- **Multiple CNN Backbones**: DenseNet121, ResNet50, VGG16, EfficientNetB0, and InceptionV3
- **Attention Mechanisms**: Squeeze-and-Excitation (SE) blocks and Multi-Head Channel Attention
- **Genetic Algorithm**: Automated feature selection using DEAP
- **Ensemble Learning**: KNN, SVM, and Random Forest classifiers
- **Majority Voting Fusion**: Final prediction aggregation


### 🎯 Performance Results

The system achieves competitive performance on lung histopathology classification:

- **Single-head CNN**: 98.13% ensemble accuracy
- **Multi-head CNN**: 98.70% ensemble accuracy
- **Individual Classifiers**: 97–98% accuracy across KNN, SVM, and Random Forest
- **Dataset**: 15,000 images (5,000 per class) with balanced distribution

## 📋 Features


- **Multi-Architecture Feature Extraction**: Leverages five pre-trained CNN models for robust feature representation
- **Attention Mechanisms**: SE blocks and multi-head channel attention enhance feature quality by focusing on important channels
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

The project uses a balanced lung histopathology dataset with the following structure:

```
dataset/lung_image_sets/
├── lung_aca/          # Adenocarcinoma images (5,000 images)
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
├── lung_n/            # Normal tissue images (5,000 images)
│   ├── image1.jpeg
│   ├── image2.jpeg
│   └── ...
└── lung_scc/          # Squamous Cell Carcinoma images (5,000 images)
    ├── image1.jpeg
    ├── image2.jpeg
    └── ...
```

### Dataset Statistics
- **Total Images**: 15,000
- **Classes**: 3 (ACA, Normal, SCC)
- **Images per Class**: 5,000
- **Format**: JPEG
- **Resolution**: Variable (resized to 224×224 during preprocessing)

**Note**: Update the `DATA_DIR` variable in the notebook to point to your dataset location. The default path is `dataset/lung_image_sets`.

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

## 📓 Notebook Implementations

This project includes multiple notebook variants for different experimental approaches:


### 1. `code.ipynb` - Single-Head Attention Implementation
- **Architecture**: 3 CNN backbones (DenseNet121, ResNet50, VGG16)
- **Attention**: Single-head Squeeze-and-Excitation blocks
- **Performance**: 98.13% ensemble accuracy
- **Output**: `output1.txt`


### 2. `code_multihead.ipynb` - Multi-Head Attention Implementation (Current)
- **Architecture**: 5 CNN backbones (DenseNet121, ResNet50, VGG16, EfficientNetB0, InceptionV3)
- **Attention**: Multi-head Channel Attention mechanism (8 heads)
- **Performance**: 98.70% ensemble accuracy
- **Output**: `output_multi.txt`
- **Features**: GPU-optimized implementation with advanced attention

### 3. `code_multihead_original.ipynb` - Original Multi-Head Prototype
- **Purpose**: Development version of multi-head attention
- **Status**: Superseded by `code_multihead.ipynb`

## 🚀 Usage

### Quick Start

1. **Choose your implementation**:
   ```bash
   # For single-head attention (faster, good performance)
   jupyter notebook code.ipynb
   
   # For multi-head attention (best performance)
   jupyter notebook code_multihead.ipynb
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

## 📈 Performance Results

### Single-Head Attention Implementation (`code.ipynb`)
- **KNN Accuracy**: 97.13%
- **SVM Accuracy**: 98.03%
- **Random Forest Accuracy**: 97.53%
- **Ensemble Accuracy**: 98.13%
- **Improvement over best individual**: 0.10%

### Multi-Head Attention Implementation (`code_multihead.ipynb`)
- **KNN Accuracy**: 98.43%
- **SVM Accuracy**: 98.17%
- **Random Forest Accuracy**: 97.53%
- **Ensemble Accuracy**: 98.70%
- **Improvement over best individual**: 0.27%

### Key Performance Insights
- **Multi-head attention** provides a 0.57% improvement over single-head attention
- **SVM** consistently performs best among individual classifiers
- **Ensemble fusion** provides reliable performance gains
- **Feature reduction** through GA maintains high accuracy while reducing computational complexity

### Detailed Classification Reports
Both implementations achieve excellent performance across all classes:
- **Precision**: 94-99% across all classes
- **Recall**: 94-100% across all classes  
- **F1-Score**: 96-98% across all classes
- **Support**: 1,000 samples per class (balanced test set)

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

## 🔧 Analysis Tools

### Results Comparison Utility (`compare_results.py`)

A Python utility for comparing results from different experimental runs:

```bash
python compare_results.py
```

**Features**:
- Parses multiple output files (`output1.txt`, `output_multi.txt`, etc.)
- Extracts key metrics: KNN, SVM, RF, and Ensemble accuracies
- Extracts GA feature selection statistics
- Generates formatted comparison tables
- Identifies best performing configuration

**Usage Example**:
```python
# Compare two experimental results
python compare_results.py
# Enter number of files: 2
# Enter path for file 1: output1.txt
# Enter path for file 2: output_multi.txt
```

**Output Format**:
- Tabular comparison of all metrics
- Highlighted best ensemble accuracy
- Feature selection efficiency comparison

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

## 📄 Project Files

### Core Implementation Files
- **`code.ipynb`**: Single-head attention implementation
- **`code_multihead.ipynb`**: Multi-head attention implementation (recommended)
- **`code_multihead_original.ipynb`**: Original multi-head prototype
- **`compare_results.py`**: Results comparison utility

### Output Files
- **`output1.txt`**: Results from single-head attention experiment
- **`output_multi.txt`**: Results from multi-head attention experiment

### Documentation & Reports
- **`lung_cancer_classification_report.tex`**: Comprehensive LaTeX report
- **`report_preview.html`**: HTML preview of the report
- **`paper.pdf`**: Final research paper
- **`PROJECT_BITS_PILANI_GROUP_1 (2) (2).pdf`**: Project documentation
- **`latex_project.zip`**: LaTeX source files archive

### Assets
- **`bits_logo.png`**: BITS Pilani logo
- **`dataset/`**: Lung histopathology dataset (15,000 images)

## 📝 Requirements

See `requirements.txt` for complete dependency list. Key requirements:

- **TensorFlow**: 2.15.0 (with GPU support recommended)
- **NumPy**: 1.24.4
- **Scikit-learn**: Latest version
- **DEAP**: Latest version (for genetic algorithms)
- **Pandas**: Latest version
- **Pillow**: Image processing
- **Jupyter**: Notebook environment
- **Matplotlib/Seaborn**: Visualization

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Add tests and documentation
5. Submit a pull request

## 📄 License

This project is provided as-is for educational and research purposes. Please ensure you have appropriate permissions for any datasets used.

## 🎓 Research Context


This project was developed as part of academic research at BITS Pilani, Hyderabad Campus, under the supervision of Prof. Swati Hait, focusing on advanced deep learning techniques for medical image analysis. The implementation demonstrates:

- **Multi-head attention mechanisms** for improved feature representation
- **Ensemble learning** for robust classification performance
- **Genetic algorithms** for automated feature selection
- **Comprehensive evaluation** with detailed performance metrics

## 📚 References

- **DenseNet**: Huang, G., et al. "Densely connected convolutional networks." CVPR 2017.
- **ResNet**: He, K., et al. "Deep residual learning for image recognition." CVPR 2016.
- **SE-Net**: Hu, J., et al. "Squeeze-and-excitation networks." CVPR 2018.
- **Multi-Head Attention**: Vaswani, A., et al. "Attention is all you need." NIPS 2017.
- **DEAP**: Fortin, F.A., et al. "DEAP: Evolutionary algorithms made easy." JMLR 2012.
- **EfficientNet**: Tan, M., et al. "EfficientNet: Rethinking model scaling for convolutional neural networks." ICML 2019.
- **Inception**: Szegedy, C., et al. "Going deeper with convolutions." CVPR 2015.

## 📞 Support

For issues, questions, or suggestions:
1. Check the troubleshooting section
2. Review the configuration parameters
3. Ensure your dataset follows the expected structure
4. Verify all dependencies are properly installed

---

## 🏆 Key Achievements

- ✅ **98.70% accuracy** with multi-head attention implementation
- ✅ **5 CNN architectures** integrated with attention mechanisms  
- ✅ **15,000 images** processed across 3 lung cancer types
- ✅ **Genetic algorithm** optimization for feature selection
- ✅ **Comprehensive evaluation** with detailed performance metrics
- ✅ **Reproducible results** with fixed random seeds
- ✅ **Complete documentation** including LaTeX report and analysis tools

**Note**: This system is designed for research and educational purposes. For clinical applications, additional validation and regulatory approval would be required.