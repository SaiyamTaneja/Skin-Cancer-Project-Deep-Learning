# Explainable Skin Lesion Classification Using Machine Learning and Deep Learning

An end-to-end skin lesion classification project that combines dermoscopic image datasets, classical machine learning, artificial neural networks, convolutional neural networks, and Vision Transformers to classify skin lesions into eight categories.

---

## Datasets Used

This project uses two publicly available dermoscopic image datasets.

### 1. DERM12345

- **Dataset:** DERM12345
- **Images:** 12,345 dermoscopic images.
- Contains multiple subclasses of skin lesions.
- Includes patient identifiers and diagnostic metadata.

**Dataset Links:**
- [Kaggle – DERM12345](https://www.kaggle.com/datasets/deba4kaggle/derm12345/code)
- [ISIC Archive – DERM12345](https://api.isic-archive.com/doi/derm12345/)

### 2. BCN20000

- **Dataset:** BCN20000
- **Collection:** ISIC Archive.
- Contains dermoscopic images collected at Hospital Clínic, Barcelona.
- Includes lesion-level diagnostic metadata.

**Dataset Link:**
- [ISIC Archive – BCN20000 Collection](https://api.isic-archive.com/collections/249/)

### Dataset Integration

Both datasets are mapped into a common eight-class classification structure.

The original diagnostic labels are mapped into eight unified categories. Unmapped or unsupported diagnostic labels are excluded during preprocessing.

The final processed dataset contains **29,610 images**, distributed across training, validation, and testing sets.

---

## Project Overview

Skin cancer and other skin conditions can present with visually similar characteristics, making accurate classification a challenging task.

This project explores different machine learning and deep learning techniques for classifying dermoscopic skin lesion images. It combines two publicly available datasets, DERM12345 and BCN20000, into a unified dataset with eight common lesion categories.

Multiple models are trained and evaluated using a consistent training, validation, and testing pipeline to understand their performance and limitations.

## Key Features

- Integration of DERM12345 and BCN20000 datasets.
- Unified classification across eight skin lesion categories.
- Exploratory Data Analysis (EDA) and dataset quality checks.
- Image-based feature extraction using HOG, LBP, and Color Histograms.
- Feature selection using Variance Threshold.
- Dimensionality reduction using Principal Component Analysis (PCA).
- Patient/lesion-group-based data splitting to reduce data leakage.
- Class imbalance handling through class weighting and sample weighting.
- Comparison of classical ML and deep learning architectures.
- Model evaluation using Accuracy, Precision, Recall, and Macro F1-score.
- Confusion matrices and classification reports.
- Grad-CAM-based visual explanations for CNN predictions.

## Classification Categories

The project uses the following eight unified classes:

| Label | Category |
|---|---|
| NV | Melanocytic Nevus |
| MEL | Melanoma |
| BCC | Basal Cell Carcinoma |
| SCC | Squamous Cell Carcinoma |
| SK | Seborrheic Keratosis |
| DF | Dermatofibroma |
| VASC | Vascular Lesions |
| BKL | Benign Keratosis-like Lesions |

## Project Workflow

The complete workflow is divided into multiple stages:

1. **Dataset Loading** – Loading images and metadata from both datasets.
2. **Label Mapping** – Mapping original diagnostic labels into eight unified categories.
3. **Data Integration** – Combining datasets and resolving image paths.
4. **Exploratory Data Analysis** – Examining class distributions, missing values, duplicates, and image characteristics.
5. **Data Leakage Prevention** – Creating group-based splits using patient IDs for DERM12345 and lesion IDs for BCN20000.
6. **Feature Extraction** – Extracting handcrafted image features.
7. **Feature Selection** – Removing low-variance features and applying PCA.
8. **Model Development** – Training classical ML, ANN, CNN, and ViT models.
9. **Model Evaluation** – Comparing models using validation and test metrics.
10. **Explainability** – Using Grad-CAM to visualize image regions influencing CNN predictions.

## Data Preprocessing and Feature Extraction

For classical machine learning models, the following handcrafted image features are extracted:

- **Histogram of Oriented Gradients (HOG):** Captures image gradients and structural patterns.
- **Color Histograms:** Extracts colour distribution information from RGB channels.
- **Local Binary Patterns (LBP):** Captures local texture patterns.

The combined feature vector contains 8,214 dimensions:

| Feature | Dimensions |
|---|---:|
| HOG | 8,100 |
| Color Histograms | 96 |
| LBP | 18 |
| **Total** | **8,214** |

Further dimensionality reduction is performed:

- Variance Threshold reduces features from 8,214 to 1,307.
- PCA reduces the feature space to 448 components while retaining 95% explained variance.

For deep learning models, images are resized to 224 × 224 pixels and processed using TensorFlow's image pipeline.

## Data Leakage Prevention

Data leakage is an important concern when working with medical image datasets because multiple images may belong to the same patient or lesion.

To reduce this risk, the project implements group-based data splitting.

- **DERM12345:** Patient ID is used as the grouping identifier.
- **BCN20000:** Lesion ID is used as the grouping identifier because patient IDs are not available in the metadata used by this pipeline.

The final split is:

| Dataset Split | Images | Percentage |
|---|---:|---:|
| Training | 20,356 | 70% |
| Validation | 4,822 | 15% |
| Testing | 4,432 | 15% |
| **Total** | **29,610** | **100%** |

Group overlap assertions are implemented to verify that the same group does not appear across different splits.

## Models Implemented

### Classical Machine Learning

- Random Forest
- Extra Trees Classifier
- Logistic Regression
- HistGradientBoosting Classifier

### Artificial Neural Network (ANN)

A multi-layer feed-forward neural network is developed using:

- Dense layers
- Batch Normalization
- ReLU activation
- Dropout regularization
- Softmax output layer

### Deep Learning Models

**1. EfficientNetB0**

- ImageNet pretrained backbone.
- Frozen pretrained feature extractor.
- Global Average Pooling.
- Dense classification head.
- Data augmentation and dropout.

**2. ResNet50**

- ImageNet pretrained backbone.
- Frozen feature extractor.
- Global Average Pooling.
- Dropout and Softmax classification layer.

**3. Vision Transformer (ViT)**

- Patch-based image representation.
- Multi-Head Self-Attention.
- Layer Normalization.
- MLP blocks.
- Classification head.

Note: The notebook uses a custom Vision Transformer implementation rather than a pretrained ViT model.

## Model Evaluation

All models are evaluated using:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- Classification Report

Macro F1-score is particularly important for this project because the dataset contains significant class imbalance.

### Experimental Results

The following results are taken from the final model comparison generated by the notebook.

| Model | Test Accuracy | Test Macro F1 |
|---|---:|---:|
| EfficientNetB0 | 56.45% | 0.3801 |
| ResNet50 | 60.65% | 0.3778 |
| HistGradientBoosting | 46.37% | 0.2140 |
| ANN | 41.58% | 0.2074 |
| Logistic Regression | 31.34% | 0.1825 |
| Vision Transformer | 48.15% | 0.1656 |
| Random Forest | 51.90% | 0.0878 |
| Extra Trees | 51.78% | 0.0856 |

EfficientNetB0 achieved the highest test Macro F1-score in the recorded experiment, while ResNet50 achieved the highest test accuracy.

These results also highlight the challenges of class imbalance and classification across multiple lesion categories.

## Explainability Using Grad-CAM

Grad-CAM (Gradient-weighted Class Activation Mapping) is incorporated to visualize the image regions that contribute to CNN classification predictions.

It generates heatmaps that can help understand which parts of a dermoscopic image influence a model's output.

This adds an explainability component beyond simply reporting classification probabilities.

## Technology Stack

- Python
- TensorFlow / Keras
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-image
- Pillow
- Joblib
- Jupyter Notebook

## Repository Structure

```text
Skin-Lesion-Classification/
│
├── Skin_Cancer_Classification_Pipeline_Updated.ipynb
│
├── data/
│   ├── DERM12345/
│   └── ISIC-images/
│
├── output/
│   ├── all_combined.csv
│   ├── train_split.csv
│   ├── val_split.csv
│   ├── test_split.csv
│   ├── final_model_comparison_with_vit.csv
│   ├── test_predictions_all_models.csv
│   ├── confusion matrices
│   └── model artifacts
│
├── README.md
└── requirements.txt
```

The dataset images and trained model files are not required to be committed to GitHub. They can be downloaded separately using the provided dataset links.

## Limitations and Future Scope

- Improve classification performance on underrepresented lesion categories.
- Experiment with fine-tuning pretrained CNN architectures.
- Explore pretrained Vision Transformer models.
- Investigate multimodal fusion of image and structured patient metadata.
- Improve generalization across different dataset sources.
- Explore additional explainability techniques.
- Evaluate the models on independent external datasets.

## Contributors

- Saiyam Taneja
- Sarthak
- Sahil

