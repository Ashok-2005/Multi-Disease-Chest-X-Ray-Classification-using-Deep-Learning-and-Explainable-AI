# Multi-Disease Chest X-Ray Classification using Deep Learning and Explainable AI

## 📌 Overview

This project presents an AI-powered system for **multi-label chest X-ray disease classification** using transfer learning and Explainable AI (XAI). The model can identify multiple thoracic diseases from a single chest X-ray image while providing visual explanations using **Grad-CAM**.

The system leverages pretrained deep learning models, including **EfficientNet-B0, DenseNet121, MobileNet, and VGGNet**, to automatically extract features from chest X-ray images and predict multiple diseases simultaneously. The project aims to support radiologists by providing fast, accurate, and interpretable disease predictions.

---

## 🎯 Objectives

- Develop a multi-label chest X-ray disease classification system.
- Detect multiple diseases from a single chest X-ray image.
- Compare the performance of different pretrained CNN models.
- Improve classification accuracy using transfer learning.
- Generate Grad-CAM heatmaps for model interpretability.
- Assist healthcare professionals with AI-supported diagnosis.

---

## 🏥 Diseases Detected

The model predicts the following 10 thoracic conditions:

- Atelectasis
- Cardiomegaly
- Effusion
- Infiltration
- Mass
- Nodule
- Pneumonia
- Pneumothorax
- Consolidation
- No Finding

---

## 🛠️ Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Torchvision
- PIL
- Grad-CAM
- Jupyter Notebook
- Kaggle

---

## 📂 Dataset

Dataset used:

- **NIH ChestX-ray14 Dataset**
- Kaggle version of NIH ChestX-ray14

### Dataset Statistics

- **Total Images:** 112,120
- **Training + Validation:** 86,524
- **Testing:** 25,596

The project uses a subset of **10 disease classes** for multi-label classification.

---

## 🔄 Image Preprocessing

The preprocessing pipeline includes:

- Resize images to **224 × 224**
- Pixel normalization
- Tensor conversion
- Data augmentation
  - Horizontal Flip
- Mini-batch loading using PyTorch DataLoader

---

# 🧠 Deep Learning Models

The following pretrained CNN models were evaluated:

- EfficientNet-B0
- DenseNet121
- MobileNet-V2
- VGG16

Transfer learning was applied by replacing the final classification layer while retaining pretrained ImageNet weights.

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Framework | PyTorch |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 20 |
| Optimizer | Adam |
| Loss Function | Binary Cross Entropy |
| Activation | Sigmoid |

---

## 📊 Multi-Label Prediction Strategy

Unlike traditional classification, the model predicts **multiple diseases simultaneously**.

Prediction strategy:

- Sigmoid activation for independent disease probabilities.
- "No Finding" threshold: **0.50**
- Disease threshold: **0.275**

This custom thresholding strategy improves sensitivity while reducing false detections.

---

# 📈 Model Performance

| Model | Training Loss | Validation Loss | Validation Accuracy |
|--------|--------------:|---------------:|--------------------:|
| MobileNet-V2 | 0.1446 | 0.1466 | **94.24%** |
| EfficientNet-B0 | 0.2260 | 0.2290 | **90.91%** |
| VGG16 | 0.2022 | 0.2091 | **82.01%** |
| DenseNet121 | 0.1166 | 0.1372 | **57.47%** |

---

## 🏆 Best Performing Model

**MobileNet**

- Validation Accuracy: **94.24%**
- Lowest validation loss among the best-performing models
- Fast convergence
- Computationally efficient

---

## 🔍 Explainable AI (Grad-CAM)

To improve model interpretability, **Grad-CAM** is used.

Grad-CAM generates heatmaps highlighting image regions responsible for disease predictions.

Benefits:

- Improves transparency
- Builds trust in AI predictions
- Helps verify model decisions
- Supports clinical interpretation

---

## 📊 Evaluation Metrics

The project evaluates models using:

- Training Loss
- Validation Loss
- Validation Accuracy
- ROC Curve
- Precision-Recall Curve
- Confusion Matrix
- Accuracy per Class
- Grad-CAM Visualization

---

## 🚀 Features

- Multi-label disease prediction
- Transfer learning with pretrained CNNs
- EfficientNet-B0 implementation
- MobileNet implementation
- DenseNet121 implementation
- VGGNet implementation
- Binary Cross Entropy loss
- Explainable AI using Grad-CAM
- Probability-based disease prediction
- ROC and Precision-Recall analysis
- Confusion Matrix visualization
- Invalid image filtering
- GPU-supported PyTorch implementation

---

## 📌 Key Contributions

- Developed an automated multi-label chest X-ray classification system.
- Compared four state-of-the-art pretrained CNN architectures.
- Implemented transfer learning for efficient medical image analysis.
- Applied Grad-CAM for explainable predictions.
- Designed a threshold-based inference strategy for improved reliability.
- Evaluated models using multiple performance metrics.

---

## 🔮 Future Work

- Increase the number of disease classes.
- Fine-tune larger transformer-based models (ViT, Swin Transformer).
- Address class imbalance using advanced loss functions.
- Improve model calibration and probability estimation.
- Deploy the system as a web application for clinical use.
- Integrate with hospital PACS systems for real-time diagnosis.

---

## 👨‍💻 Authors

- **Venkata Ashok Adithya**
- **Harshavardhan Reddy**
- **Dinesh Reddy**

**School of Computer Science and Engineering**

**VIT-AP University**

