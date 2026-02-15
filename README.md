# Heart Disease Prediction - Machine Learning Assignment 2

## Problem Statement
This project implements multiple machine learning classification models to predict the presence or absence of heart disease in patients based on various medical features. The goal is to compare the performance of different algorithms and deploy an interactive web application for real-time predictions.

## Dataset Description
- **Dataset Name**: Heart Disease Classification Dataset
- **Source**: UCI Machine Learning Repository / Kaggle
- **Type**: Binary Classification Problem
- **Target Variable**: Heart Disease (Presence/Absence)
- **Number of Instances**: 5,000 (sampled from larger dataset)
- **Number of Features**: 13

### Features:
1. Age - Age in years
2. Sex - Gender (1 = male, 0 = female)
3. Chest pain type - Type of chest pain (1-4)
4. BP - Resting blood pressure (mm Hg)
5. Cholesterol - Serum cholesterol in mg/dl
6. FBS over 120 - Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. EKG results - Resting electrocardiographic results (0-2)
8. Max HR - Maximum heart rate achieved
9. Exercise angina - Exercise induced angina (1 = yes, 0 = no)
10. ST depression - ST depression induced by exercise
11. Slope of ST - Slope of peak exercise ST segment
12. Number of vessels fluro - Number of major vessels (0-3)
13. Thallium - Thallium stress test result

## Models Used

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8840 | 0.9532 | 0.8825 | 0.8860 | 0.8842 | 0.7680 |
| Decision Tree | 0.8100 | 0.8100 | 0.8163 | 0.8000 | 0.8081 | 0.6201 |
| KNN | 0.7210 | 0.7863 | 0.7297 | 0.7020 | 0.7156 | 0.4423 |
| Naive Bayes | 0.8680 | 0.9389 | 0.8710 | 0.8640 | 0.8675 | 0.7360 |
| Random Forest (Ensemble) | 0.8720 | 0.9500 | 0.8735 | 0.8700 | 0.8717 | 0.7440 |
| XGBoost (Ensemble) | 0.8730 | 0.9461 | 0.8635 | 0.8860 | 0.8746 | 0.7463 |

### Model Performance Observations

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | **Best overall performer** with highest accuracy (88.4%) and AUC (95.3%). Demonstrates excellent linear separability in the dataset. Highly interpretable and reliable for medical predictions. Best choice for deployment due to consistent performance across all metrics. |
| Decision Tree | Moderate performance with 81% accuracy. Prone to overfitting despite good training performance. Quick training time but lacks generalization. May benefit from pruning or ensemble methods. Useful for understanding feature importance and decision rules. |
| KNN | **Lowest performer** with 72.1% accuracy. Distance-based approach struggles with this dataset's feature space. Sensitive to feature scaling and curse of dimensionality. High computational cost during prediction. Not recommended for this problem. |
| Naive Bayes | Strong performance (86.8% accuracy) with excellent AUC (93.9%). Fast training and prediction. Assumes feature independence which may not hold perfectly but still performs well. Good balance between simplicity and accuracy. Efficient choice for real-time predictions. |
| Random Forest (Ensemble) | Excellent ensemble method with 87.2% accuracy and 95% AUC. Robust to overfitting through bagging. Handles non-linear relationships well. Feature importance insights valuable for medical interpretation. Slightly slower than single models but highly reliable. |
| XGBoost (Ensemble) | Top ensemble performer with 87.3% accuracy. Advanced gradient boosting handles complex patterns effectively. Best recall (88.6%) - crucial for medical diagnosis to minimize false negatives. Excellent balance of precision and recall. Recommended for production alongside Logistic Regression. |

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Steps