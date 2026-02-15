# Heart Disease Prediction - Machine Learning Classification

## Problem Statement

The objective of this project is to develop and compare multiple machine learning classification models to predict the presence or absence of heart disease in patients based on various medical and physiological features. This is a binary classification problem where the goal is to accurately identify patients at risk of heart disease, enabling early intervention and treatment.

## Dataset Description

- **Dataset Name**: Heart Disease Classification Dataset
- **Source**: Public medical dataset
- **Problem Type**: Binary Classification
- **Target Variable**: Heart Disease (Presence/Absence)
- **Number of Instances**: 5,000 rows
- **Number of Features**: 13 features
- **Feature Types**: Mix of numerical and categorical variables

### Features:
1. **Age** - Age of the patient in years
2. **Sex** - Gender (1 = male, 0 = female)
3. **Chest pain type** - Type of chest pain experienced (1-4)
4. **BP** - Resting blood pressure (mm Hg)
5. **Cholesterol** - Serum cholesterol level in mg/dl
6. **FBS over 120** - Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
7. **EKG results** - Resting electrocardiographic results (0-2)
8. **Max HR** - Maximum heart rate achieved
9. **Exercise angina** - Exercise induced angina (1 = yes, 0 = no)
10. **ST depression** - ST depression induced by exercise relative to rest
11. **Slope of ST** - Slope of the peak exercise ST segment
12. **Number of vessels fluro** - Number of major vessels colored by fluoroscopy (0-3)
13. **Thallium** - Thallium stress test result

### Target Variable:
- **Heart Disease**: Presence (1) or Absence (0) of heart disease

The dataset meets the assignment requirements with 13 features (minimum 12 required) and 5,000 instances (minimum 500 required).

## Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8840 | 0.9532 | 0.8825 | 0.8860 | 0.8842 | 0.7680 |
| Decision Tree | 0.8100 | 0.8100 | 0.8163 | 0.8000 | 0.8081 | 0.6201 |
| kNN | 0.7210 | 0.7863 | 0.7297 | 0.7020 | 0.7156 | 0.4423 |
| Naive Bayes | 0.8680 | 0.9389 | 0.8710 | 0.8640 | 0.8675 | 0.7360 |
| Random Forest (Ensemble) | 0.8720 | 0.9500 | 0.8735 | 0.8700 | 0.8717 | 0.7440 |
| XGBoost (Ensemble) | 0.8730 | 0.9461 | 0.8635 | 0.8860 | 0.8746 | 0.7463 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Logistic Regression achieved the best overall performance with the highest accuracy (88.40%) and AUC score (95.32%). It demonstrates excellent linear separability in the heart disease dataset and provides highly interpretable results through coefficient analysis. The model shows balanced precision (88.25%) and recall (88.60%), making it highly reliable for medical diagnosis where both false positives and false negatives need to be minimized. Its superior AUC score indicates excellent ability to distinguish between the two classes across all threshold values. |
| Decision Tree | Decision Tree classifier achieved moderate performance with 81.00% accuracy. While it offers excellent interpretability through visual tree structure and feature importance, it shows signs of overfitting with AUC equal to accuracy (81.00%). The model has the fastest training time but relatively lower generalization capability compared to ensemble methods. The MCC score of 0.6201 indicates moderate correlation between predictions and actual values. It may benefit from pruning or converting to an ensemble approach. |
| kNN | k-Nearest Neighbors showed the weakest performance among all models with only 72.10% accuracy and the lowest MCC score (0.4423). The distance-based approach struggles with the high-dimensional feature space of this dataset. The model is sensitive to feature scaling and suffers from the curse of dimensionality. Additionally, it has high computational cost during prediction phase as it requires calculating distances to all training samples. Not recommended for this particular problem. |
| Naive Bayes | Naive Bayes (Gaussian) delivered strong performance with 86.80% accuracy and an impressive AUC of 93.89%. Despite its assumption of feature independence (which may not hold perfectly in medical data), it performs remarkably well. The model offers extremely fast training and prediction times, making it suitable for real-time applications. With balanced precision (87.10%) and recall (86.40%), it provides a good tradeoff between computational efficiency and predictive accuracy. |
| Random Forest (Ensemble) | Random Forest achieved excellent performance with 87.20% accuracy and 95.00% AUC score, demonstrating the power of ensemble learning. The bagging approach with 100 decision trees effectively reduces overfitting and captures complex non-linear relationships in the data. The model provides robust feature importance rankings which are valuable for medical interpretation. With MCC of 0.7440, it shows strong correlation between predictions and ground truth. Slightly slower than single models but offers superior reliability and generalization. |
| XGBoost (Ensemble) | XGBoost (Gradient Boosting) achieved the second-highest accuracy (87.30%) and excellent performance across all metrics. The advanced gradient boosting technique handles complex patterns effectively through sequential learning. It achieved the highest recall (88.60%) among ensemble methods, which is crucial in medical diagnosis to minimize false negatives (missing actual heart disease cases). The MCC score of 0.7463 is the highest, indicating best overall classification quality. Balanced precision-recall tradeoff makes it highly suitable for deployment in clinical settings. |
