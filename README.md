# Credit-Scoring-Model
This project involves the development and implementation of two XGBoost-based classifiers to evaluate credit risk across two separate datasets. The classifiers are designed to distinguish between:

Default / Non-default cases

Good / Bad credit risk categories

The key challenge addressed in this project is the class imbalance inherent in both datasets, where the minority class (e.g., default or bad credit) is significantly underrepresented.

# Project Highlights
Implemented two distinct XGBoost classifiers tailored for different credit risk datasets.

Applied BorderlineSMOTE for oversampling to address severe class imbalance.

Conducted feature selection using statistical tests (two-sample t-tests and chi-squared tests).

Employed stratified k-fold cross-validation to ensure reliable model evaluation.

Performed hyperparameter tuning via RandomizedSearchCV for optimal model performance.

Visualized model performance using confusion matrices and ROC AUC curves.

# Methodology
1. Exploratory Data Analysis
Used Seaborn and Matplotlib for visualizing data distributions and class relationships.

Identified missing values, outliers, and distribution skews.

2. Feature Selection
Continuous features: Filtered using two-sample t-tests.

Categorical features: Selected using chi-squared tests.

Removed features with weak predictive power or high correlation.

3. Handling Class Imbalance
Applied BorderlineSMOTE (from imblearn) to oversample the minority class, preserving boundary samples for better decision-making.

4. Model Training and Validation
Utilized XGBoost Classifier (xgboost.XGBClassifier) with stratified k-fold cross-validation to prevent overfitting and ensure class representation.

Performed hyperparameter optimization using RandomizedSearchCV.

5. Evaluation Metrics
Confusion Matrix to evaluate true vs predicted labels.

ROC AUC Score to assess classifier performance on imbalanced data.

Precision, Recall, F1-Score for additional performance insight.

# Technologies Used

Python 3.12
XGBoost
scikit-learn
imbalanced-learn (BorderlineSMOTE)
NumPy, Pandas
Seaborn, Matplotlib

# Key Takeaways

This project provided hands-on experience with:

Addressing real-world class imbalance using advanced oversampling.

Making data-driven feature selection decisions.

Applying robust machine learning techniques in financial risk contexts.

Interpreting model results using comprehensive evaluation metrics.

# Results

Confusion Matrix:
![fhads](https://github.com/user-attachments/assets/e5fa7d80-ea01-42c4-b78d-fc0db944d9c8)

Roc Auc Curve:
![fadac](https://github.com/user-attachments/assets/b37243ec-3b4e-4e36-b45d-e4000e442bef)

Sharp Graph:
![afds](https://github.com/user-attachments/assets/f211080a-4494-412d-ad03-e053ddf4fe51)

Output From Model One:
![image](https://github.com/user-attachments/assets/5fb1f527-7657-4412-8d18-5b736dfa94a0)

Additional Plots for Analysis:
![fdc](https://github.com/user-attachments/assets/e0375ff4-6045-4934-93fa-dd83c5e1c33b)
![dadc](https://github.com/user-attachments/assets/ecc9f611-0581-4139-a03a-f5627595c7ee)
![dacse](https://github.com/user-attachments/assets/726ee546-138b-42b7-b445-912f0ee90ecd)







