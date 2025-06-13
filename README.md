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
