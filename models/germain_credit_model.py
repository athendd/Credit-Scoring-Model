import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, auc, roc_curve, classification_report, precision_recall_fscore_support, make_scorer
import xgboost as xgb
from imblearn.over_sampling import BorderlineSMOTE

"""
Find outliers using the z score
"""
def find_outliers_zscore(df):
    outlier_indices = {}
    for col in df.columns:
        z_score = stats.zscore(df[col])
        outliers = df[abs(z_score) > 3].index.tolist()
        outlier_indices[col] = len(outliers)
    
    return outlier_indices

df = pd.read_csv('german_credit_dataset.csv')

#Only using necessary columns
df = df[['Duration_in_month', 'Credit_amount', 'Status_of_existing_checking_account',
         'Credit_history', 'Savings_account/bonds', 'Purpose', 'Property', 'Present_employment_since', 'Housing', 'Other_installment_plans',
         'Other_debtors/guarantors', 'credit_risk']]

num_df = df.select_dtypes(include = 'number')
    
outliers = find_outliers_zscore(num_df)

sns.violinplot(y = df['Duration_in_month'])
plt.show()

#Removed extreme outliers from Duration_in_month
df = df.drop(df[df['Duration_in_month'] > 60].index)

sns.violinplot(y = df['Credit_amount'])
plt.show()

#Removed extreme outliers from Credit_amount
df = df.drop(df[df['Credit_amount'] > 17000].index)

sns.violinplot(y = df['Installment_rate_in_percentage_of_disposable_income'])
plt.show()

#Obtained total null values for each column
total_null_values = df.isnull().sum()

#Converted categorical columns to numerical using one-hot encoding 
df = pd.get_dummies(df, columns = ['Status_of_existing_checking_account', 'Credit_history',
                    'Savings_account/bonds', 'Purpose', 'Property', 'Present_employment_since',
                    'Housing', 'Other_installment_plans', 'Other_debtors/guarantors'])

#Checked the correlation between each column
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.show()

#Created training and testing datasets
x = df.drop('credit_risk', axis = 1)
y = df[['credit_risk']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

#Turns data into a one dimensional array
y_train_1d = np.ravel(y_train)

#Resampls data distribution 
sm = BorderlineSMOTE(random_state = 42)
x_train_res, y_train_res = sm.fit_resample(x_train, y_train_1d)

count_good = y_train.value_counts().get(1, 0) 
count_bad = y_train.value_counts().get(0, 0)   

scale_pos_weight_value = count_good / count_bad if count_bad > 0 else 1

params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "scale_pos_weight": [scale_pos_weight_value, scale_pos_weight_value * 1.5, scale_pos_weight_value * 0.8],
    "gamma": [0, 1, 5, 7, 10],
    "reg_alpha": [0, 0.5, 1],
    "reg_lambda": [1, 2, 3]
}

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

f1_scorer_bad_class = make_scorer(f1_score, pos_label = 0, average= 'binary')

search = RandomizedSearchCV(xgb.XGBClassifier(), param_distributions=params, 
                            n_iter=20, scoring= f1_scorer_bad_class, cv=skf, random_state=42, verbose=1)

search.fit(x_train_res, y_train_res)
y_pred = search.predict(x_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')

preds = search.predict_proba(x_test)
preds_df = pd.DataFrame(preds[:, 1], columns = ['prob_default'])

thresholds = [0.1, 0.9, 0.95, 0.8, 0.85, 0.7, 0.75, 0.6, 0.65, 0.5, 0.55, 0.4, 0.45, 0.2, 0.25, 0.3, 0.25, 0.15, 0.05]
f1_scores = [f1_score(y_test, (preds[:, 1] > t).astype(int)) for t in thresholds]
threshold = thresholds[np.argmax(f1_scores)]

print(f"Best threshold by F1 score: {threshold:.2f}")

#Assign credit_risk based on threshold
preds_df['credit_risk'] = preds_df['prob_default'].apply(lambda x: 1 if x > threshold else 0)
target_names = ['Bad', 'Good']
print(classification_report(y_test, preds_df["credit_risk"], target_names = target_names))

cm = confusion_matrix(y_test, preds_df['credit_risk'])

display_cm = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = target_names)
display_cm.plot()
plt.show()

prob_good = preds[:, 1]

fallout, sensitivity, thresholds = roc_curve(y_test, prob_good)

roc_auc = roc_auc_score(y_test, prob_good)

print(f'ROC AUC Value: {roc_auc}')

plt.figure(figsize = (12,8))
plt.plot(fallout, sensitivity, linewidth = 2.5)
plt.plot([0,1], [0,1], linestyle = '--', color = 'black')
plt.title("ROC Curve for Model", fontsize=18, fontweight='bold')
plt.xlabel('Fall-out (False Positive Rate)', fontsize=14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=14)
plt.grid(True)
plt.legend(loc='lower right', fontsize=12)
plt.tight_layout()
plt.show()
