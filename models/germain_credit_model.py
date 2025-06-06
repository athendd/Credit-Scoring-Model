import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, auc, roc_curve, classification_report, precision_recall_fscore_support
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

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

"""
Number of Outliers in each Numerical Column
    Duration_in_month: 14
    Credit_amount: 25
    Installment_rate_in_percentage_of_disposable_income: 0
    Age_in_years: 7
"""
outliers = find_outliers_zscore(num_df)

#Checked age to see if outliers are really outliers
#print(max(df['Age_in_years']))

"""
sns.violinplot(y = df['Duration_in_month'])
plt.show()
"""

#Removed extreme outliers from Duration_in_month
df = df.drop(df[df['Duration_in_month'] > 60].index)

"""
sns.violinplot(y = df['Credit_amount'])
plt.show()
"""

#Removed extreme outliers from Credit_amount
df = df.drop(df[df['Credit_amount'] > 17000].index)

"""
sns.violinplot(y = df['Installment_rate_in_percentage_of_disposable_income'])
plt.show()
"""

#Obtained total null values for each column
total_null_values = df.isnull().sum()

#Converted categorical columns to numerical using one-hot encoding 
df = pd.get_dummies(df, columns = ['Status_of_existing_checking_account', 'Credit_history',
                    'Savings_account/bonds', 'Purpose', 'Property', 'Present_employment_since',
                    'Housing', 'Other_installment_plans', 'Other_debtors/guarantors'])

"""
#Checked the correlation between each column
correlation_matrix = df.corr()

sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.show()
"""

#Created training and testing datasets
x = df.drop('credit_risk', axis = 1)
y = df[['credit_risk']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

#Turns data into a one dimensional array
y_train_1d = np.ravel(y_train)

#Added a scale position weight since there are way more 1s than 0s in credit_risk
scale = sum(y_train_1d == 0) / sum(y_train_1d == 1)

xgb_clf = xgb.XGBClassifier(scale_pos_weight = scale)

xgb_clf.fit(x_train, y_train_1d)
y_pred = xgb_clf.predict(x_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred):.4f}')
print(f'Precision: {precision_score(y_test, y_pred):.4f}')

preds = xgb_clf.predict_proba(x_test)
preds_df = pd.DataFrame(preds[:, 1], columns = ['prob_default'])
threshold = 0.5

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

"""
Accuracy: 0.7467
Precision Good: 0.8535
Precision Bad: 0.54
F1 Score Good: 0.82
F1 Score Bad: 0.59
Performed poorly on bad credit risk vs good credit risk because
there are a lot more good credit risk rows
"""

preds_df["prob_default"].hist()
plt.show()

#Increasing the threshold from 50% to 85%
threshold_85 = np.quantile(preds_df['prob_default'], 0.85)
print(threshold_85)

preds_df['credit_risk'] = preds_df['prob_default'].apply(lambda x: 1 if x > threshold_85 else 0)
print(preds_df['credit_risk'].value_counts())

#Percentage of new loans accepted
accept_rates = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05]
test_pred = pd.DataFrame()

test_pred['prob_default'] = preds_df['prob_default']

thresholds = []
#Number of accepted loans that are bad credit risk
bad_rates = []
num_accepted_loans = []
avg_loan_amnt = []
estimated_value = []

for idx, rate in enumerate(accept_rates):
    thresh = np.quantile(test_pred['prob_default'], rate).round(3)
    
    thresholds.append(thresh)
    
    preds_df['credit_risk'] = preds_df['prob_default'].apply(lambda x : 1 if x > thresh else 0)
    
    accepted_loans = preds_df[preds_df['credit_risk'] == 0]
    
