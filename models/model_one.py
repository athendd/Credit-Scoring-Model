import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, classification_report
import xgboost as xgb
import numpy as np
import shap
from imblearn.over_sampling import ADASYN

"""
Find outliers using the z score
"""
def find_outliers_zscore(df):
    outlier_indices = {}
    for col in df.columns:
        z_score = stats.zscore(df[col])
        outliers = df[abs(z_score) > 3].index.tolist()
        outlier_indices[col] = outliers
    
    return outlier_indices

"""
Creates a boxplot of each numerical column
to find outliers in that column
"""  
def plot_boxplot(df):
    for col in df.columns:
        plt.figure(figsize = (6,4))
        sns.boxplot(x = df[col])
        plt.title(f"Box Plot of {col}")
        plt.show()
    
    
df = pd.read_csv("C:/Users/thynnea/OneDrive - Delsys Inc/Documents/GitHub/Credit-Scoring-Model/dataset/cr_loan2_new.csv")

#Creating a dataframe with only numerical columns
num_df = df.select_dtypes(include=['number'])

#Obtaining number of outliers for each numerical column
outliers = find_outliers_zscore(num_df)

for key in outliers.keys():
    num_outliers = len(outliers[key])

pd.crosstab(df["loan_status"], df["person_home_ownership"], values = df["person_emp_length"], aggfunc="max")

#Removing outliers
df = df.drop(df[df["person_emp_length"] > 60].index)

#Removing more outliers (Did this become there is one that is age 120)
df = df.drop(df[df["person_age"] > 100].index)

#Getting count of null value for each column
total_null_values = df.isnull().sum()

#Replace all null value with median value for employment length column
df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

#Dropping rows with null values in interest rate column
df = df.drop(df[df["loan_int_rate"].isnull()].index)

df = df.drop(columns = ['loan_grade'], axis = 1)

#Convert categorical data to numerical data through one-hot encoding
df = pd.get_dummies(df, columns = ["person_home_ownership", "cb_person_default_on_file", "loan_intent"])

#Checking the correlation between each column in the dataframe
correlation_matrix = df.corr()

"""
sns.heatmap(correlation_matrix, annot = True, cmap = 'coolwarm', fmt = '.2f')
plt.show()
"""

"""
strong positive correlations: person_age and cb_person_cred_hist_length
strong negative correlations: loan_int_rate and loan_grade_oridanl, home_ownership_MORTGAGE and home_ownership_RENT 
"""

#Creating training and testing datasets
x = df.drop("loan_status", axis = 1)
y = df[["loan_status"]]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

#Flatten y_train into a 1d array
y_train_1d = np.ravel(y_train)

#Oversample default to deal with the imbalance between default and non-default
adasyn = ADASYN(random_state=42)
x_train_over, y_train_over = adasyn.fit_resample(x_train, y_train_1d)

#Hyperparameters
params = {
    "n_estimators": [50, 100, 150, 200, 250, 300],
    "max_depth": [3, 5, 7, 9],
    "learning_rate": [0.01, 0.05, 0.1, 0.4, 0.3, 0.5],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "reg_alpha": [0, 0.5, 1],
    "reg_lambda": [1, 2, 3, 4, 5]
}

#Creating classifier model
xgb_clf = xgb.XGBClassifier()
                
#Fit data to each classification model  and evaluate each model
xgb_clf.fit(x_train_over, y_train_over)
y_pred = xgb_clf.predict(x_test)
print(f"The {xgb_clf.__class__.__name__} has an Accuracy: {accuracy_score(y_test, y_pred):.4f} , and a Precision of: {precision_score(y_test, y_pred):.4f} ")

threshold = 0.5

preds = xgb_clf.predict_proba(x_test)

preds_df = pd.DataFrame(preds[:, 1], columns = ['prob_default'])

#Assign loan status based on threshold
preds_df["loan_status"] = preds_df["prob_default"].apply(lambda x: 1 if x > threshold else 0)

#Print each loan status (0 for non default and 1 for default)
print(preds_df["loan_status"].value_counts()) 

target_names = ['Non-Default', 'Default'] 
print(classification_report(y_test, preds_df["loan_status"], target_names=target_names)) 

#Confusion matrix of actual vs predicted values
cm = confusion_matrix(y_test, preds_df['loan_status']) 

class_names = ['Non-Default', 'Default']

#Display the confusion matrix
display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
display_cm.plot()  
plt.show()

#Get probabilities of the positive class
prob_default = preds[:, 1]

#Calculate roc metrics
fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)

#Calculate area under auc curve
roc_auc = roc_auc_score(y_test, prob_default)

plt.plot(fallout, sensitivity,
            linewidth = 2.5, label = f'AUC: {roc_auc:.2f}')
    
#Plot the diagnol line
plt.plot([0,1], [0,1], linestyle = '--', color = 'black', label = 'Random Classifier')
plt.title("ROC Curve Comparison for Different Models", fontsize=18, fontweight='bold')
plt.xlabel('Fall-out (False Positive Rate)', fontsize=14)
plt.ylabel('Sensitivity (True Positive Rate)', fontsize=14)

plt.grid(True)

plt.legend(loc='lower right', fontsize=12)

plt.tight_layout()

plt.show()

"""

explainer = shap.TreeExplainer(xgb_clf)

shap_values = explainer.shap_values(x_test)

feature_names = x.columns

print(feature_names)

#Shows the mean absolute value for each feature
shap.summary_plot(shap_values, x_test, plot_type = 'bar', feature_names = feature_names)
plt.show()

shap.summary_plot(shap_values, x_test, feature_names = feature_names)
plt.show()
"""