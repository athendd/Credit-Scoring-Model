import sklearn
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay, auc, roc_curve, classification_report, precision_recall_fscore_support
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np

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
    
"""
person_age: 558
person_income: 233
loan_amnt: 336
loan_percent_income: 335
cb_person_cred_hist_length: 283
Rest have none
"""

pd.crosstab(df["loan_status"], df["person_home_ownership"], values = df["person_emp_length"], aggfunc="max")

#Removing outliers
df = df.drop(df[df["person_emp_length"] > 60].index)


#Removing more outliers (Did this become there is one that is age 120)
df = df.drop(df[df["person_age"] > 100].index)

#Getting count of null value for each column
total_null_values = df.isnull().sum()

"""
loan interest rate has a lot of null values (3115) and
so does person_emp_length (895) while the rest have no null
values
"""

#Replace all null value with median value for employment length column
df["person_emp_length"] = df["person_emp_length"].fillna(df["person_emp_length"].median())

#Dropping rows with null values in interest rate column
df = df.drop(df[df["loan_int_rate"].isnull()].index)

# *Differed from example in label encoding

#Convert person_home_ownership and loan_intent to numerical data through one-hot encoding
df = pd.get_dummies(df, columns = ["person_home_ownership", "loan_intent"])

#Converting cb_person_default_on_file to numerical data through binary classification
df["cb_person_default_on_file_binary"] = (df["cb_person_default_on_file"] == "Y").astype(int)

#Convert loan_grade to numerical data through ordinal encoding
grade_order = {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
df["loan_grade_ordinal"] = df["loan_grade"].map(grade_order)

#Get rid of all non-numerical columns from the dataset
df = df.drop(columns = ["cb_person_default_on_file", "loan_grade"])

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

#Standardize the training and testing datasets
scaler = StandardScaler()

x_train_st = scaler.fit_transform(x_train)
x_test_st = scaler.fit_transform(x_test)

#Creating classifier models
log_clf = LogisticRegression(solver = 'lbfgs', max_iter = 1000)
rnd_clf = RandomForestClassifier()
svm_clf = SVC()
knn_clf = KNeighborsClassifier()
grd_clf = GradientBoostingClassifier()
xgb_clf = xgb.XGBClassifier()

voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf), 
        ('rf', rnd_clf), 
        ('xgb', xgb_clf)
    ],
    voting='soft'
)

#Fit data to each classification model, flatten y_train into a 1d array, and evaluate each model
for clf in (log_clf, rnd_clf, svm_clf, knn_clf, grd_clf, xgb_clf, voting_clf):
    clf.fit(x_train_st, np.ravel(y_train))
    y_pred = clf.predict(x_test_st)
    print(f"The {clf.__class__.__name__} has an Accuracy: {accuracy_score(y_test, y_pred):.4f} , and a Precision of: {precision_score(y_test, y_pred):.4f} ")

"""
Logisitc Regression: 0.85, 0.76 which are both good 
Random Forest Classifier: 0.93, 0.94 which are both great
SVC: 0.9, 0.93 which are both great
KNN: 0.88, 0.87 which are both really good
Gradient Boosting: 0.92, 0.92 which are both great
XGB: 0.89, 0.84 which are both good
Voting: 0.92, 0.92 which are both great
Top models are voting, gradient boosting, random forest, svc, and knn
"""

plotting_colors = ['orange', 'blue', 'green', 'purple', 'red', 'brown']
model_list = []
fallout_list = []
sensitivity_list = []
thresholds_list = []
roc_auc_list = []
#Contains probabilities and predictions for each model
preds_df_all = pd.DataFrame()

for idx, clf in enumerate((rnd_clf, xgb_clf, voting_clf, log_clf)):
    model_list.append(clf.__class__.__name__)
    
    clf.fit(x_train_st, np.ravel(y_train))
    
    preds = clf.predict_proba(x_test_st)
    
    preds_df = pd.DataFrame(preds[:, 1], columns = ['prob_default'])
    
    preds_df_all[f"prob_default {model_list[idx]}"] = preds_df['prob_default'] 
    
    threshold = 0.5
    
    #Assign loan status based on threshold
    preds_df["loan_status"] = preds_df["prob_default"].apply(lambda x: 1 if x > threshold else 0)
    
    preds_df_all[f"loan_status {model_list[idx]}"] = preds_df['loan_status']

    #Print each loan status (0 for non default and 1 for default)
    print(preds_df["loan_status"].value_counts()) 
    
    print(f"The Report from {model_list[idx]}:")
    target_names = ['Non-Default', 'Default'] 
    print(classification_report(y_test, preds_df["loan_status"], target_names=target_names)) 

    #Confusion matrix of actual vs predicted values
    cm = confusion_matrix(y_test, preds_df['loan_status']) 

    class_names = ['Non-Default', 'Default']
    
    #Display the confusion matrix
    display_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)  # Configure the display
    display_cm.plot()  
    plt.show()
    
    #Get probabilities of the positive class
    prob_default = preds[:, 1]
    
    #Calculate roc metrics
    fallout, sensitivity, thresholds = roc_curve(y_test, prob_default)
    
    #Calculate area under auc curve
    roc_auc = roc_auc_score(y_test, prob_default)
    
    fallout_list.append(fallout) 
    sensitivity_list.append(sensitivity) 
    thresholds_list.append(thresholds) 
    roc_auc_list.append(roc_auc) 
    
"""
Random Forest Classifier:
f1-score non-default: 0.96
f1-score default: 0.82
accuracy: 0.93
XGB Classifier:
f1-score non-default: 0.93
f1-score default: 0.74
accuracy: 0.90
Voting Classifier:
f1-score non-default: 0.95
f1-score default: 0.79
accuracy: 0.92
Logistic Regression:
f1-score non-default: 0.91
f1-score default: 0.61
accuracy: .86
"""

plt.figure(figsize = (12,8))

for idx in range(len(model_list)):
    plt.plot(fallout_list[idx], sensitivity_list[idx], color = plotting_colors[idx],
             linewidth = 2.5, label = f'{model_list[idx]} (AUC: {roc_auc_list[idx]:.2f})')
    
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
Voting Classifier and Random Forest Classifier stand out as the two best models.
XGB would've done better than those two if all the categorical columns had been one-hot encoded. 
"""
