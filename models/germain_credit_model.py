import pandas as pd
import scipy.stats as stats

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

df = pd.read_csv('german_credit_dataset.csv')

#Only using necessary columns
df = df[['Duration_in_month', 'Credit_amount', 'Installment_rate_in_percentage_of_disposable_income', 'Age_in_years', 'Status_of_existing_checking_account',
         'Credit_history', 'Savings_account/bonds', 'Purpose', 'Property', 'Present_employment_since', 'Housing', 'Other_installment_plans',
         'Other_debtors/guarantors', 'foreign_worker']]

num_df = df.select_dtypes(include = 'number')

