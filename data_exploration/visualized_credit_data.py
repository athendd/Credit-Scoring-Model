import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
import scipy.stats

df = pd.read_csv('german_credit_dataset.csv')

print(df['credit_risk'].dtype)

"""
Comparing the amount of good credit risk to bad credit risk.
"""
df["credit_risk"].value_counts().plot(kind='bar', color = ['red', 'blue'])
plt.title("Distribution of Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Value Counts")
plt.xticks(rotation = 0)
plt.show()

df_numerical = df.select_dtypes(include = ['number'])

"""
Comparing all numerical values to the target variable (credit_risk).
"""
for col in df_numerical.columns:
    df.boxplot(column = col, by = 'credit_risk')
    plt.title(f'Credit Risk vs {col}')
    plt.xlabel("Credit Risk")
    plt.ylabel(col)
    plt.show()

    
df_non_numerical = df.select_dtypes(exclude = ['number'])

for col in df_non_numerical.columns:
    df_non_numerical[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Value Counts")
    plt.xticks(rotation = 0)
    plt.show()

df_non_numerical['credit_risk'] = df['credit_risk']

for col in df_non_numerical.columns:
    counts = df_non_numerical.groupby([col, 'credit_risk']).size().unstack(fill_value = 0)
    counts.plot(kind = 'bar', figsize = (8, 6))
    plt.title(f"{col} vs Credit Risk")
    plt.xlabel(col)
    plt.legend()
    plt.ylabel("Count")
    plt.show()

significance_threshold = 0.05

"""
Chi square test to compare categorical columns to credit_risk
"""
df_non_numerical['credit_risk'] = df_non_numerical['credit_risk'].map({1: '1', 0: '0'})

chi_test_results = {}

for col in df_non_numerical:
    contingency_table = pd.crosstab(df_non_numerical[col], df_non_numerical['credit_risk'])

    chi, p, df, expected = chi2_contingency(contingency_table)
    
    if p < significance_threshold:
        chi_test_results[col] = chi
        
sorted_dic = dict(sorted(chi_test_results.items(), key = lambda item: item[1], reverse = True))
print(sorted_dic)

"""
Two sample t-tests to compare numerical variables to credit_risk
"""

hypothesis_test_results = {}
    
for col in df_numerical.columns:
    group_0 = np.array(df_numerical[df_numerical['credit_risk'] == 0][col])
    group_1 = np.array(df_numerical[df_numerical['credit_risk'] == 1][col])
    
    t_statistic, p_value = scipy.stats.ttest_ind(group_0, group_1)
    
    if p_value < significance_threshold:
        hypothesis_test_results[col] = t_statistic
        
sorted_hypothesis_test_results = dict(sorted(hypothesis_test_results.items(), key = lambda item: item[1], reverse = True))
print(sorted_hypothesis_test_results)


