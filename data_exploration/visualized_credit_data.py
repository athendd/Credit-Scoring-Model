import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2_contingency
import scipy.stats

df = pd.read_csv('german_credit_dataset.csv')

#credit risk 1 is good and 0 is bad

print(df['credit_risk'].dtype)

"""
Comparing the amount of good credit risk to bad credit risk.
A lot more 1s than 0s
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
Bad credit risk tend to have loans that last longer on average. 
Bad credit risk tend to have larger loan amounts.
Good credit risk have loan make up less of their disposable income.
Present residence doesn't seem to affect outcome.
Older people tend to be more of a good credit risk (not by much though).
Numer of existing credits at this bank doesn't affect the outcome. 
Number of people being liable to provide maintenance for doesn't affect the outcoe.
"""

"""
for col in df_numerical.columns:
    df.boxplot(column = col, by = 'credit_risk')
    plt.title(f'Credit Risk vs {col}')
    plt.xlabel("Credit Risk")
    plt.ylabel(col)
    plt.show()
"""
    
df_non_numerical = df.select_dtypes(exclude = ['number'])

"""
Checking the distributions and categories of categorical data
status_of_existing_checking_account: 
    no_checking_account: nearly 400
    less than 0: over 250
    less than 116: over 250
    less than or equal to 116: over 50
    
credit_history:
    exiting credit paid back till now: over 500
    critical account/other credits existing: 300
    delay in paying off in past: 100
    all credits at bank paid back: around 50
    all credit paid back: around 50
    
purpose: vacation column has no values but that was noted in document
    radio/tv: over 250
    new car: under 250
    furniture/equipment: under 200
    used car: over 100
    business: around 100
    education: over 50
    repairs: around 25
    domestic appliances: under 25
    others: under 25
    restraining: under 25
    
savings_account/bonds:
    less than 116: around 600
    unkown/no savings: under 200
    less than 291: over 100
    less than 582: under 100
    greater than 582: under 50
    
employment:
    less than 4 years: under 350
    greater than 7 years: around 250
    less than 7 years: over 150
    less than 1 year: over 150
    unemployed: over 50
    
status and sex: no values for female single
    male single: over 500
    female divocred separated married: around 300
    male married/widowed: under 100
    male: divorced/separated: around 50

other debators:
    none: over 800
    guarantor: under 100
    co-applicant: under 100
    
property:
    car or other: over 300
    real estate: over 250
    building society/life insurance: under 250
    unkown/no property: around 150
    
other installment plans: no values for bank
    none: under 140
    bank: around 50
    stores: over 40

housing:
    own: around 700
    rent: under 200
    for free: around 100

job:
    skilled employee: over 600
    unskilled employee: around 200
    highly skilled employee/self-employed: around 150
    unemployed: under 25
    
telephone:
    none: under 600
    yes: around 400

foreign worker:
    yes: under 1000
    no: under 50
"""

"""
for col in df_non_numerical.columns:
    df_non_numerical[col].value_counts().plot(kind='bar')
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Value Counts")
    plt.xticks(rotation = 0)
    plt.show()
"""



df_non_numerical['credit_risk'] = df['credit_risk']

"""
Comparing all non-numerical columns in the dataframe to credit risk.
No checking account is very good credit risk. Less than zero is nearly equal good or bad. The rest are good credit risks.  
Existing credit paid back until now and other credits existing not at bank are way more liekly to be good. All credits paid back and all credits paid back at this bank are slightly more likely to be bad. 
Check three above again
television, used car, equipment, business, and new car is seen way more as a good credit risk. Repairs, education, appliances, others, and training are seen only slightly as good credit risks. 
Saving account more than 582, less than 582, and none are seen as very good credit risk. more than 116 and less than 291 are seen as slightly good credit risks.
more than 7 years, less than 7 years, more than 4 years of employment are seen as very good credit risks. Unemployed and less than 1 year are seen as slightly good credit risks. 
male: single and male: married/widowed is seen as very good credit risk. Rest are seen as slightly good credit risks
No other debators are seen as very good risk while rest are seen only as slightly good credit risk. 
real estate, car other other, saving agreement/life insruance are seen as very good credit risks. Rest is seen as slightly good credit risk
No other installments are seen as very good credit risks while stores are seen as slightly good.
Owning a property is seen as very good credit risk while rest are seen as slightly good credit risk. 
self employed. highly skilled and skilled employee are seen as very good credit risks while unemployed and unskilled employee are seen as slightly good credit risks. 
Having a telephone is seen as a better credit risk than not having one. 
Non foreign workers are seen as better credit risk than foreign workers. 
"""

"""
for col in df_non_numerical.columns:
    counts = df_non_numerical.groupby([col, 'credit_risk']).size().unstack(fill_value = 0)
    counts.plot(kind = 'bar', figsize = (8, 6))
    plt.title(f"{col} vs Credit Risk")
    plt.xlabel(col)
    plt.legend()
    plt.ylabel("Count")
    plt.show()
"""

significance_threshold = 0.05

"""
Chi square test to compare categorical columns to credit_risk
1. Status_of_existing_checking_accountL 123.7
2. Credit_history: 61.6
3. Savings_account/bonds: 36
4. Purpose: 33.3
5. Property: 23.7
6. Present_employment_since: 18.3
6. Housing: 18.1
7. Other_installment_plans: 12.8
8. Personal_status_and_sex: 9.6
9. Other_debators/guarantors: 6.6
10. foreign_workers: 5.8
"""

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

"""
Hypothesis Testing to compare numerical variables to credit_risk
1. Duration_in_month: 6.9
2. Credit_amount: 4.9
3. Installment_rate_in_percentage_of_dispoable_income: 2.2
4. Age_in_years: -2.8
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


