import pandas as pd

column_names = [
    'Status_of_existing_checking_account', 'Duration_in_month', 'Credit_history',
    'Purpose', 'Credit_amount', 'Savings_account/bonds', 'Present_employment_since',
    'Installment_rate_in_percentage_of_disposable_income', 'Personal_status_and_sex',
    'Other_debtors/guarantors', 'Present_residence_since', 'Property', 'Age_in_years',
    'Other_installment_plans', 'Housing', 'Number_of_existing_credits_at_this_bank',
    'Job', 'Number_of_people_being_liable_to_provide_maintenance_for',
    'Telephone', 'foreign_worker', 'credit_risk'
]

Telephone_dic = {'A191': 'none', 'A192': 'yes'}
foreign_worker_dic = {'A201': 'Y', 'A202': 'N'}
job_dic = {'A171': 'unemployed', 'A172': 'unskilled', 'A173': 'skilled employee', 'A174': 'self_employed/highly qualified employee'}
housing_dic = {'A151': 'rent', 'A152': 'own', 'A153': 'for free'}
installment_plans_dic = {'A141': 'bank', 'A142': 'stores', 'A143': 'none'}
property_dic = {'A121': 'real estate', 'A122': 'building society saving agreement/life insurance',
                'A123': 'car or other', 'A124': 'unknown/no property'}
other_debators_dic = {'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'}
status_sex_dic = {'A91': 'male : divorced/separated', 'A92': 'female : divorced/separated/married',
                  'A93': 'male : single', 'A94': 'male : married/widowed', 'A95': 'female : single'}
employment_dic = {'A71': 'unemployed', 'A72':'less than 1 year', 'A73': 'less than 4 years', 'A74': 'less than 7 years', 'A75': 'greater than 7 years'}
saving_accounts_dic = {'A61': 'less than 116', 'A62': 'less than 291', 'A63': 'less than 582', 'A64': 'greater than 582', 'A65': 'unknown/no savings account'}
purpose_dic = {'A40': 'new car', 'A41': 'used car', 'A42': 'furniture/equipment', 'A43': 'radio/television', 'A44': 'domestic appliances',
               'A45': 'repairs', 'A46': 'education', 'A47': 'vacation', 'A48': 'retraining', 'A49': 'business', 'A410': 'others'}
credit_history_dic = {'A30': 'no credits taken/all credit paid back', 'A31': 'all credits at this bank paid back', 
                      'A32': 'existing credits paid back till now', 'A33': 'delay in paying off in the past',
                      'A34': 'critical account/ other credits existing'}
existing_checking_accounts_dic = {'A11': 'less than 0', 'A12': 'less than 116', 'A13': 'greater than or equal to 116', 'A14': 'no checking account'}

#credit_risk 1 = Good and 2 = Bad

#Load the data
df = pd.read_csv('C:/Users/thynnea/OneDrive - Delsys Inc/Documents/GitHub/Credit-Scoring-Model/dataset/statlog+german+credit+data/german.data', sep=' ', header=None, names=column_names)

#Convert target values to binary: 1 = good, 2 = bad
df['credit_risk'] = df['credit_risk'].map({1: 1, 2: 0})

df['Status_of_existing_checking_account'] = df['Status_of_existing_checking_account'].map(existing_checking_accounts_dic)
df['Credit_history'] = df['Credit_history'].map(credit_history_dic)
df['Purpose'] = df['Purpose'].map(purpose_dic)
df['Savings_account/bonds'] = df['Savings_account/bonds'].map(saving_accounts_dic)
df['Present_employment_since'] = df['Present_employment_since'].map(employment_dic)
df['Personal_status_and_sex'] = df['Personal_status_and_sex'].map(status_sex_dic)
df['Other_debtors/guarantors'] = df['Other_debtors/guarantors'].map(other_debators_dic)
df['Property'] = df['Property'].map(property_dic)
df['Other_installment_plans'] = df['Other_installment_plans'].map(installment_plans_dic)
df['Housing'] = df['Housing'].map(housing_dic)
df['Job'] = df['Job'].map(job_dic)
df['Telephone'] = df['Telephone'].map(Telephone_dic)
df['foreign_worker'] = df['foreign_worker'].map(foreign_worker_dic)
df.to_csv('german_credit_dataset.csv', index = False)