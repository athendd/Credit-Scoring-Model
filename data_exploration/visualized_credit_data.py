import pandas as pd

# Define column names based on dataset documentation
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
installment_plans_dic = {'A141': 'bank', 'A142': 'stores', 'A141': 'none'}
property_dic = {'A121': 'real estate', 'A122': 'building society saving agreement/life insurance',
                'A123': 'car or other', 'A124': 'unkown/no property'}
other_debators_dic = {'A101': 'none', 'A102': 'co-applicant', 'A103': 'guarantor'}
status_sex_dic = {'A91': 'male : divocred/separated', 'A92': 'female : divorced/separated/married',
                  'A93': 'male : single', 'A94': 'male : married/widowed', 'A95': 'female : single'}
employment_dic = {'A71': 'unemployed', 'A72':'less than 1 year', 'A73': 'less than 4 years', 'A74': 'less than 7 years', 'A75': 'greater than 7 years'}
saving_accounts_dic = {'A61': 'less than 100', 'A62': 'less than 500', 'A63': 'less than 1000', 'A64': 'greater than 1000', 'A65': 'unkown/no savings account'}


#credit_risk 1 = Good and 2 = Bad

# Load the data
df = pd.read_csv('C:/Users/thynnea/OneDrive - Delsys Inc/Documents/GitHub/Credit-Scoring-Model/dataset/statlog+german+credit+data/german.data', sep=' ', header=None, names=column_names)

# Convert target values to binary: 1 = good, 2 = bad
df['credit_risk'] = df['credit_risk'].map({1: 1, 2: 0})

# Optional: quick preview
print(df["Number_of_people_being_liable_to_provide_maintenance_for"])
