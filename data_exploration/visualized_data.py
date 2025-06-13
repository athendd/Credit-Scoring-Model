import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("C:/Users/thynnea/OneDrive - Delsys Inc/Documents/GitHub/Credit-Scoring-Model/dataset/cr_loan2_new.csv")

"""
Checking to see how loans accepted vs reject compare
"""
df["loan_status"].value_counts().plot(kind='bar', color = ['red', 'blue'])
plt.title("Distribution of Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Value Counts")
plt.xticks(rotation = 0)
plt.show()

"""
Checking to see if their is a relationship between a person's age and their loan amount
"""
plt.scatter(df["loan_amnt"], df["person_age"])
plt.title("Age vs Loan Amount")
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()

"""
Checking to see the frequency of loan percent income
"""
plt.hist(df["loan_percent_income"])
plt.title("Percentage of Income Going to Loan")
plt.xlabel("Percentage of Loan")
plt.ylabel("Frequency")
plt.show()

"""
Comparing home ownership to loan status
"""
counts = df.groupby(['person_home_ownership', 'loan_status']).size().unstack(fill_value = 0)
counts.plot(kind = 'bar', figsize = (8, 6))
plt.title("Home Ownership vs Loan Status")
plt.xlabel('Home Ownership')
plt.ylabel('Count')
plt.show()

"""
Comparing employment length days to loan status
"""
df.boxplot(column = 'person_emp_length', by = 'loan_status')
plt.title('Loan Status vs Employment Length')
plt.xlabel("Loan Status")
plt.ylabel("Employment Length")
plt.show()

"""
Comparing how many people defaulted on their loans to those who didn't
"""
df["cb_person_default_on_file"].value_counts().plot(kind = 'bar', color = ['red', 'blue'])
plt.title("Distribution of Faulting on Loan")
plt.xlabel("Default on Loan")
plt.ylabel("Count")
plt.show()

"""
Comparing cred history length to loan status
"""
df.boxplot(column = 'cb_person_cred_hist_length', by = 'loan_status')
plt.title("Loan Status vs Credit History Length")
plt.xlabel("Loan Status")
plt.ylabel("Credit History Length")
plt.show()

"""
Comparing loan intent to loan amount
"""
df.boxplot(column = 'loan_amnt', by = 'loan_intent')
plt.title("Loan Amount vs Loan Intent")
plt.xlabel("Loan Intent")
plt.ylabel("Loan Amount")
plt.show()

"""
Comparing loan grade to loan status
"""
new_counts = df.groupby(['loan_grade', 'loan_status']).size().unstack(fill_value = 0)
new_counts.plot(kind = 'bar', figsize = (8, 6))
plt.title("Loan grade vs Loan Status")
plt.xlabel("Loan Grade")
plt.legend()
plt.ylabel("Count")
plt.show()

"""
Comparing income to loan status
"""
df.boxplot(column = 'person_income', by = 'loan_status')
plt.title("Loan Status vs Income")
plt.xlabel("Loan Status")
plt.ylabel("Income")
plt.show()
