import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df = pd.read_csv("C:/Users/thynnea/OneDrive - Delsys Inc/Documents/GitHub/Credit-Scoring-Model/dataset/cr_loan2_new.csv")

"""
Checking to see how loans accepted vs reject compare. Loans tend to receive a 0 more than a 1. 
"""
df["loan_status"].value_counts().plot(kind='bar', color = ['red', 'blue'])
plt.title("Distribution of Loan Status")
plt.xlabel("Loan Status")
plt.ylabel("Value Counts")
plt.xticks(rotation = 0)
plt.show()

"""
Checking to see if their is a relationship between a person's age and their loan amount.
Appears as though people majority of people request loans in their 20-50. 
"""
plt.scatter(df["loan_amnt"], df["person_age"])
plt.title("Age vs Loan Amount")
plt.xlabel("Person Age")
plt.ylabel("Loan Amount")
plt.show()

"""
Checking to see the frequency of loan percent income. Appears as though loans do tend to take
up 10-20% of a person's income with the highest being over 60%
"""
plt.hist(df["loan_percent_income"])
plt.title("Percentage of Income Going to Loan")
plt.xlabel("Percentage of Loan")
plt.ylabel("Frequency")
plt.show()

"""
Comparing home ownership to loan status.
Rent tends to have 1 have the time while mortage has 0 a majority of the time. 
People who own asks for loans less than both mortgage and rent. 
"""
counts = df.groupby(['person_home_ownership', 'loan_status']).size().unstack(fill_value = 0)
counts.plot(kind = 'bar', figsize = (8, 6))
plt.title("Home Ownership vs Loan Status")
plt.xlabel('Home Ownership')
plt.ylabel('Count')
plt.show()

"""
Comparing employment length days to loan status. Tend to be very similar to each other. 
"""
df.boxplot(column = 'person_emp_length', by = 'loan_status')
plt.title('Loan Status vs Employment Length')
plt.xlabel("Loan Status")
plt.ylabel("Employment Length")
plt.show()

"""
Comparing how many people defaulted on their loans to those who didn't. People tend to
not default on their loans.
"""
df["cb_person_default_on_file"].value_counts().plot(kind = 'bar', color = ['red', 'blue'])
plt.title("Distribution of Faulting on Loan")
plt.xlabel("Default on Loan")
plt.ylabel("Count")
plt.show()

"""
Comparing cred history length to loan status. Very similar for btoh 0 and 1 loan status.
"""
df.boxplot(column = 'cb_person_cred_hist_length', by = 'loan_status')
plt.title("Loan Status vs Credit History Length")
plt.xlabel("Loan Status")
plt.ylabel("Credit History Length")
plt.show()

"""
Comparing loan intent to loan amount. Have similar averages. Home Improvement has
higher maximum and upper quartile than the rest. They all the the same lower quartile. 
"""
df.boxplot(column = 'loan_amnt', by = 'loan_intent')
plt.title("Loan Amount vs Loan Intent")
plt.xlabel("Loan Intent")
plt.ylabel("Loan Amount")
plt.show()

"""
Comparing loan grade to loan status. Higher the loan grade the more liekly it is to receive 0. 
Loan Grade G doesn't have a single 0. 
"""
new_counts = df.groupby(['loan_grade', 'loan_status']).size().unstack(fill_value = 0)
new_counts.plot(kind = 'bar', figsize = (8, 6))
plt.title("Loan grade vs Loan Status")
plt.xlabel("Loan Grade")
plt.legend()
plt.ylabel("Count")
plt.show()

"""
Comparing income to loan status. Higher income is morelikely to receive a loan status of 0. 
"""
df.boxplot(column = 'person_income', by = 'loan_status')
plt.title("Loan Status vs Income")
plt.xlabel("Loan Status")
plt.ylabel("Income")
plt.show()