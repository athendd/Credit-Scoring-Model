import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:/Users/thynnea/OneDrive - Delsys Inc/Documents/GitHub/Credit-Scoring-Model/dataset/cr_loan2_new.csv")

"""
Checking to see how loans accepted vs reject compare.
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


