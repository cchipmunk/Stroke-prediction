import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import math
import scipy.stats as sts
import seaborn as sns
from itertools import combinations

# Load the data 
data = pd.read_csv("../data/healthcare-dataset-stroke-data.csv")

# Look at data
# Row and column number 
print("rows:", data.shape[0], "columns:", data.shape[1])

# Data types
print(data.dtypes)

# Missing values --> fill in with other data --> HOW - method needed --> multiple linear regression mit feature selection
print(data.isna().sum(axis = 0)) 

# How many patients have all attributes present?
print((data.isna().sum(axis=1) == 0).sum())

# How is the outcome (stroke) distributed?
data["stroke"].value_counts()

# features hypertension and heart_disease: 1 = yes, 0 = no --> categorical?

# Number of groups in the object data frames
print('There are', data.groupby('gender').ngroups,'unique genders in the data.') # Three --> display those?
print('There are', data.groupby('ever_married').ngroups,'unique groups if they ever got married in the data.') # binary
print('There are', data.groupby('work_type').ngroups,'unique work types in the data.') # Five
print('There are', data.groupby('Residence_type').ngroups,'unique residence types in the data.') # binary
print('There are', data.groupby('smoking_status').ngroups,'unique smoking in the data.') #Four

# Show different groups in features
print("Gender", data["gender"].unique())
print("Ever married?", data["ever_married"].unique())
print("Work type", data["work_type"].unique())
print("Residence type", data["Residence_type"].unique())
print("Smoking status", data["smoking_status"].unique())

# Object type --> no ordinal data (Smoking status could be ...)

# Object --> categorical
data["gender"] = pd.Categorical(data["gender"])
data["ever_married"] = pd.Categorical(data["ever_married"])
data["work_type"] = pd.Categorical(data["work_type"])
data["Residence_type"] = pd.Categorical(data["Residence_type"])
data["smoking_status"] = pd.Categorical(data["smoking_status"])

print(data.dtypes)

# Summarising data in more detail
# Distribution of continuous variables 
for var in data.dtypes[data.dtypes == "float64"].index:
    print(f"Shapiro-Wilk for {var}, p-value: {sts.shapiro(data[var]).pvalue: .10f}")

vars = data.dtypes[data.dtypes == "float64"].index.tolist()
fig, axs = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 8))
titles = {
    "age": "Age",
    "avg_glucose_level": "Average glucose level",
    "bmi": "Body Mass Index",
}
xlabels = {
    "age": "Age (years)",
    "avg_glucose_level": "Average glucose level (mg/dl)",
    "bmi": "Body Mass Index (kg/m**2)",
}
for i, ax in enumerate(axs.flatten()):
    sns.histplot(
        x=vars[i], 
        data=data, 
        kde=True, 
        ax=ax
    )
    ax.set_title(f"Distribution of\n{titles[vars[i]]}", fontsize=9)
    ax.set_xlabel(xlabels[vars[i]], fontsize=8)
    if i not in [0, 3]:
        ax.set_ylabel(None)
fig.tight_layout()
plt.show()


# Perform one-hot encoding for all categorical data

