import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import math
import scipy.stats as sts
import seaborn as sns
from itertools import combinations
from statistics import mean
import warnings

warnings.filterwarnings("ignore")



# Load the data 
df_path = "../data/Truncated_data/Stroke_data.csv"

data = pd.read_csv(df_path)

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

""" BMI estimation code """

def bmi_scores(model, X_train, y_train, X_test, y_test, y_mean):
    ### Only use for BMI estimation scores ###

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = (sum((y_test - y_pred_test)**2)/len(y_test))**0.5

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = (mean_squared_error(y_train, y_pred_train, squared=True))**0.5

    r2_mean = r2_score(y_test, y_mean)
    rmse_mean = (sum((y_test - y_mean)**2)/len(y_test))**0.5

    print('Training set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_train, rmse_train))
    print('Test set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_test, rmse_test))
    print('Score using means (no model): R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_mean, rmse_mean))

def bmi_train(data):
    print()
    df = data.loc[data['bmi'].notna()] #Dropping values without labels
    df = df.drop(['stroke'], axis = 1) # We do not want strokes in our model, as we will be estimating it later (Data leakage)

    encoded_df = df #One hot encoding the columns
    for i in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        encoded_df = pd.get_dummies(encoded_df, columns = [i], prefix = [i], drop_first = True )

    #dropping irrelevant Columns for our model (feature selection)
    encoded_df = encoded_df.drop(['Residence_type_Urban', 'gender_Male', 'id', 'work_type_Private', 'work_type_Self-employed'], axis=1)

    #Creating Labels and Features
    X = encoded_df.drop('bmi', axis=1)
    y = encoded_df['bmi']

    #Seperating into train / test data-set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)
    
    #Scaling the data
    global scaler
    global num_cols

    scaler = RobustScaler() #Using robust because data not normally distributed
    num_cols = ['age', 'avg_glucose_level']
    X_train_scaled, X_test_scaled = X_train, X_test
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols]) 
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    #Training the model
    global LR

    LR = LinearRegression()
    LR.fit(X_train_scaled, y_train)
    
    #Creating a Model which calculates the mean for every estimated value as comparaison
    y_mean = [mean(y)] * len(y_test)

    #Evaluating performance
    bmi_scores(LR, X_train, y_train, X_test, y_test, y_mean)

def bmi_main():
    
    #importing Data, seperating the data which needs truncation from the known data
    #LEAVE THE PATH HARD CODED, NOT THE SAME AS "df_path"
    data = pd.read_csv('../data/ORIGINAL_DATA/stroke_dataset/healthcare-dataset-stroke-data.csv')

    #One hot encoding the data 
    encoded_df = data
    for i in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        encoded_df = pd.get_dummies(encoded_df, columns = [i], prefix = [i], drop_first=True)

    #dropping label
    X = encoded_df.drop(['bmi'], axis= 1) 

    #Dropping various other Features (Feature selection), as well as Stroke Feature (Which we want to estimate later)
    X = X.drop(['Residence_type_Urban', 'gender_Male', 'work_type_Private', 'work_type_Self-employed', 'stroke', 'id'], axis = 1) 

    #Selecting data to truncate
    X = X.loc[data['bmi'].isna()]

    #Training model
    bmi_train(data)

    #Scaling data to truncate (Using same Scaler as in training model)
    X_scaled = X
    X_scaled[num_cols] = scaler.transform(X[num_cols])

    #Applying model on data to estimate
    y = LR.predict(X_scaled)

    # Saving data in the panda table and saving it as csv:
    truncation = data.loc[data['bmi'].isna()]
    truncation['bmi'] = y
    data.loc[data['bmi'].isna()] = truncation

    
    data.to_csv('../data/Truncated_data/Stroke_data.csv')
    print()
    print('Truncated data saved to: ../data/Truncated_data/Stroke_data.csv')
    print()


    #Manual test to check if the saving into truncation effectively worked
    """
    columns = data.columns
    for i in columns:
        print(i, ": ", data[i].isna().sum() ,"missing data" )
    
    print()
    print(data.loc[data['bmi'].isna()]) #Checking if there are no empty values

    test = X_scaled.iloc[1:10]
    print(test[['hypertension', 'heart_disease']])
    print(truncation.iloc[1:10])
    """
    
def Kolmogorov_Smirnov(df):
    # Kolgorov_Smirnov test for normality with transformations
    for i in ['age', 'avg_glucose_level', 'bmi']:
        print(f"{i} - Kolgorov Smirnov test for normality:")
        res = stats.kstest(df[i], stats.norm.cdf)
        print(f"probability that {i} is normaly distributed with no transformation = {1 - res.statistic}")
        res = stats.kstest((df[i])**0.5, stats.norm.cdf)
        print(f"probability that {i} is normaly distributed with sqrt transformation = {1 - res.statistic}")
        res = stats.kstest(np.log10(df[i]), stats.norm.cdf)
        print(f"probability that {i} is normaly distributed with log transformation = {1 - res.statistic}")
        print()

