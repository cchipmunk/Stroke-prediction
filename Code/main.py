import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats as sts
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial.distance import euclidean, cityblock, minkowski
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import math
import seaborn as sns
from itertools import combinations
from itertools import cycle, islice
from statistics import mean
import warnings
import time



warnings.filterwarnings("ignore")


"""Data Preprocessing"""
# Load the data 
df_path = "../Data/Truncated_data/Stroke_data.csv"

data = pd.read_csv(df_path)

# Look at data
# Row and column number 
print("rows:", data.shape[0], "columns:", data.shape[1])

# Data types
print(data.dtypes)

# Missing values --> fill in with other data --> see bmi
print(data.isna().sum(axis = 0)) 

# How many patients have all attributes present?
print((data.isna().sum(axis=1) == 0).sum())

# How is the outcome (stroke) distributed?
data["stroke"].value_counts()

# Number of groups in the object data frames
print('There are', data.groupby('gender').ngroups,'unique genders in the data.') # Three --> display those?
print('There are', data.groupby('ever_married').ngroups,'unique groups if they ever got married in the data.') # binary
print('There are', data.groupby('work_type').ngroups,'unique work types in the data.') # Five
print('There are', data.groupby('Residence_type').ngroups,'unique residence types in the data.') # Binary
print('There are', data.groupby('smoking_status').ngroups,'unique smoking in the data.') #Four
print('There are', data.groupby('hypertension').ngroups,'unique hypertensions in the data.') # Two
print('There are', data.groupby('heart_disease').ngroups,'unique heart diseases in the data.') # Twp

# Show different groups in features
print("Gender", data["gender"].unique())
print("Ever married?", data["ever_married"].unique())
print("Work type", data["work_type"].unique())
print("Residence type", data["Residence_type"].unique())
print("Smoking status", data["smoking_status"].unique())
print("Hypertension?", data["hypertension"].unique())
print("Heart disease?", data["heart_disease"].unique())
print("Stroke?", data["stroke"].unique())

# Object type --> no ordinal data

# Object --> categorical
data["gender"] = pd.Categorical(data["gender"])
data["ever_married"] = pd.Categorical(data["ever_married"])
data["work_type"] = pd.Categorical(data["work_type"])
data["Residence_type"] = pd.Categorical(data["Residence_type"])
data["smoking_status"] = pd.Categorical(data["smoking_status"])
data["hypertension"] = pd.Categorical(data["hypertension"])
data["heart_disease"] = pd.Categorical(data["heart_disease"])

print(data.dtypes)

# Rename categories of hypertensions, heart_disease and stroke
data["hypertension"] = data["hypertension"].cat.rename_categories(
    {0: "Yes", 1: "No"}
)
data["heart_disease"] = data["heart_disease"].cat.rename_categories(
    {0: "Yes", 1: "No"}
)

# How many "Other" in gender are there? --> only 1 --> Reason: see report
print("Value counts", data["gender"].value_counts())
data.drop(data[data["gender"] == "Other"].index, axis = 0, inplace = True)
data["gender"] = data["gender"].astype('object')
data["gender"] = pd.Categorical(data["gender"])
data.reset_index(drop = True, inplace = True)
print("Value counts", data["gender"].value_counts())

# Are there any duplicates? --> NO
print(dict(data.duplicated(subset = ["id"], keep = False)) == True)


### Summarising data in more detail ###
# Distribution of continuous variables 
# Does not work as we have too many features --> use another test
"""
for var in data.dtypes[data.dtypes == "float64"].index:
    print(f"Shapiro-Wilk for {var}, p-value: {sts.shapiro(data[var]).pvalue: .10f}")
"""
# Use this instead
for var in data.dtypes[data.dtypes == "float64"].index:
    print(f"Normality test for {var}, p-value: {sts.normaltest(data[var]).pvalue: .10f}")

# Distribution of continuous variables
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

### Demographc data insights ###
dem_data = data.copy()
dem_data["stroke"] = pd.Categorical(dem_data["stroke"])
dem_data["stroke"] = dem_data["stroke"].cat.rename_categories(
    {0: "Yes", 1: "No"}
)

# Age distribution
fig, ax = plt.subplots(1, 2, figsize = (12,6))

# Plot the age distribution separated by outcome
age = sns.histplot(dem_data, x = "age", binwidth = 5, hue = "stroke", ax = ax[0])
label_axes = age.set(xlabel = "Age [year]", ylabel = "Number of Patients", title = "Age Distribution by Stroke Outcome")

# Plot age dispersion separated by gender
age_range = sns.boxplot(dem_data, y = "age", x = "gender", hue = "stroke", ax = ax[1], width = 0.4)
age_range = age_range.set(ylabel = "Age [year]", xlabel='Gender', title= 'Age Range split by Gender and Stroke Outcome')
plt.show()


### One-hot endoding --> all use the same --> do not define in any other function ###


### Test and training data set split ###



""" Correlation estimation code """
# show correlation --> needed?
corr_plot= sns.pairplot(data=data, x_vars=vars, y_vars=vars)
corr_plot.fig.suptitle("Correlation between continuous variables", y = 1)
plt.show()

def estimate_correlation(data):
    num_cols = ['age', 'avg_glucose_level', 'bmi']

    for i in num_cols:
        print(i)
        pear_v = sts.pearsonr(data['stroke'], data[i])
        print(f"""Pearson coefficient for {i} :
            value = {pear_v[0]}
            p = {pear_v[1]}""")
        if pear_v[1] <= 0.05/10: # With Bonferoni correcture
            print("            *** Heavily Correlated")
        elif pear_v[1] >= 0.05/10 and p <= 0.05:
            print("          * correlated")
        else:
            print("            No significant correlation")
            print()
        
    for i in ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]:
        cont_table = pd.crosstab(data[i], data['stroke'])
        p = sts.chi2_contingency(cont_table)[1]
        s = sts.chi2_contingency(cont_table)[0]

        print(f"""Chi_sq for {i} :
            Value = {s}
            p = {p}""")
        if p <= 0.05/10: # With Bonferoni correcture
            print("            *** Heavily Correlated")
        elif p >= 0.05/10 and p <= 0.05:
            print("          * correlated")
        else:
            print("            No significant correlation")
        print()

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

def bmi_train(data): #Parametric Model should be changed to non parametric because data is not normally distributed :"(
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

estimate_correlation(data)


"""K-Nearest Neighbour (KNN) Model"""
def KNN(data): # Or: def KNN(X_train, X_test, y_train, y_test), has to be scaled
   
    # Definition of X and y ??? -> X_train, X_test, y_train, y_test
    X = data[['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']]
    y = data['stroke']
    
    # Standardization - necessary ???
    X = StandardScaler().fit_transform(X)
    
    # K tuning
    # Selection of the optimal k value. K is a hyperparameter. There is no one proper method of estimation. K should be an odd number.
    # Square Root Method is used: Square root of the number of samples in the training dataset.
    k_neighbors = sqrt(len(y_train)
    
    # Define the model: Initiate KNN
    classifier = KNeighborsClassifier(n_neighbors = k_neighbors, metric = 'euclidean')
    # Euclidean Distance used: It is the most commonly used distance formula in machine learning.
    classifier.fit(X_train, y_train)
    
    # Predict the test results
    y_pred = classifier.predict(X_test)
    
    # Evaluation of model
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix of KNN:"\n cm)
    
    accuracy_score = accuracy_score(y_test, y_pred)
    print("Accuracy Score of KNN:"\n accuracy_score)
    
    f1_score = f1_score(y_test, y_pred)
    print("F1 Score of KNN:"\n f1_score)
    
    
"""Random Forest"""
def evaluation_metrics(tcl, y, X, ax, legend_entry = "my legendEntry"):
    """
    Comput evaluation metrics for provided classifier given true labels and input features
    Provides a plot of ROC curve 

    tcl: true class label, numpy array

    y:  true class labels, numpy array

    X: feature matrix, numpy array

    ax: matplotlip axis to plot on

    legend_entry: the legend entry that should be displayed on plot, string

    return: confusion matrix comprising true positives (tp), true negatives (tn), false positives (fp), and false negatives (fn)
            four integers
    """

    # Get label prediction
    y_test_pred = clf.predict(X)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_test_pred).ravel()

    # Calculate the evaoluation metrics
    precision   = tp / (tp + fp)
    specificity = tn / (fp + tn)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    recall      = tp / (tp + fn)
    f1          = tp / (tp + 0.5 * (fn + fp))

    # Get the roc curve using sklearn function
    y_test_predict_proba = clf.predict_proba(X)
    fp_rates, tp_rates, _ = roc_curve(y, y_test_predict_proba[:,1])

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    # Plot on the provided axis
    ax.plot(fp_rates, tp_rates ,label = legend_entry)

    return [accuracy,precision,recall,specificity,f1, roc_auc]

def random_forest()