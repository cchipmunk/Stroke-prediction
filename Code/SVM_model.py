import pandas as pd
import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score
from sklearn import svm
import matplotlib.pyplot as plt
import math
import scipy.stats as sts
import seaborn as sns
from itertools import combinations
from statistics import mean
import warnings

warnings.filterwarnings("ignore")


### Importing data ### 

df_path = "./data/Truncated_data/Stroke_data.csv"

data = pd.read_csv(df_path)
print('success')

data = data.iloc[:,1:] # Deleting second id columns

# Weighted data?

print(f"percentage of stroke patient: {(sum(data.stroke)/data.shape[0])*100}%")
print("Unbalanced data set")

### One hot encoding + Initiating k_folds / Splits ###

encoded_df = data

for i in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    encoded_df = pd.get_dummies(encoded_df, columns = [i], prefix = [i], drop_first=True)

X, y = encoded_df.iloc[:,1:-1], encoded_df.loc[:, "stroke"]

#Correlation between all Spalten in the data
num_cols = ['age', 'avg_glucose_level', 'bmi']

for i in num_cols:
    pear_v = sts.pearsonr(data['stroke'], data[i] )
    print(f"""Pearson coefficient for {i} :
          value = {pear_v[0]}
          p = {pear_v[1]}""")
    if pear_v[1] <= 0.05/10: # With Bonferoni correcture
        print("          *** Heavily Correlated")
    elif pear_v[1] >= 0.05/10 and p <= 0.05:
        print("          * correlated")
    else:
        print("          No significant correlation")
    print()
    
for i in ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    cont_table = pd.crosstab(data[i], data['stroke'])
    # print(cont_table)
    # print(sts.chi2_contingency(cont_table))
    print()
    p = sts.chi2_contingency(cont_table)[1]
    s = sts.chi2_contingency(cont_table)[0]

    print(f"""Chi_sq for {i} :
          Value = {s}
          p = {p}""")
    if p <= 0.05/10: # With Bonferoni correcture
        print("          *** Heavily Correlated")
    elif p >= 0.05/10 and p <= 0.05:
        print("          * correlated")
    else:
        print("          No significant correlation")
# Absolutely no idea how this works in python sk_folds = StratifiedKFold(n_splits = 5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = RobustScaler() #Using robust because data not normally distributed
num_cols = ['age', 'avg_glucose_level', 'bmi'] #already defined above
X_train_scaled, X_test_scaled = X_train, X_test
X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols]) 
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

""" ### Support Vector machine implementation ### """

# Linear svm https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC
# dual = False, because n_samples > n_features
# class_weight = 'Balanced', because our Data set is heavily unbalanced
# C = 10, set at random, I have no idea how big it should be, but probably quite big
#seeing as our data set is huge


#Way too precise, no idea how this is working // Not working

print(X_train.shape)
print(y_train.shape)

clf = svm.SVC(kernel = 'rbf', C=1, gamma='auto')

clf.fit(X_train_scaled, y_train)

y_pred = (clf.predict(X_test_scaled))

print(clf.score(X_test_scaled, y_test))

print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
print("F1 score", metrics.f1_score(y_test, y_pred))


X_scaled = X_train
X_scaled[num_cols] = scaler.transform(X_train[num_cols])
scores = cross_val_score(clf, X_scaled, y_train, cv =5)


print(scores)
print(mean(scores))




print("###########")

print(data['smoking_status'].unique())


#Correlation between features and labels
#Information leakage
#Maybe linearly solvble / very easy