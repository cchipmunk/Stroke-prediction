import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats as sts
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial.distance import euclidean, cityblock, minkowski
from sklearn import cluster, datasets, mixture, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, auc
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, roc_curve
import matplotlib.pyplot as plt
import math
import seaborn as sns
from itertools import combinations
from itertools import cycle, islice
from statistics import mean
import warnings
import time


warnings.filterwarnings("ignore")

""" All Models """
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

"""Test for normality"""    
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


"""K-Nearest Neighbour (KNN) Model"""
def KNN(X_train_scaled,X_test_scaled,y_train,y_test): 
   
    # K tuning
    # Selection of the optimal k value. K is a hyperparameter. There is no one proper method of estimation. K should be an odd number.
    # Square Root Method is used: Square root of the number of samples in the training dataset.
    k_neighbors = round(math.sqrt(len(y_train)))
    
    # Define the model: Initiate KNN
    classifier = KNeighborsClassifier(n_neighbors = k_neighbors, metric = 'euclidean')
    # Euclidean Distance used: It is the most commonly used distance formula in machine learning.
    classifier.fit(X_train_scaled, y_train)
    
    # Predict the test results
    y_pred = classifier.predict(X_test_scaled)
    
    # Evaluation of model
    cm = confusion_matrix(y_test, y_pred)
    """
    print("Confusion Matrix of KNN:"\n cm)
    """
    
    accuracy_score = accuracy_score(y_test, y_pred)
    """
    print("Accuracy Score of KNN:"\n accuracy_score)
    """
    
    f1_score = f1_score(y_test, y_pred)
    """
    print("F1 Score of KNN:"\n f1_score)
    """
    

print('hello World')

def smoking_model (X_train_scaled):

    ###                                             ###
    ### Step 1: Reverse the one hot encoding :'(    ###
    ###                                             ###

    #Using scaled and encoded data
    df = X_train_scaled
    #Creating df with only smoking data
    smoker_information = df[['smoking_status_never smoked', 'smoking_status_smokes', 'smoking_status_formerly smoked']]
    #Selecting column names of the one hot encoding and saving them in new column
    smoker_information['smoking_status'] = smoker_information.idxmax(axis = 1)
    sm_col = smoker_information['smoking_status']

    #Editing the names
    for i in sm_col.index:
        if sm_col[i] == 'smoking_status_never smoked':
            #because we used drop first in the one hot encoding, the "unknown" values are dropped. 
            #Using the idx function with 3 False values the first one is stored as max
            #We use this function to check if in reality the value saved as "neversmoked" are not just "unknown"
            if smoker_information['smoking_status_never smoked'][i] == False:
                sm_col[i] = 'unknown'
            else:
                sm_col[i] = 'never smoked'
        elif sm_col[i] == 'smoking_status_formerly smoked':
            sm_col[i] = 'formerly smoked'
        elif sm_col[i] == 'smoking_status_smokes':
            sm_col[i] = 'smokes'
    df['smoking_status'] = sm_col
    df.drop(['smoking_status_never smoked', 'smoking_status_smokes', 'smoking_status_formerly smoked'], axis = 1)
    
    df = df.loc[df['smoking_status'] != 'unknown']

    X = df.drop(['smoking_status'], axis = 1)
    y = df[['smoking_status']]

    #step 2: train test splits

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

    #step 3 : model and hyperparameter tuning

    clf = KNeighborsClassifier(n_neighbors = 8, metric = 'euclidean')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_simp = ['never smoked' for i in range(len(y_pred))]
    print(" ### Smoking model performance ###")
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 score", f1_score(y_test, y_pred, average='macro'))
    print()
    print(" ###  simple model performance ###")
    print("Precision:", precision_score(y_test, y_simp, average='macro'))
    print("Recall:", recall_score(y_test, y_simp, average='macro'))
    print("F1 score", f1_score(y_test, y_simp, average='macro'))


    print(sm_col.value_counts())

    # Step 4 Saving the data in truncated data set.


"""Test for normality"""    
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

"""Logstic Regression"""


"""K-Nearest Neighbour (KNN) Model"""
def KNN(X_train_scaled,X_test_scaled,y_train,y_test): 
   
    # K tuning
    # Selection of the optimal k value. K is a hyperparameter. There is no one proper method of estimation. K should be an odd number.
    # Square Root Method is used: Square root of the number of samples in the training dataset.
    k_neighbors = round(math.sqrt(len(y_train)))
    
    # Define the model: Initiate KNN
    classifier = KNeighborsClassifier(n_neighbors = k_neighbors, metric = 'euclidean')
    # Euclidean Distance used: It is the most commonly used distance formula in machine learning.
    classifier.fit(X_train_scaled, y_train)
    
    # Predict the test results
    y_pred = classifier.predict(X_test_scaled)
    
    # Evaluation of model
    cm = confusion_matrix(y_test, y_pred)
    """
    print("Confusion Matrix of KNN:"\n cm)
    """
    
    accuracy_score = accuracy_score(y_test, y_pred)
    """
    print("Accuracy Score of KNN:"\n accuracy_score)
    """
    
    f1_score = f1_score(y_test, y_pred)
    """
    print("F1 Score of KNN:"\n f1_score)
    """
    
"""Random Forest"""
# Plot the diagonal line 
def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

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
    y_test_pred = tcl.predict(X)

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y, y_test_pred).ravel()

    # Calculate the evaoluation metrics
    precision   = tp / (tp + fp)
    specificity = tn / (fp + tn)
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    recall      = tp / (tp + fn)
    f1          = tp / (tp + 0.5 * (fn + fp))

    # Get the roc curve using sklearn function
    y_test_predict_proba = tcl.predict_proba(X)
    fp_rates, tp_rates, _ = roc_curve(y, y_test_predict_proba[:,1])

    # Calculate the area under the roc curve using a sklearn function
    roc_auc = auc(fp_rates, tp_rates)

    # Plot on the provided axis
    ax.plot(fp_rates, tp_rates ,label = legend_entry)

    return [accuracy,precision,recall,specificity,f1, roc_auc]

def random_forest(X, y, n_splits):
    """
    Random forest function

    X: data

    y: true outcomes

    n_splits: number of splits for Stratified KFolds because of inbalanced data, integer
    """

    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 50)

    # Prepare the performance overview data frame
    df_performance = pd.DataFrame(columns = ['fold','clf','accuracy','precision','recall',
                                            'specificity','F1','roc_auc'])
    df_LR_normcoef = pd.DataFrame(index = X.columns, columns = np.arange(n_splits)) # evtl. löschen

    # Plot to save performance metrics
    fold = 0
    fig, axs = plt.subplots(figsize=(9, 4))

    # Loop over all splits
    for train_index, test_index in skf.split(X, y):
        # Get the relevant subsets for training and testing
        X_test  = X.iloc[test_index]
        y_test  = y.iloc[test_index]
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]


        # Standardize the numerical features using training set statistics
        sc = StandardScaler()
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc  = sc.transform(X_test)

        # Random forest
        tcl = RandomForestClassifier(random_state = 50)

        # Fit
        tcl.fit(X_train_sc, y_train)

        # Evaluate classifiers using evaluation metrics
        eval_metrics_RF = evaluation_metrics(tcl, y_test, X_test_sc, axs, legend_entry=str(fold)) 
        df_performance.loc[len(df_performance), :] = [fold, 'RF'] + eval_metrics_RF

        # Increase counter for folds
        fold += 1

    # Edit plot
    model_name = "Random Forest"
    axs.set_xlabel("FPR")
    axs.set_ylabel("TPR")
    add_identity(axs, color="r", ls="--",label = 'random\nclassifier')
    plt.legend()
    axs.set_title(f"ROC curve of {model_name}", fontsize=9)

    # Save the plot 
    plt.tight_layout()
    # plt.savefig('../output/roc_curves.png')
    plt.show()

    # Summarize the performance metrics over all folds
    print(df_performance.groupby(by = 'clf').mean())
    print(df_performance.groupby(by = 'clf').std())


def support_v_m(X_train_scaled, y_train, y_test):

    clf = svm.SVC(kernel = 'linear', C=5)
    clf.fit(X_train_scaled, y_train)

    y_pred = (clf.predict(X_test_scaled))

    print(clf.score(X_test_scaled, y_test))

    print(len(y_pred))
    print(len(y_test))


    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 score", f1_score(y_test, y_pred))
    
    fpr,tpr, _ = roc_curve(y_test, y_pred)

    plt.plot(fpr,tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title("Support Vector - ROC curve")
    plt.show()  


""" Correlation estimation code """
# show correlation --> needed?
def correlation_plot():
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

"""Load data and preprocessing"""

### Load the data ### 
df_path = "../Data/Truncated_data/Stroke_data.csv"
data = pd.read_csv(df_path)

### Data Exploration ###
def data_exploration():
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
    print('There are', data.groupby('heart_disease').ngroups,'unique heart diseases in the data.') # Two

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

    # How many "Other" in gender are there? --> only 1, drop the person --> Reason: see report
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


### Test/train splits and data encoding ###
def split_and_encode(data):
    cat_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status'] # TODO add 2 further categorical data!
    encoded_df = data
    encoded_df = pd.get_dummies(encoded_df, columns = cat_columns, prefix = cat_columns, drop_first = True)
    
    X = encoded_df.drop("stroke", axis = 1)
    y = encoded_df["stroke"]

    return X, y

def do_train_test_split(X, y):
    # Defaul of train test_split is not stratified, need for specification
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56, stratify = y)

    return X_train, X_test, y_train, y_test

### Scaling the data using Robust Scaler ###
def scale_data(X_train, X_test):
    scaler = RobustScaler() #Using robust because data not normally distributed
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    X_train_scaled, X_test_scaled = X_train, X_test
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols]) 
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled

### Calling split, encode and scaling functions ###

X, y = split_and_encode(data)

X_train, X_test, y_train, y_test = do_train_test_split(X, y)

X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
   


# estimate_correlation(data)
random_forest(X, y, 5)

smoking_model(X_train_scaled)
