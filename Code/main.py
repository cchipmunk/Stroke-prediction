import pandas as pd
import numpy as np
from scipy import stats
import scipy.stats as sts
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.spatial.distance import euclidean, cityblock, minkowski
from sklearn import cluster, datasets, mixture, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, confusion_matrix, auc, roc_auc_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, f1_score, accuracy_score, precision_score, recall_score, roc_curve, auc
import matplotlib.pyplot as plt
import math
import seaborn as sns
from itertools import combinations
from itertools import cycle, islice
from statistics import mean
import warnings
import time

warnings.filterwarnings("ignore")

def get_ids():
    #gets train test split ids.
    # Originally, the idea was to have this one split to prevent all data leakage and only work on the training set.
    # We did not coordinate enough for it to be used, but it's still necessary for the data truncation.
      
    
    global og_data

    og_data = pd.read_csv("../Data/ORIGINAL_DATA/stroke_dataset/healthcare-dataset-stroke-data.csv")

    #dropping value because only 1 observation with that value
    og_data = og_data[og_data['gender'] != 'Other']

    #Preparing feature and label
    X = og_data.drop('stroke', axis = 1)
    y = og_data['stroke']

    #Performing startified splits
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56, stratify = y)

    global train_id
    global test_id

    #Saving ids in global variables for later localisation
    train_id = X_train.index
    test_id = X_test.index

    print('Train test split ids have been generated')
    print()


def bmi_scores(model, X_train, y_train, X_test, y_test, y_mean):
    ### Function to print out metrics of bmi truncation model ###

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


def Kolmogorov_Smirnov(df):
    ### Kolgorov_Smirnov test for normality on all continuous variable (with and without transformations) ###
    for i in ['age', 'avg_glucose_level', 'bmi']:
        print(f"{i} - Kolgorov Smirnov test for normality:")
        res = stats.kstest(df[i], stats.norm.cdf)
        print(f"probability that {i} is normaly distributed with no transformation = {1 - res.statistic}")
        res = stats.kstest((df[i])**0.5, stats.norm.cdf)
        print(f"probability that {i} is normaly distributed with sqrt transformation = {1 - res.statistic}")
        res = stats.kstest(np.log10(df[i]), stats.norm.cdf)
        print(f"probability that {i} is normaly distributed with log transformation = {1 - res.statistic}")
        print()


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
    

def smoking_model ():

    ### Step 1 - Importing and encoding ###

    #Importing data
    data = pd.read_csv('../data/ORIGINAL_DATA/stroke_dataset/healthcare-dataset-stroke-data.csv')
    #dropping gender other because only one observation
    data = data[data['gender'] != 'Other']

    # Using Ids located with get_ids function
    df = data.iloc[train_id]
    df = df.drop(['stroke', 'bmi'], axis = 1)
    df = df.loc[data['smoking_status'] != 'Unknown'] 

    #Creating features and labels data frames
    X = df.drop(['smoking_status'], axis= 1)
    y = df['smoking_status']

    #encoding categorical columns
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type']
    encoded_X = pd.get_dummies(X, columns = cat_cols, prefix = cat_cols, drop_first = True )

    ### step 2 - train test splits and scaling ###

    #Spliting the already split data, not necessary, we lose a bit of accuracy, but at this point there's not enough time to change it
    X_train, X_test, y_train, y_test = train_test_split(encoded_X, y, test_size=0.2, random_state=56, stratify=y)

    #Using robust because data not normally distributed
    scaler = RobustScaler() 
    num_cols = ['age', 'avg_glucose_level']
    X_train_scaled, X_test_scaled = X_train, X_test
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols]) 
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    ### step 3 - model training and hyperparameter tuning ###

    #initiating classifier and fitting it, no specific reason for using knn, except ease of implementation
    clf = KNeighborsClassifier(n_neighbors = 8, metric = 'euclidean')
    clf.fit(X_train_scaled, y_train)

    #creating prediction with our model vs simple classifying model.
    y_pred = clf.predict(X_test_scaled)
    y_simp = ['never smoked' for i in range(len(y_pred))]

    ### Step 4 - performance metrics ###

    print(" ### Smoking model performance ###")
    print("Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Recall:", recall_score(y_test, y_pred, average='macro'))
    print("F1 score", f1_score(y_test, y_pred, average='macro'))
    print()
    print(" ###  simple model performance ###")
    print("Precision:", precision_score(y_test, y_simp, average='macro'))
    print("Recall:", recall_score(y_test, y_simp, average='macro'))
    print("F1 score", f1_score(y_test, y_simp, average='macro'))

    ### Step 5 - Saving the data in truncated data set ###

    #Selecting rows where truncation is needed
    truncation = data.loc[data['smoking_status'] == 'Unknown']
    #creating features (X)
    trunc_X = truncation.drop(['smoking_status', 'bmi', 'stroke'], axis = 1)
    #Encoding features (X)
    encoded_trunc_X = pd.get_dummies(trunc_X, columns = cat_cols, prefix = cat_cols, drop_first = True )
    #scaling features with our scaler (X)
    trunc_X_scaled = encoded_trunc_X
    trunc_X_scaled[num_cols] = scaler.transform(encoded_trunc_X[num_cols])

    # predicting outcome y
    y = clf.predict(trunc_X_scaled)

    #saving it in truncation data
    truncation['smoking_status'] = y
    
    #saving truncation data in original data frame
    data.loc[data['smoking_status'] == 'Unknown'] = truncation

    #saving it as csv
    data.to_csv('../data/Truncated_data/Stroke_data_smoking.csv')
    print()
    print('Truncated data saved to: ../data/Truncated_data/Stroke_data_smoking.csv')
    print()


def new_bmi_model():
    ### Bmi Truncation model ###


    # importing the already truncated data with smoking status
    data = pd.read_csv('../data/Truncated_data/Stroke_data_smoking.csv')
    df = data.drop(['Unnamed: 0', 'id', 'smoking_status', 'stroke'], axis=1)
    
    #Encoding the categorical columns
    cat_cols = ['gender', 'ever_married', 'work_type', 'Residence_type']
    encoded_df = pd.get_dummies(df, columns = cat_cols, prefix = cat_cols, drop_first = True )
    

    #using our ids generated with the first function to make our train / test splits
    learning_Set = encoded_df.iloc[train_id]
    learning_Set = learning_Set.loc[df['bmi'].notna()]
    X = learning_Set.drop('bmi', axis = 1)
    y = learning_Set['bmi']

    # Splitting our data a second time for no reason (we lose a bit of accuracy, but at this point there's not enough time to change it)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56) # TODO do we need stratify as well?

    #Scaling the data with a robust scaler (because our data is not normally distributed)
    scaler = RobustScaler() 
    num_cols = ['age', 'avg_glucose_level']
    X_train_scaled, X_test_scaled = X_train, X_test
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols]) 
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    #fitting linear regresstion (model choice: easy to implement)
    LR = LinearRegression()
    LR.fit(X_train_scaled, y_train)
    
    #Creating a Model which calculates the mean for every estimated value as comparaison
    y_mean = [mean(y)] * len(y_test)

    #Evaluating performance
    bmi_scores(LR, X_train, y_train, X_test, y_test, y_mean)


    ### Saving Data in file 

    X = encoded_df.loc[data['bmi'].isna()]
    X = X.drop('bmi', axis = 1)

    #Scaling data to truncate (Using same Scaler as in training model)
    X_scaled = X
    X_scaled[num_cols] = scaler.transform(X[num_cols])


    #Applying model on data to estimate
    y = LR.predict(X_scaled) 

    # Saving data in the panda table and saving it as csv:
    # We can do that with the One-hot encoded data because it does not affect the  lentgh of rows or indexes.
    truncation = data.loc[data['bmi'].isna()]
    truncation['bmi'] = y

    #putting truncation in original dataset    
    data.loc[data['bmi'].isna()] = truncation
    
    #saving it as csv
    data.to_csv('../data/Truncated_data/Stroke_data.csv')
    print()
    print('Truncated data saved to: ../data/Truncated_data/Stroke_data.csv')
    print()



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

def logistic_regression(X,y,n_splits):
    # 0.1) Calculate multicollinearity variance inflation factor (VIF)
    # to ensure that indepentend variables do not correlate with each other?
    # 0.2) Imbalanced or balanced -> look at how many have a stroke and how many do not
    # 1) Get data -> X and y

    # 2) Create a model and train it
    model = LogisticRegression(solver='liblinear', C=10.0, random_state=0)
    model.fit(X, y)

    # 3) Evaluation metrics: Perform a n-fold crossvalidation - prepare the splits
    skf      = StratifiedKFold(n_splits=n_splits)

    # Prepare the performance overview data frame
    df_performance = pd.DataFrame(columns = ['fold','clf','accuracy','precision','recall',
                                             'specificity','F1','roc_auc'])
    df_LR_normcoef = pd.DataFrame(index = X.columns, columns = np.arange(n_splits))

    # Use this counter to save your performance metrics for each crossvalidation fold
    # also plot the roc curve for each model and fold into a joint subplot
    fold = 0
    fig,axs = plt.subplots(figsize=(9, 4))

    # Loop over all splits
    for train_index, test_index in skf.split(X, y):

        # Get the relevant subsets for training and testing
        X_test  = X.iloc[test_index]
        y_test  = y.iloc[test_index]
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]

        # Standardize the numerical features using training set statistics
        rs = RobustScaler()
        X_train_sc = rs.fit_transform(X_train)
        X_test_sc  = rs.transform(X_test)

        # Creat prediction models and fit them to the training data
        # Logistic regression
        clf = LogisticRegression(random_state=1)
        clf.fit(X_train_sc, y_train)

        # Get the top 5 features that contribute most to the classification
        df_this_LR_coefs       = pd.DataFrame(zip(X_train.columns, np.transpose(clf.coef_[0])), columns=['features', 'coef'])
        df_LR_normcoef.loc[:,fold] = df_this_LR_coefs['coef'].values/df_this_LR_coefs['coef'].abs().sum()

        # Evaluate your classifiers
        eval_metrics = evaluation_metrics(clf, y_test, X_test_sc, axs,legend_entry=str(fold))
        df_performance.loc[len(df_performance)-1,:] = [fold,'LR']+eval_metrics

        # increase counter for folds
        fold += 1

    axs.set_xlabel('FPR', fontsize = 11)
    axs.set_ylabel('TPR', fontsize = 11)
    add_identity(axs, color="r", ls="--",label = 'random\nclassifier')
    axs.legend()
    axs.set_title("ROC curve for LR", fontsize = 12)
    plt.tight_layout()
    plt.savefig('../output/LR_roc_curve.png')
    plt.show()

    # Summarize the folds
    print(df_performance.groupby(by = 'clf').mean())
    print(df_performance.groupby(by = 'clf').std())

    # Average the feature importance across the n folds and sort them
    df_LR_normcoef['mean_coef'] = df_LR_normcoef.mean(axis=1)
    df_LR_normcoef_sorted = df_LR_normcoef.sort_values('mean_coef', ascending=False)

    # Visualize the normalized feature importance across the n folds and add error bar to indicate the std
    top_features_15 = df_LR_normcoef_sorted[:15]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(top_features_15.index.tolist(), top_features_15['mean_coef'].values, yerr=top_features_15.std(axis=1).tolist(), capsize=4)
    ax.set_xlabel('Features', fontsize = 11)
    ax.set_ylabel('Coefficient Value (normalized)', fontsize = 11)
    ax.set_title('Feature importance for LR across 5 folds', fontsize = 12)
    ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.savefig('../output/LR_feature_importance.png')
    plt.show()
    
    # Summarize the performance metrics over all folds (with std)
    df_mean_std_performance = df_performance.drop('fold', axis=1).groupby('clf').agg(['mean', 'std'])
    df_mean_std_performance.columns = ['_'.join(col).strip() for col in df_mean_std_performance.columns.values]
    df_mean_std_performance = df_mean_std_performance.reset_index()
    df_mean_std_performance.to_csv('../output/LR_mean_std_performance.csv', index=False)


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
    print("Confusion Matrix of KNN:", cm)
    
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy Score of KNN:", accuracy)
    
    f1 = f1_score(y_test, y_pred)
    print("F1 Score of KNN:", f1)
    
    
def add_identity(axes, *line_args, **line_kwargs):
    # Plot the diagonal line 
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
        rs = RobustScaler() #because not normal distrbuted
        X_train_rs = rs.fit_transform(X_train)
        X_test_rs  = rs.transform(X_test)

        
        # Random forest
        tcl1 = RandomForestClassifier(random_state = 50, class_weight = "balanced")
        space = dict() 
        space['max_depth'] = [1,5,10]
        space['min_samples_split'] = [5,10,100,500]
        space['min_samples_leaf'] = [10,50,250,500,750]
        space['max_features'] = [1,5,10]
        space['max_samples'] = [20,500,2000,3000]

        # Get best hyperparameters with Random Search
        # define search 
        search = RandomizedSearchCV(tcl1, space, n_iter=10, scoring='roc_auc', n_jobs=-1)

        # execute search 
        tunned_tree = search.fit(X_train_rs, y_train) 

        # Fit
        # tcl.fit(X_train_rs, y_train)

        # summarize result 
        print("Parameters for RandomSearch", tunned_tree.best_params_)

        # Adjust RF
        tcl = RandomForestClassifier(min_samples_split= tunned_tree.best_params_["min_samples_split"], min_samples_leaf=tunned_tree.best_params_["min_samples_leaf"], 
                             max_depth=tunned_tree.best_params_["max_depth"], max_features = tunned_tree.best_params_["max_features"], max_samples = tunned_tree.best_params_["max_samples"], random_state=50, class_weight='balanced') 

        
        tcl = RandomForestClassifier(random_state = 50, class_weight= "balanced")

        tcl.fit(X_train_rs,y_train)

        y_pred = tcl.predict(X_test_rs)

        
        print('Accuracy of Random Forest classifier on train set: {:.2f}'.format(tcl.score(X_train_rs, y_train)))
        print('Accuracy of Random Forest classifier on test set: {:.2f}'.format(tcl.score(X_test_rs, y_test)))
        print('Recall of Random Forest classifier on test set: {:.2f}'.format(recall_score(y_test, y_pred)))
        print('Precision of Random Forest classifier on test set: {:.2f}'.format(precision_score(y_test, y_pred)))
        print('ROC/AUC of Random Forest classifier on test set: {:.2f}'.format(roc_auc_score(y_test, tcl.predict_proba(X_test)[:,1])))
        

        # Evaluate classifiers using evaluation metrics
        eval_metrics_RF = evaluation_metrics(tcl, y_test, X_test_rs, axs, legend_entry=str(fold)) 
        df_performance.loc[len(df_performance), :] = [fold, 'RF'] + eval_metrics_RF

        # Increase counter for folds
        fold += 1

    # Edit plot
    model_name = "Random Forest"
    axs.set_xlabel("FPR", fontsize = 11)
    axs.set_ylabel("TPR", fontsize = 11)
    add_identity(axs, color="r", ls="--",label = 'random\nclassifier')
    plt.legend()
    axs.set_title(f"ROC curve of {model_name}", fontsize=12)

    # Save the plot 
    plt.tight_layout()
    plt.savefig('../output/RF_roc_curve.png')
    plt.show()

    # Summarize the performance metrics over all folds
    print(df_performance.groupby(by = 'clf').mean())
    print(df_performance.groupby(by = 'clf').std())


def svm_hyp_search(X_train,y_train):
    #Using randomised gridsearch for hyperparameter tuning

        #for rbf kernel
        grid_param = {'C': [0, 0.01, 0.1, 1, 10], 
                      'gamma': [10, 5, 2, 1, 0.1, 0.01, 0.001],
                      'kernel': ['rbf'],
                      'class_weight' : ['balanced']}
        
        #for linear kernel
        grid_param_02 = {'C': [0, 0.01, 0.1, 1, 10], 
                      'gamma': [10, 5, 2, 1, 0.1, 0.01, 0.001],
                      'kernel': ['linear'],
                      'class_weight' : ['balanced']}
        
        #for sigmoid kernel
        grid_param_03 = {'C': [0, 0.01, 0.1, 1, 5,10], 
                      'gamma': ['scale', 'auto'],
                      'coef0': [-20, -10, -5, 5, 10, 20],
                      'kernel': ['sigmoid'],
                      'class_weight' : ['balanced']}

        grid = RandomizedSearchCV(svm.SVC(), grid_param_03, refit = True, verbose = 3)

        grid.fit(X_train, y_train)


        print("griddy grid grid searchy search")
        print()
        print(grid.best_params_)

        print(grid.best_estimator_)


def support_v_m(X, y, n_splits):
    """ 
    ### Support Vector function ###

    X = Features

    y = labels

    n_splits = number of splits for CV
    
    """
    
    # Prepare Split indexes
    skf = StratifiedKFold(n_splits = n_splits, shuffle = True, random_state = 50)

    # Prepare the performance overview data frame
    precision = []
    recall = []
    accuracy = []
    F1 = []

    fold = 1

    # Loop over all splits
    for train_index, test_index in skf.split(X, y):
        # Get the relevant subsets for training and testing
        X_test  = X.iloc[test_index]
        y_test  = y.iloc[test_index]
        X_train = X.iloc[train_index]
        y_train = y.iloc[train_index]


        # Standardize the numerical features using training set statistics
        rs = RobustScaler() #because not normal distrbuted
        X_train_rs = rs.fit_transform(X_train)
        X_test_rs  = rs.transform(X_test)

        
        # Initiating classifier with hyperparameters found using randomised grid search (see svm_hyp_search() function)
        svm_clf = svm.SVC(kernel='sigmoid', C=5, coef0= 0, gamma= 'scale', class_weight = 'balanced', random_state=40)

        # Fit the function and save performance metrics
        svm_clf.fit(X_train_rs, y_train)

        y_pred = svm_clf.predict(X_test_rs)

        
        print(f"### Fold number: {fold} ###")
        print()
        print('Accuracy of SVM on train set: {:.2f}'.format(svm_clf.score(X_train_rs, y_train)))
        print('')
        print('Accuracy of SVM on test set: {:.2f}'.format(svm_clf.score(X_test_rs, y_test)))
        accuracy.append(svm_clf.score(X_test_rs, y_test))
        print('Recall of SVM on test set: {:.2f}'.format(recall_score(y_test, y_pred)))
        recall.append(round(recall_score(y_test, y_pred), 3))
        print('Precision of SVM on test set: {:.2f}'.format(precision_score(y_test, y_pred)))
        precision.append(round(precision_score(y_test, y_pred),3))
        print(f'F1 score of SVM: {f1_score(y_test, y_pred)}')
        F1.append(round(f1_score(y_test, y_pred), 3))

        print("area under curve (auc): ", roc_auc_score(y_test, y_pred))

        fold += 1
        print()
        print()


    # Printing mean values from the cross validation
    print(f'Precision across folds: {precision}, mean = {mean(precision)}')
    print(f'Recall across folds, {recall}, mean = {mean(recall)}')
    print(f'F1 across folds: {F1}, mean = {mean(F1)}')
    print()
    print()

    test_fpr, test_tpr, te_thresholds = roc_curve(y_test, y_pred)


    # Only 3 points because ROC curve uses probability estimates and SVM does not create these estimate
    
    plt.grid()
    plt.plot(test_fpr, test_tpr)
    plt.plot([0,1],[0,1],'g--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve for SVM")
    plt.grid(color='black', linestyle='-', linewidth=0.5)
    plt.savefig("../output/SVM_roc_curve.png")
    plt.show()

    #Clearing figure for future graphs
    plt.clf



# show correlation --> needed?
def correlation_plot():
    corr_plot= sns.pairplot(data=data, x_vars=vars, y_vars=vars)
    corr_plot.fig.suptitle("Correlation between continuous variables", y = 1)
    plt.show()

def estimate_correlation(data):
    ### Simple function to estimate correlation between Features and label ###
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


def data_exploration(data):
    ### Data Exploration ###

    # Look at data
    # Look at Head
    print(data.head())

    # Row and column number 
    print("rows:", data.shape[0], "columns:", data.shape[1])

    # Data types
    print(data.dtypes)

    # Missing values --> fill in with other data --> see bmi and smoking
    print(data.isna().sum(axis = 0)) 

    # How many patients have all attributes present?
    print((data.isna().sum(axis=1) == 0).sum())

    # How is the outcome (stroke) distributed?
    print("Distribution: \n", data["stroke"].value_counts())

    # Number of females and males with stroke
    print("Female:", len(data[(data["gender"] == "Female") & (data["stroke"] == 1)]))
    print("Male:", len(data[(data["gender"] == "Male") & (data["stroke"] == 1)]))

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

    print(data.dtypes)

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
        "avg_glucose_level": "Average Glucose Level (mg/dl)",
        "bmi": "Body Mass Index (kg/m**2)",
    }
    for i, ax in enumerate(axs.flatten()):
        sns.histplot(
            x=vars[i], 
            data=data, 
            kde=True, 
            ax=ax
        )
        ax.set_title(f"Distribution of\n{titles[vars[i]]}", fontsize=12)
        ax.set_xlabel(xlabels[vars[i]], fontsize=11)
        if i not in [0, 3]:
            ax.set_ylabel(None)
    fig.tight_layout()
    plt.savefig("../output/cont_data.png")
    plt.show()


    ### Demographic data insights ###
    dem_data = data.copy()
    dem_data["stroke"] = pd.Categorical(dem_data["stroke"])
    dem_data["stroke"] = dem_data["stroke"].cat.rename_categories(
        {0: "No", 1: "Yes"}
    )

    # Age distribution
    fig, ax = plt.subplots(1, 2, figsize = (12,6))

    # Plot the age distribution separated by outcome
    age = sns.histplot(dem_data, x = "age", binwidth = 5, hue = "stroke", ax = ax[0])
    age.set_xlabel(xlabel = "Age [year]", fontsize = 11) 
    age.set_ylabel(ylabel = "Number of Patients", fontsize = 11) 
    age.set_title(label = "Age Distribution by Stroke Outcome", fontsize = 12, loc = "center")

    # Plot age dispersion separated by gender
    age_range = sns.boxplot(dem_data, y = "age", x = "gender", hue = "stroke", ax = ax[1], width = 0.4)
    age_range.set_ylabel(ylabel = "Age [year]", fontsize = 11) 
    age_range.set_xlabel(xlabel='Gender', fontsize = 11) 
    age_range.set_title(label = 'Age Range split by Gender and Stroke Outcome', fontsize = 12, loc = "center")
    plt.savefig("../output/demo_insights.png")
    plt.show()



def encode(data):
    ### Test/train splits and data encoding ###

    cat_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    encoded_df = data # TODO evtl. copy()
    encoded_df = pd.get_dummies(encoded_df, columns = cat_columns, prefix = cat_columns, drop_first = True)

    encoded_df.drop(['Unnamed: 0.1', 'Unnamed: 0', 'id'], axis=1, inplace = True)

    return encoded_df


def scale_data(X_train, X_test):
    ### Scaling the data using Robust Scaler ###

    scaler = RobustScaler() #Using robust because data not normally distributed
    num_cols = ['age', 'avg_glucose_level', 'bmi']
    X_train_scaled, X_test_scaled = X_train, X_test
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols]) 
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    return X_train_scaled, X_test_scaled

#getting test/train ids
get_ids()

#Data truncation
smoking_model()
new_bmi_model()

#Loading truncated data
df_path = "../Data/Truncated_data/Stroke_data.csv"
data = pd.read_csv(df_path)

#Exploring data
data_exploration(data)

# One hot encoding
encoded_df = encode(data)

X = encoded_df.drop(['stroke'], axis = 1)
y = encoded_df['stroke']

#Encoding and spliting (for KNN)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56, stratify = y)

#Scaling data (for KNN)
X_train_scaled, X_test_scaled = scale_data(X_train, X_test)


#Calling all models
KNN(X_train_scaled, X_test_scaled, y_train, y_test)
random_forest(X, y, 5)
logistic_regression(X,y,5)
support_v_m(X, y, 5)
