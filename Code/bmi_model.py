import pandas as pd
import numpy as np
from statistics import mean
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings


warnings.filterwarnings("ignore")

"""
### Importing data and checking data types + distribution ###

data = pd.read_csv('../data/ORIGINAL_DATA/stroke_dataset/healthcare-dataset-stroke-data.csv')

#We do not need the strole column as we will be using it as a label later (preventing leakage)
data = data.drop('stroke', axis=1)


print(data.dtypes)

col = data.columns

for i in col:
    print(f"### {i}  ###" )
    print(data[i].unique()[0:10])
    print()

### Dropping Values without label ###

df = data.loc[data['bmi'].notna()]

 Checking if it worked
print(data.shape)
print(df.shape)
print(5110-201)


### Histograms and distributions ###

# histogram function: x = variable, df = dataframe, ex = file name (for file saving)
def smp_hist(x, df, ex):
    hist = sns.histplot(
    x = str(x),
    data= df
    )

    hist.set(title = x + ' distribution plot', xlabel = x, ylabel = 'Count') 
    fig = hist.get_figure()
    path = "../output/" + str(ex) + "_" + str(x) + ".pdf"
    
    fig.savefig(
    path,
    backend="pdf",
    )
    
    print('histogram of', x, 'has been created and saved under', path)
    print()
    plt.clf() 


smp_hist('age', df, 'hist')
smp_hist('avg_glucose_level', df, 'hist')
smp_hist('bmi',df, 'hist')



# Checking distributions unsing Kolmogorov–Smirnov test:

for i in ['age', 'avg_glucose_level', 'bmi']:
    res = stats.kstest(df[i], stats.norm.cdf)
    print(f"probability that {i} is normaly distributed with no trf = {1 - res.statistic}")
    print()

    res = stats.kstest((df[i])**0.5, stats.norm.cdf)
    print(f"probability that {i} is normaly distributed with sqrt trf = {1 - res.statistic}")
    print()

    res = stats.kstest(np.log10(df[i]), stats.norm.cdf)
    print(f"probability that {i} is normaly distributed with log trf = {1 - res.statistic}")
    print()

# Very much not normally distributed...

# Problematic with smoking data, half of th smoking status data is labeled as unknown...
# I could write two models, 1 to predict for smoking status known and one for smoking status unknown... 
print((data.loc[data["smoking_status"] == 'Unknown']).shape)
print()




### One hot encoding ###

encoded_df = df

for i in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    encoded_df = pd.get_dummies(encoded_df, columns = [i], prefix = [i], drop_first = True )

 To check if encoding worked
print(encoded_df.columns)
print(encoded_df.dtypes)


### Training model with LM ###

# Splitting and scaling #
X = encoded_df.drop('bmi', axis=1)
y = encoded_df['bmi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data is not normally distributed so we want to robustscaler for transformation...
scaler = RobustScaler()
num_cols = ['age', 'avg_glucose_level']

X_train_scaled, X_test_scaled = X_train, X_test

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols]) 
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

LR = LinearRegression()

LR.fit(X_train_scaled, y_train)


y_mean = [mean(y)] * len(y_test)


def get_scores(model, X_train, y_train, X_test, y_test):

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    # evaluation
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = (sum((y_test - y_pred_test)**2)/len(y_test))**0.5

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = (mean_squared_error(y_train, y_pred_train, squared=True))**0.5

    r2_mean = r2_score(y_test, y_mean)
    rmse_mean = (sum((y_test - y_mean)**2)/len(y_test))**0.5

    print('Training set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_train, rmse_train))
    print('Test set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_test, rmse_test))
    print('mean score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_mean, rmse_mean))

get_scores(LR, X_train_scaled, y_train, X_test_scaled, y_test)


### Visualisation model from hw 04 ###
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


# Make a scatter plot (prediction vs ground truth)
# Add a dashed line indicating the optimal fit (you can use the add_identity function or you can implement this by yourself)
# Ensure the label is annotated
fig, ax = plt.subplots(figsize=(9, 6))

fig = add_identity(sns.scatterplot(x = y_test, y = LR.predict(X_test), color = 'green'), linestyle = 'dashed')
fig.set(xlabel = 'Real Bmi', ylabel = 'predicted Bmi', title = 'Prediction vs ground truth scatter plot')

plt.legend(labels = ['Data points', 'Optimal fit'])

# Save the figure in this specified path (upon submission this path has to be as written below!)
plt.savefig("../output/bmi_1.png", dpi=200)

print()
print()

cdf = pd.DataFrame(LR.coef_, X.columns, columns=['Coefficients']) #creating a dataframe with the coefficients
print(cdf)

### Optimising model, dropping unimportant coesfficients ###

encoded_df = encoded_df.drop(['Residence_type_Urban', 'gender_Male', 'id', 'work_type_Private', 'work_type_Self-employed' ], axis=1 )


########################## Same code with feature selection ############################## 
print()
print("###################### Model with feature selection ######################")
print()
### Training model with LM ###

# Splitting and scaling #
X = encoded_df.drop('bmi', axis=1)
y = encoded_df['bmi']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data is not normally distributed so we want to robustscaler for transformation...
scaler = RobustScaler()
num_cols = ['age', 'avg_glucose_level']

X_train_scaled, X_test_scaled = X_train, X_test

X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols]) 
X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

LR = LinearRegression()

LR.fit(X_train_scaled, y_train)


y_mean = [mean(y)] * len(y_test)


def get_scores(model, X_train, y_train, X_test, y_test):

    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    # evaluation
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = (sum((y_test - y_pred_test)**2)/len(y_test))**0.5

    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = (mean_squared_error(y_train, y_pred_train, squared=True))**0.5

    r2_mean = r2_score(y_test, y_mean)
    rmse_mean = (sum((y_test - y_mean)**2)/len(y_test))**0.5

    print('Training set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_train, rmse_train))
    print('Test set score: R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_test, rmse_test))
    print('Score using means (no model): R2 score: {:.3f}, RMSE: {:.3f}'.format(r2_mean, rmse_mean))

get_scores(LR, X_train_scaled, y_train, X_test_scaled, y_test)


### Visualisation model from hw 04 ###
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


# Make a scatter plot (prediction vs ground truth)
# Add a dashed line indicating the optimal fit (you can use the add_identity function or you can implement this by yourself)
# Ensure the label is annotated
fig, ax = plt.subplots(figsize=(9, 6))

fig = add_identity(sns.scatterplot(x = y_test, y = LR.predict(X_test), color = 'green'), linestyle = 'dashed')
fig.set(xlabel = 'Real Bmi', ylabel = 'predicted Bmi', title = 'Prediction vs ground truth scatter plot')

plt.legend(labels = ['Data points', 'Optimal fit'])

# Save the figure in this specified path (upon submission this path has to be as written below!)
plt.savefig("../output/bmi_2.png", dpi=200)

print()
print()

cdf = pd.DataFrame(LR.coef_, X.columns, columns=['Coefficients']) #creating a dataframe with the coefficients
print(cdf)

### Modyfing the data and saving it in a new Csv ###

data = pd.read_csv('../data/ORIGINAL_DATA/stroke_dataset/healthcare-dataset-stroke-data.csv')
truncation = data.loc[data['bmi'].isna()]
print(truncation)

col = truncation.columns
for i in col:
    print(f"### {i}  ###" )
    print(data[i].unique()[0:10])
    print()


encoded_df = truncation

for i in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    encoded_df = pd.get_dummies(encoded_df, columns = [i], prefix = [i])

print(encoded_df.columns)

X = encoded_df.drop(['stroke','Residence_type_Urban', 'gender_Male', 'id', 'work_type_Private', 'work_type_Self-employed', 'bmi' ], axis=1)
X_scaled = X
X_scaled[num_cols] = scaler.transform(X[num_cols])



y_pred = LR.predict(X_scaled)

print(len(y_pred))
"""


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

    df = data.loc[data['bmi'].notna()] #Dropping values without labels
    df = df.drop(['stroke'], axis = 1) # We do not want strokes in our model, as we will be estimating it later (Data leakage)

    encoded_df = df #One hot encoding the columns
    for i in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
        encoded_df = pd.get_dummies(encoded_df, columns = [i], prefix = [i], drop_first = True )

    #dropping irrelevant Columns for our model (feature selection)
    encoded_df = encoded_df.drop(['Residence_type_Urban', 'gender_Male', 'id', 'work_type_Private', 'work_type_Self-employed'], axis=1)

    print("########## Model Labels #########")
    print(encoded_df.columns)
    print()

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
    
bmi_main()