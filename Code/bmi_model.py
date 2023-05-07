import pandas as pd
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
import warnings


warnings.filterwarnings("ignore")


### Importing data and checking data types + distribution ###

data = pd.read_csv('../data/ORIGINAL_DATA/stroke_dataset/healthcare-dataset-stroke-data.csv')

#We do not need the strole column as we will be using it as a label later (preventing leakage)
data.columns.drop('stroke')


print(data.dtypes)

col = data.columns

for i in col:
    print(f"### {i}  ###" )
    print(data[i].unique()[0:10])
    print()

### Dropping Values without label ###

df = data.loc[data['bmi'].notna()]

""" Checking if it worked
print(data.shape)
print(df.shape)
print(5110-201)
"""

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

""" To check if encoding worked
print(encoded_df.columns)
print(encoded_df.dtypes)
"""

### Training model ###

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Data is not normally distributed so we want to robustscaler for transformation...

