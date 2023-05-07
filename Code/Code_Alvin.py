### Basic data wrangling ###

import pandas as pd

data = pd.read_csv('../data/ORIGINAL_DATA/stroke_dataset/healthcare-dataset-stroke-data.csv')

print(data[1:5])

columns = data.columns

print()
print(columns)
print()


print("###### Missing data ######")

for i in columns:
    print(i, ": ", data[i].isna().sum() ,"missing data" )

print()


print(data.shape, "number of rows, columns")

print("")


print("number of people with stroke:", data.stroke.sum())
print("")

Indexes = (data["bmi"].isna())

Missing_bmi = (data[Indexes])

print(Missing_bmi.stroke.sum(), "= Number of people with missing bmi that have had a stroke")

print("")
