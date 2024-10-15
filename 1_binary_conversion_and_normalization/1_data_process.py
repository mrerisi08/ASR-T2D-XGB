# process data for binary conversion and normalization
import numpy as np
import pandas as pd

# load first row of data for headers
headers = pd.read_csv("../0_inputs/pseudo_data.csv").iloc[0]
headers = list(headers)
headers = [*headers[:-2], headers[-1]]

# reload the data, drop the empty column
df = pd.read_csv("../0_inputs/pseudo_data.csv").drop("Unnamed: 0", axis=1)

# rename the columns, from generic indices to their codes
df.rename(columns={str(a): b for a, b in zip(range(1, 1863), headers[:-1])}, inplace=True)
df.rename(columns={
    "Unnamed: 1863": headers[-1]
}, inplace=True)

# parse file with old vs new column names
file = open("../0_inputs/column_names_translated.csv", 'r')
lines = file.readlines()[1:]
file.close()
old = []
new = []
for line in lines:
    x = line.split(",")
    old.append(x[0])
    new.append(x[1][:-1])

# rename the columns accordingly
df.rename(columns={a: b for a, b in zip(old, new)}, inplace=True)

# find and count duplicate columns
duplicate_cols = {}
for col in list(df.columns):
    if list(df.columns).count(col) != 1 and col not in duplicate_cols:
        duplicate_cols[col] = list(df.columns).count(col)

# make a list that takes all columns and renames the duplicates
old_cols = list(df.columns)
for dupe_col in duplicate_cols:
    for i in range(duplicate_cols[dupe_col]):
        old_cols[old_cols.index(dupe_col)] = f"{dupe_col}##{i}"

# assigns the new list to the dataframe
df.columns = old_cols

# features that don't need to be converted to binary
nb_features = ["Min Diastolic", "Mean Diastolic", "Max Diastolic", "Min Systolic", "Mean Systolic", "Max Systolic",
               "Gender", "Age", "LABEL"]

# disabled because pseudo data is already binary
# noinspection PyUnreachableCode
if False:
    # for all the binary columns, take any non-zero value and convert it to 1 *really slow*
    for col in list(df.drop(nb_features, axis=1).columns):
        df = df.astype({
                           col: "int"
                       })
        df.loc[df[col] != 0, col] = 1

    # converts 2 to 0, so 1 is male 0 is female
    df = df.astype({
                       "Gender": "int"
                   })
    df.loc[df["Gender"] == 2, "Gender"] = 0

# convert appropriate datatypes
df = df.astype({a: "category" for a in list(df.drop(nb_features, axis=1).columns)})
df = df.astype({a: "float" for a in nb_features})
df = df.astype({a: "category" for a in ["Gender", "LABEL"]})

# 999 values are 0 for BP readings, gender, age and LABEL are non issues
for col in nb_features:
    df.loc[df[col] == 999, col] = np.nan

# disabled because pseudo data is already from 0 to 1
# noinspection PyUnreachableCode
if False:
    # normalize all 7 numerical features
    for col in ["Min Diastolic", "Mean Diastolic", "Max Diastolic", "Min Systolic", "Mean Systolic", "Max Systolic",
                "Age"]:
        df[col] = df[col] / abs(df[col]).max()

# save dataframe to a file
df.to_csv("1_binary_and_normalized.csv")
