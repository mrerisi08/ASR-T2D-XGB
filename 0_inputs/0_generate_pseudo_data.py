# DO NOT RUN
# there is no dataframe as input

import numpy as np
import pandas as pd
import random
from seed_everything import seed_everything

# sets random seed
seed_everything(42)

# initializes the dataframe
df = None

# counts negative and positive values in the dataset
binary_counts = {}
for dex, col in enumerate(list(df.drop(
        ["Min Diastolic", "Max Diastolic", "Mean Diastolic", "Min Systolic", "Max Systolic", "Mean Systolic", "Age"],
        axis=1).columns)):
    ones, zeros = 0, 0
    for row in df[col]:
        if row == 1:
            ones += 1
        elif row == 0:
            zeros += 1

    binary_counts[col] = [ones, zeros]

# counts the valid and invalid values for each feature in the dataset
num_counts = {}
for col in ["Min Diastolic", "Max Diastolic", "Mean Diastolic", "Min Systolic", "Max Systolic", "Mean Systolic", "Age"]:
    invalid, valid = 0, 0
    for row in df[col]:
        if pd.isna(row):
            invalid += 1
        else:
            valid += 1
    num_counts[col] = [invalid, valid]

ROWS = 1000  # how many rows in the new data
newdf = {}  # initializes a dictionary which will be converted to a dataframe later

# generates a random column of data using the weights of the corresponding column in the real dataset
for col in binary_counts:
    newdf[col] = random.choices([1, 0], weights=binary_counts[col], k=ROWS)

# generates a random column of data if the random choice is 0, and an invalid value if it's 1
for col in num_counts:
    validity = random.choices([1, 0], weights=num_counts[col], k=ROWS)
    output = []
    for row in validity:
        if row == 1:
            output.append(999)
        else:
            output.append(np.random.uniform(0, 1))
    newdf[col] = output

# saves and then deletes the "LABEL" column, which is diabetic state
label = newdf["LABEL"]
del newdf["LABEL"]

# creates the dataframe and ensure label is correctly processed (also adds it as the last column)
newdf = pd.DataFrame(data=newdf)
newdf["LABEL"] = label

# converts the "good" column names to the "bad" ones to make the following files relevant
namesdf = open('column_names_translated.csv', 'r').readlines()

translate_code = {}
for a in namesdf:
    a = a.split(',')
    translate_code[a[1][:-1]] = a[0]

newdf = newdf.rename(columns=translate_code)

# saves the dataframe to a file
newdf.to_csv("pseudo_data.csv")
