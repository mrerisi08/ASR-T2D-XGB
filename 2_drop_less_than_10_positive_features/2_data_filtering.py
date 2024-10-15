# remove features with < 10 positive cases
import pandas as pd

# load the dataframe from the previous step
df = pd.read_csv("../1_binary_conversion_and_normalization/1_binary_and_normalized.csv", index_col=0)

# noinspection PyUnreachableCode
if False:
    # disabled because the class imbalances may be different in  pseudo data
    # drops columns with less than 10 positive instances
    cols_to_drop = []
    for col in list(df.drop(["Min Diastolic", "Mean Diastolic", "Max Diastolic", "Min Systolic", "Mean Systolic",
                             "Max Systolic", "Age", "LABEL"], axis=1).columns):
        ones = list(df[col].iloc).count(1)
        if ones < 10:
            cols_to_drop.append(col)

    df = df.drop(cols_to_drop, axis=1)
else:
    # opens file with hard-coded features to use and makes the dataframe just those features
    with open('../0_inputs/used_features.txt', 'r') as fp:
        cols = fp.readlines()
        cols_to_keep = []
        for col in cols:
            cols_to_keep.append(col[:-1])  # removing newline character.

    df = df[cols_to_keep]

# saves dataframe to a file
df.to_csv("2_bin_norm_>10.csv")
