import pandas as pd
import json

# loads the manually made CSV of feature categorization
categories_df = pd.read_csv("../0_inputs/feature_categories.csv")

cats_dict = {}
for col_dex in range(len(list(categories_df.iloc))):  # iterates through all the indices in the length of the CSV
    for cat in ["Cats", "Cats 2"]:  # doing this twice in case the feature has two categorizations
        cat = categories_df[cat].iloc[col_dex]  # gets the categorization
        if not pd.isna(cat):  # if the category is not NA (for features that don't have a second category)
            if cat not in cats_dict:  # if the category isn't in the dictionary already, add it as an empty list
                cats_dict[cat] = []
            cats_dict[cat].append(categories_df["New Names"].iloc[
                                      col_dex])  # add to the dictionary at the particular category the feature's name

# dump the dictionary into a JSON file
fp = open("feature_categories.json", 'w')
json.dump(cats_dict, fp, indent=4)
fp.close()
