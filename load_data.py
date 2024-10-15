import numpy as np
import re
import pandas as pd
from seed_everything import seed_everything
import os

# manually determined data leaks
DATA_LEAKS = ["GLYCATED HEMOGLOBIN#90.28.1", "INSULIN GLARGINE#INSULINA GLARGINE", "METFORMIN#METFORMINA"]


def load_dataframe(for_lgbm: bool = False, raw: bool = False, label: bool = False):
    """
    Load data from the newest CSV with all processing performed.
    Remove pre-identified data leaks.
    Return the dataframe.
    """

    df_path = os.path.join(os.path.dirname(__file__), '2_drop_less_than_10_positive_features', '2_bin_norm_>10.csv')
    df = pd.read_csv(df_path, index_col=0)

    # data leak(s)
    try:
        if not raw:
            df = df.drop(DATA_LEAKS, axis=1)
        elif label:
            df = df["LABEL"]
    except KeyError:
        pass

    if for_lgbm:
        df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))  # fixes encoding error LGBM has

    return df


def load_data_to_numpy_randomized(seed: int = 42, df=False, label: bool = False):
    """
    Load dataframe.
    Convert that to two numpy arrays (notated as X and y).
    Shuffle them according to a fixed random permutation.
    Return the data as X, y.
    Other parameters are used for specific instances that need different datasets (e.g. for feature subsetting).
    """
    if not isinstance(df, pd.DataFrame):
        df = load_dataframe()

    seed_everything(seed)

    if label:
        y = df["LABEL"].to_numpy()
        p = np.random.permutation(len(y))
        y = df[p]
        return y

    X = df.drop("LABEL", axis=1).to_numpy()
    y = df["LABEL"].to_numpy()

    p = np.random.permutation(len(y))
    X = X[p]
    y = y[p]
    return X, y


def load_data_for_kfold_CV(fold_count: int = 5, df=False, label: bool = False):
    """
    Load data as X, y (already randomized).
    Split data for first 4 (0-3) folds then "round-up" for the last (4th) fold.
        ** Actually configured to be adaptable to any number of folds **
    Return a list (ordered from 0th to 4th fold) containing tuples: (X_train, y_train, X_test, y_test)
    """

    X, y = load_data_to_numpy_randomized(df=df, label=label)

    fold_size = int(len(y) / fold_count)

    out = []
    for j in range(fold_count):
        start = j * fold_size
        end = (j + 1) * fold_size
        if j == fold_count - 1:
            X_train, X_test = X[:start], X[start:]
            y_train, y_test = y[:start], y[start:]
        else:
            X_train, X_test = [*X[:start], *X[end:]], X[start:end]
            y_train, y_test = [*y[:start], *y[end:]], y[start:end]
        sub_out = (X_train, y_train, X_test, y_test)
        out.append(tuple([np.array(a) for a in sub_out]))

    return out


def get_columns(label=True, for_lgbm=False, just_binary=False):
    """
    Load the dataframe.
    Return the columns as the pandas dtype Index.
    """
    df = load_dataframe(for_lgbm=for_lgbm)
    if not label:
        df = df.drop("LABEL", axis=1)
    if just_binary:
        df = df.drop(
            ["Min Diastolic", "Max Diastolic", "Mean Diastolic", "Min Systolic", "Max Systolic", "Mean Systolic",
             "Age"], axis=1)
    return df.columns
