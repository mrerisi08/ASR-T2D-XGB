from feature_categories import df_from_categories
from load_data import load_data_for_kfold_CV, load_data_to_numpy_randomized
import itertools
import xgboost as xgb
from get_hyperparameters import xgboost_hyperparameters
from sklearn.metrics import roc_auc_score as auc

# the categories that exist
categories = ["Medical History", "SDoH", "Procedure", "Test", "Medication"]
subsets = []
# generates all the subsets with anywhere from 1 to 5 categories (5 to ensure it matches with 3_xgboost.py)
for ct in range(1, 6):
    subsets += itertools.combinations(categories, ct)

for subset in subsets:
    subset = list(subset)  # tuple to list
    data = df_from_categories(cats=subset)  # load the data from a list of categories
    col_count = len(list(data.columns))  # count the number of columns included
    data = load_data_for_kfold_CV(df=data)  # loads the data into 5-fold CV
    all_predictions = []

    # trains XGBoost model using same hyperparameters as 3_xgboost.py using the same process.
    for fold in data:
        X_train, y_train, X_test, y_test = fold

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        eval_list = [(dtrain, 'train'), (dtest, 'eval')]

        xgb_model = xgb.train(xgboost_hyperparameters, dtrain, 500, eval_list, early_stopping_rounds=1000)

        all_predictions += list(xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1)))

    y = load_data_to_numpy_randomized()[1]
    auc_val = auc(y, all_predictions)
    print(auc_val)

    # noinspection PyUnreachableCode
    if False:
        # save the subset counts to a file, disabled because it's already there
        with open("feature_subset_col_count.tsv", 'a') as fp:
            out = ", ".join(subset)
            out += f"\t{col_count}\n"
            fp.write(out)

    # save which features were used and their AUC to a file
    with open("feature_subset_data.tsv", 'a') as fp:
        out = ""
        for cat in subset:
            out += f"{cat}, "
        out = out[:-2]
        out += f"\t{auc_val}\n"
        fp.write(out)
