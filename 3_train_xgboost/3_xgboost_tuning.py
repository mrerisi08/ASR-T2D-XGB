import xgboost as xgb
from load_data import load_data_for_kfold_CV, load_data_to_numpy_randomized
from sklearn.metrics import roc_auc_score as auc
from get_paramater_lists import get_param_list

# initializes a dictionary of arbitrary hyperparameter values
param_list = {
    "max_depth": [5, 15, 25, 50],
    "eta": [0.001, 0.1, 0.5],
    "subsample": 1,
    "colsample_bytree": 1,
    "colsample_bylevel": 1,
    "min_child_weight": [5, 15, 25, 50],
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "exact",
    "alpha": [0, 0.01, 1, 5, 25],
    "gamma": [0, 0.01, 1, 5, 25],
    "lambda": [0, 0.01, 1, 5, 25],
    "scale_pos_weight": [1, 5, 10],
    "nthread": 8,
    "verbosity": 0,
    "num_parallel_tree": 1
}

# gets all the possible sets of the above feature ranges, to use for a grid search.
PARAMS = get_param_list(param_list)

# gets the data for the 5-fold CV
FOLDS = load_data_for_kfold_CV()

# variables to store best iteration
bst_auc = -1
bst_param = None

# gets the true output values for each feature
y = load_data_to_numpy_randomized()[1]
# begins grid search loop
for params in PARAMS:
    all_preds = []
    # initializes 5-fold CV in the same way as in 3_xgboost.py
    for fold in FOLDS:
        X_train, y_train, X_test, y_test = fold
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        eval_list = [(dtrain, 'train'), (dtest, 'eval')]
        xgb_model = xgb.train(params, dtrain, 100, eval_list, early_stopping_rounds=1000)
        all_preds += list(xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1)))

    auc_val = auc(y, all_preds)
    # checks if the new AUC value is better than previous
    if auc_val > bst_auc:
        bst_param = params
        bst_auc = auc_val
        with open("best_param.txt", 'w') as fp:  # saves the best parameters to a file in case program errors later on
            fp.write(str(bst_param))

        with open("best_auc.txt", 'w') as fp:  # saves the best AUC to a file in case program errors later on
            fp.write(str(bst_auc))

    print(auc_val)

print(bst_auc)
print(bst_param)
