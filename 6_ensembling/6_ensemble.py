import pickle as pkl
import xgboost as xgb
from sklearn.metrics import roc_auc_score as auc
import numpy as np
from load_data import load_data_for_kfold_CV, load_data_to_numpy_randomized, load_dataframe

FOLDS = load_data_for_kfold_CV()

xgb_preds = []
lgbm_preds = []
cat_preds = []

for index, fold in enumerate(FOLDS):
    X_train, y_train, X_test, y_test = fold

    # xgboost loading and predicting
    with open(f'../3_train_xgboost/3_xgboost_models/xgboost_model_{index}_fold.pkl', 'rb') as file:
        xgb_mdl: xgb.core.Booster = pkl.load(file)

    dtest = xgb.DMatrix(X_test, label=y_test)
    xgb_preds += list(xgb_mdl.predict(dtest, iteration_range=(0, xgb_mdl.best_iteration + 1)))

    # lightgbm loading and predicting
    with open(f'../4_train_lightgbm/4_lightgbm_models/lightgbm_model_{index}_fold.pkl', 'rb') as file:
        lgbm_mdl = pkl.load(file)

    lgbm_preds += list(lgbm_mdl.predict(X_test))

    # catboost loading and predicting
    with open(f'../5_train_catboost/5_catboost_models/catboost_model_{index}_fold.pkl', 'rb') as file:
        cat_mdl = pkl.load(file)

    cat_preds += list(np.hsplit(cat_mdl.predict_proba(X_test), 2)[1])

# long way of averaging predictions, but it works
all_predictions = []
for a, b, c in zip(xgb_preds, lgbm_preds, cat_preds):
    all_predictions.append((a + b + c) / 3)

# calculate AUC based on averaged predictions
y = load_data_to_numpy_randomized()[1]
auc_val = auc(y, all_predictions)
print(auc_val)
