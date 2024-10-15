import catboost as cat
import numpy as np
from sklearn.metrics import roc_auc_score as auc
from load_data import load_data_to_numpy_randomized, load_data_for_kfold_CV
from get_paramater_lists import get_param_list

param_list = {
    "depth": [3, 5, 10],
    "iterations": [10, 50, 100, 250, 1000],
    "learning_rate": [0.001, 0.01, 0.1, 0.5],
    "l2_leaf_reg": [0, 1, 2, 3, 5, 10, 25],
    "scale_pos_weight": [1, 10, 100],
    "verbose": False
}

PARAMS = get_param_list(param_list)
print(len(PARAMS))
FOLDS = load_data_for_kfold_CV()

bst_auc = -1
bst_param = None

y = load_data_to_numpy_randomized()[1]
for DEX, params in enumerate(PARAMS):

    with open("iteration.txt", 'w') as fp:
        fp.write(str(DEX))

    all_preds = []
    for fold in FOLDS:
        X_train, y_train, X_test, y_test = fold
        cat_model = cat.CatBoostClassifier(**params)
        cat_model.fit(X_train, y_train)
        all_preds += list(np.hsplit(cat_model.predict_proba(X_test), 2)[1])

    auc_val = auc(y, all_preds)

    if auc_val > bst_auc:
        bst_param = params
        bst_auc = auc_val
        with open("best_param.txt", 'w') as fp:
            fp.write(str(bst_param))

        with open("best_auc.txt", 'w') as fp:
            fp.write(str(bst_auc))

    print(DEX, auc_val)

print(bst_auc)
print(bst_param)
