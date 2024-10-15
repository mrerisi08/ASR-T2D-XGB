from get_paramater_lists import get_param_list
from load_data import load_data_for_kfold_CV, load_data_to_numpy_randomized, get_columns
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score as auc

#
# def grid_search_param():
#     param_ranges = {"eta": [0.001, 0.01, 0.1, 0.5],
#                     "max_leaf": [5, 10, 15, 20, 31, 50],
#                     "min_data": [5, 10, 20, 50],
#                     "feature_fraction": [0.1, 0.25, 0.5, 0.75, 1],
#                     "reg_alpha": [0.0, 0.01, 1, 5, 25],
#                     "lambda": [0.0, 0.01, 1, 5, 25],
#                     "min_split_gain": [0, 50, 100, 250],
#                     "verbosity": -1,
#                     "max_bin": 2553,
#                     "objective": "binary",
#                     "metric": "auc",
#                     "num_threads": 9
#                     }
#
#     constant_hyperparams = {k: v for k, v in param_ranges.items() if not isinstance(v, list)}
#     list_hyperparams = {k: v for k, v in param_ranges.items() if isinstance(v, list)}
#
#     # Generate all combinations of list hyperparameters
#     list_keys = list(list_hyperparams.keys())
#     list_values = list(list_hyperparams.values())
#     combinations = list(itertools.product(*list_values))
#     final_hyperparams = []
#     for combination in combinations:
#         config = constant_hyperparams.copy()
#         config.update(zip(list_keys, combination))
#         final_hyperparams.append(config)
#
#     final_hyperparams = list(zip(list(range(len(final_hyperparams))),final_hyperparams))
#     return final_hyperparams

param_list = {"eta": [0.001, 0.01, 0.1, 0.5],
                    "max_leaf": [5, 10, 15, 20, 31, 50],
                    "min_data": [5, 10, 20, 50],
                    "feature_fraction": [0.1, 0.25, 0.5, 0.75, 1],
                    "reg_alpha": [0.0, 0.01, 1, 5, 25],
                    "lambda": [0.0, 0.01, 1, 5, 25],
                    "min_split_gain": [0, 50, 100, 250],
                    "verbosity": -1,
                    "max_bin": 2553,
                    "objective": "binary",
                    "metric": "auc",
                    "num_threads": 9
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
        train_data = lgbm.Dataset(X_train, label=y_train, feature_name=list(get_columns(label=False, for_lgbm=True)))
        lgbm_model = lgbm.train(params, train_data, 100)
        all_preds += list(lgbm_model.predict(X_test))

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
