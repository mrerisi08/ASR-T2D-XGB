# hyperparameters pre-generated after an extensive grid search

xgboost_hyperparameters = {
    'max_depth': 5,
    'eta': 0.5,
    'min_child_weight': 5,
    'alpha': 5,
    'gamma': 0,
    'lambda': 25,
    'scale_pos_weight': 10,
    'subsample': 1,
    'colsample_bytree': 1,
    'colsample_bylevel': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'exact',
    'nthread': 8,
    'verbosity': 0,
    'num_parallel_tree': 1,
    'seed': 42
}

lightgbm_hyperparameters = {
    "eta": 0.1,
    "max_leaf": 5,
    "min_data": 20,
    "feature_fraction": 1,
    "reg_alpha": 0.01,
    "lambda": 0.01,
    "min_split_gain": 0,
    "verbosity": -1,
    "max_bin": 2553,
    "objective": "binary",
    "metric": "auc",
    "num_threads": 9
}

catboost_hyperparameters = {
    'depth': 10,
    'iterations': 250,
    'learning_rate': 0.01,
    'l2_leaf_reg': 25,
    'scale_pos_weight': 10,
    'verbose': False
}
