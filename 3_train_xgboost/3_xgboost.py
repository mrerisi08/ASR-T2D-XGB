import shap
import xgboost as xgb
from load_data import load_data_for_kfold_CV, load_data_to_numpy_randomized
from get_hyperparameters import xgboost_hyperparameters
from sklearn.metrics import roc_auc_score as auc
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from shap_object import contribs_to_shap_object

# initializes lists to store data later
all_predictions = []
all_contributions = []

# gets the data for the 5-fold CV
FOLDS = load_data_for_kfold_CV()

for index, fold in enumerate(FOLDS):
    # load the fold data
    X_train, y_train, X_test, y_test = fold
    # convert numpy arrays to xgboost data format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    eval_list = [(dtrain, 'train'), (dtest, 'eval')]

    # train the model
    xgb_model = xgb.train(xgboost_hyperparameters, dtrain, 500, eval_list, early_stopping_rounds=1000)

    # take the model's predictions based on the best iteration
    all_predictions += list(xgb_model.predict(dtest, iteration_range=(0, xgb_model.best_iteration + 1)))

    # get contributions for SHAP
    explainer = shap.Explainer(xgb_model)
    shap_values = explainer(X_test)
    all_contributions.append(shap_values)

    # save model to file
    save_models_to_file = True
    if save_models_to_file:
        with open(f'3_xgboost_models/xgboost_model_{index}_fold.pkl', 'wb') as file:
            pkl.dump(xgb_model, file)

# calculate the AUC value
y = load_data_to_numpy_randomized()[1]
auc_val = auc(y, all_predictions)
print(auc_val)

# noinspection PyUnreachableCode
if False:
    # Generates a SHAP plot, and displays or saves it. Disabled because pseudo data leads to a nonsensical SHAP plot.
    combined_shap_values_obj = contribs_to_shap_object(all_contributions)

    # whether to display the plot and / or save it to a file
    show_plot = True  # False saves plot to file, True shows it

    shap.plots.beeswarm(combined_shap_values_obj, show=show_plot, max_display=20)

    if not show_plot:
        plt.savefig("3_xgboost_shap_beeswarm_plot.png", bbox_inches="tight")
