import catboost as cat
import numpy as np
from sklearn.metrics import roc_auc_score as auc
from load_data import load_data_to_numpy_randomized, load_data_for_kfold_CV
from shap_object import contribs_to_shap_object
import shap
import matplotlib.pyplot as plt
import pickle as pkl
from get_hyperparameters import catboost_hyperparameters

FOLDS = load_data_for_kfold_CV()

all_preds = []
all_contributions = []

for index, fold in enumerate(FOLDS):
    X_train, y_train, X_test, y_test = fold
    cat_model = cat.CatBoostClassifier(**catboost_hyperparameters)
    cat_model.fit(X_train, y_train)
    all_preds += list(np.hsplit(cat_model.predict_proba(X_test), 2)[1])

    explainer = shap.Explainer(cat_model)
    shap_values = explainer(X_test)
    all_contributions.append(shap_values)

    save_models_to_file = True
    if save_models_to_file:
        with open(f'5_catboost_models/catboost_model_{index}_fold.pkl', 'wb') as file:
            pkl.dump(cat_model, file)

y = load_data_to_numpy_randomized()[1]
print(auc(y, all_preds))

# noinspection PyUnreachableCode
if False:
    # Generates a SHAP plot, and displays or saves it. Disabled because pseudo data leads to a nonsensical SHAP plot.
    combined_shap_values_obj = contribs_to_shap_object(all_contributions)

    show_plot = True  # False saves plot to file, True shows it

    shap.plots.beeswarm(combined_shap_values_obj, show=show_plot, max_display=20)

    if not show_plot:
        plt.savefig("5_catboost_shap_beeswarm_plot.png", bbox_inches="tight")
