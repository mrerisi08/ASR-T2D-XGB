import lightgbm as lgbm
from load_data import load_data_for_kfold_CV, load_dataframe, get_columns, load_data_to_numpy_randomized
from get_hyperparameters import lightgbm_hyperparameters
from sklearn.metrics import roc_auc_score as auc
import shap
from shap_object import contribs_to_shap_object
import matplotlib.pyplot as plt
import pickle as pkl

# loading data a little uniquely due to an encoding error in feature names
df = load_dataframe()
FOLDS = load_data_for_kfold_CV(df=df)

all_pred = []
all_contributions = []

for index, fold in enumerate(FOLDS):
    X_train, y_train, X_test, y_test = fold
    train_data = lgbm.Dataset(X_train, label=y_train, feature_name=list(get_columns(label=False, for_lgbm=True)))

    lgbm_model = lgbm.train(lightgbm_hyperparameters, train_data, 100)
    all_pred += list(lgbm_model.predict(X_test))

    explainer = shap.Explainer(lgbm_model)
    shap_values = explainer(X_test)
    all_contributions.append(shap_values)

    save_models_to_file = True
    if save_models_to_file:
        with open(f'4_lightgbm_models/lightgbm_model_{index}_fold.pkl', 'wb') as file:
            pkl.dump(lgbm_model, file)

y = load_data_to_numpy_randomized()[1]
print(auc(y, all_pred))

# noinspection PyUnreachableCode
if False:
    # Generates a SHAP plot, and displays or saves it. Disabled because pseudo data leads to a nonsensical SHAP plot.
    combined_shap_values_obj = contribs_to_shap_object(all_contributions)

    show_plot = True  # False saves plot to file, True shows it

    shap.plots.beeswarm(combined_shap_values_obj, show=show_plot, max_display=20)

    if not show_plot:
        plt.savefig("4_lightgbm_shap_beeswarm_plot.png", bbox_inches="tight")
