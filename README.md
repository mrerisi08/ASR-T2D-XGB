# ASR-T2DM-ML

Repository for a research project done by Max Rerisi, mentored by Dr. Mathieu Ravaut.

Do not attempt to run 0_generate_pseudo_data.py. It does not have the real data from which to generate the pseudo data.

1_data_process.py has the normalization of numerical features in an unreachable if statement because the pseudo data by default is from 0-1. Similarly, processing binary features is disabled because the binary pseudo data was generated as 0/1 and didn't need to be processed like the original dataset.<br>
used_features.txt is a hard-coded list of features, because the pseudo data process may not mean the same features are eliminated / kept in 2_drop_>10_features. 2_data_filtering.py has the real code in an unreachable if statement, but otherwise pulls from the hard-coded list in 0_inputs.

SHAP plots included are generated from real data.

## Steps to Run
Here is the order in which to run the files.
1. 1_binary_conversion_and_normalization/1_data_process.py will make the raw dataset better suited for pandas.
2. 2_drop_less_than_10_positive_features/2_data_filtering.py will use the pre-generated set of features to make the actual used dataset smaller.
3. 3_train_xgboost/3_xgboost.py will train the XGBoost model using a 5-fold CV.
   1. Ensure line 37, save_models_to_file is set to True if you want ensembling to run.
4. 4_train_lightgbm/4_lightgbm.py will train the LightGBM model using a 5-fold CV.
   1. Ensure line 28, save_models_to_file is set to True if you want ensembling to run.
5. 5_train_catboost/5_catboost.py will train the CatBoost model using a 5-fold CV.
   1. Ensure line 27, save_models_to_file is set to True if you want ensembling to run.
6. 6_ensembling/6_ensemble.py will ensemble the 15 models trained and saved (3 models, 5 folds).
7. 7_process_data_for_feature_subsets/7_load_categories.py simply takes the CSV file in 0_inputs and converts it to a JSON for ease of use in future steps.
8. 8_train_on_feature_subsets/8_feature_subset_training.py requires the JSON created in the previous step to create the subsets. It will print a lot of training output, but the subsets and their corresponding AUC will be generated in the file 8_train_on_feature_subsets/feature_subset_data.tsv.
