import shap
import numpy as np
from load_data import get_columns


def contribs_to_shap_object(contribs):
    """
    Takes the contributions generated by each model and applies a series of steps to it to return the
    shap.Explanation object which can be used to create a SHAP beeswarm plot.
    """
    combined_shap_values = np.concatenate([sv.values for sv in contribs], axis=0)
    combined_base_values = np.concatenate([sv.base_values for sv in contribs], axis=0)
    combined_data = np.concatenate([sv.data for sv in contribs], axis=0)
    return shap.Explanation(values=combined_shap_values, base_values=combined_base_values, data=combined_data,
                            feature_names=list(get_columns(label=False)))
