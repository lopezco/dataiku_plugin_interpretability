from dataiku.customrecipe import *
import dataiku
from dataiku import pandasutils as pdu
import pandas as pd
import numpy as np
import shap

#############################
# PARAMETERS
#############################
# Inputs
model_name   = get_input_names_for_role('Model')[0].split('.')
dataset_name = get_input_names_for_role('Dataset')[0].split('.')[1]
# Configuration
conf = get_recipe_config()
model_version = conf.get('model_version', 'active').lower()
n_samples = int(conf.get('n_samples', -1))
idx_variables = conf.get('copy_cols', None)
compute_importance = len(get_output_names_for_role('Shap_imp'))
# Outputs
out_dataset_name = get_output_names_for_role('Shap_values')[0].split('.')[1]
if compute_importance:
    out_imp_name = get_output_names_for_role('Shap_imp')[0].split('.')[1]

#############################
# Load inputs
#############################
# Load model
model = dataiku.Model(lookup=model_name[1], project_key=model_name[0])
# Get version_id of the 'active' version if model_version != 'active' or empty
version_id = ([version['versionId'] for version in model.list_versions() if version['active']][0]) if model_version in (u'active', u'') else model_version
# Get predictor from selected version
predictor = model.get_predictor(version_id=version_id)
# Load the dataset
limit = None if n_samples < 0 else n_samples
dku_dataset = dataiku.Dataset(dataset_name)
shap_values_output = dataiku.Dataset(out_dataset_name)
shap_imp_output = dataiku.Dataset(out_imp_name)
n_rows = 0
# Is classification or regression?
is_regression = len(predictor.classes) == 0
is_classification = not is_regression
is_multiclass = is_classification and len(predictor.classes) > 2
if is_classification:
    if is_multiclass:
        classes_target = predictor.classes
    else:  # binary
        classes_target = [predictor.classes[1]]
        shap_values_list = [shap_values_list]
else: # regression
    classes_target = [None]
    shap_values_list = [shap_values_list]

#############################
# Compute shap values in iterative mode to avoid memory problems
#############################
with shap_values_output.get_writer() as writer:
    first_time = True
    shap_sum_buffer = []
    shap_count_buffer = None
    for df in dku_dataset.iter_dataframes(limit=limit):
        # Proecess the dataset before computing interpretations
        df_processed = pd.DataFrame(predictor.preprocess(df)[0], columns=predictor.get_features())
        # Interpret
        # Get interpreter
        tree_explainer = shap.TreeExplainer(predictor._clf)
        # Get shap values for each class: binary = np.array for class 1, multiclass = list of np.array for evary class
        shap_values_list = tree_explainer.shap_values(df_processed)
        # Create a dataframe for shap values of each classs and compute importance for each class if necessary
        shap_values_buffer = []
        for idx, c in enumerate(classes_target):
            shap_values = shap_values_list[idx]
            if df_processed.shape != shap_values.shape:
                raise ValueError('Shap values from {} have the wrong size. Got {}, expected {}'.format(c, df_processed.shape, shap_values.shape))
            del df_processed
            df_shap_values = pd.DataFrame(shap_values, columns=predictor.get_features())
            # Recipe output for shap values
            # Add copy of variables (if any)
            if idx_variables is not None:
                df_shap_values = pd.merge(df[idx_variables].reset_index(drop=True), df_shap_values.reset_index(drop=True), left_index=True, right_index=True)
            del df
            if is_classification:
                df_shap_values['TARGET_CLASS'] = c
            shap_values_buffer.append(df_shap_values)
            # Global Importance
            if compute_importance:
                if idx_variables is not None:
                    tmp = df_shap_values.loc[:, ~df_shap_values.columns.isin(idx_variables)]
                else:
                    tmp = df_shap_values
                del df_shap_values
                if is_classification:
                    g = tmp.groupby('TARGET_CLASS')
                    tmp = g.agg(lambda x: np.nansum(np.absolute(x)))
                    tmp_nrows = g.count().iloc[:, 0].to_dict()
                else:
                    tmp = tmp.agg(lambda x: np.nansum(np.absolute(x))).to_frame('importance').transpose()
                tmp.index.name = 'idx'
                shap_sum_buffer.append(tmp)
                # Add count
                if shap_count_buffer is None:
                    shap_count_buffer = tmp_nrows
                else:
                    for k, v in tmp_nrows.items():
                        shap_count_buffer[k] += v

        # Write shap values output
        df_out = pd.concat(shap_values_buffer, axis=0)
        del shap_values_buffer
        n_rows += len(df_out)

        if first_time:
            shap_values_output.write_schema_from_dataframe(df_out)
            first_time = False

        writer.write_dataframe(df_out)
        del df_out

#############################
# Aggregate importance
#############################
if compute_importance:
    df_out_imp = (pd.concat(shap_sum_buffer, axis=0)
                  .reset_index().groupby('idx').sum(axis=0)
                  .stack().reset_index())
    df_out_imp.columns = ['TARGET_CLASS', 'VARIABLE', 'IMPORTANCE'] if is_classification else ['Index', 'VARIABLE', 'IMPORTANCE']

    if is_classification:
        for c in df_out_imp['TARGET_CLASS'].unique():
            df_out_imp.loc[df_out_imp['TARGET_CLASS'] == c, 'IMPORTANCE'] /= shap_count_buffer[c]
    else:  # regression
        df_out_imp = df_out_imp[['VARIABLE', 'IMPORTANCE']]
        df_out_imp['IMPORTANCE'] /= n_rows
    # Recipe output for importance
    shap_imp_output.write_with_schema(df_out_imp)
