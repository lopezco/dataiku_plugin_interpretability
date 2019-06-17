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

# Outputs
out_dataset_name = get_output_names_for_role('Shap_values')[0].split('.')[1]

if len(get_output_names_for_role('Shap_imp')):
    out_imp_name = get_output_names_for_role('Shap_imp')[0].split('.')[1]

#############################
# Loan inputs
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
df = dku_dataset.get_dataframe(limit=limit)

#############################
# Preprocess
#############################
# Proecess the dataset before computing interpretations
df_processed = pd.DataFrame(predictor.preprocess(df), columns=predictor.get_features())

#############################
# Interpret
#############################
# Get interpreter
tree_explainer = shap.TreeExplainer(predictor.clf)

# Get shap values for each class: binary = np.array for class 1, multiclass = list of np.array for evary class
shap_values_list = tree_explainer.shap_values(df_processed)

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
                                   
# Create a dataframe for shap values of each classs and compute importance for each class if necessary
shap_values_buffer = []
shap_imp_buffer = []
for idx, c in enumerate(classes_target):
    shap_values = shap_values_list[idx]
    
    if df_processed.shape != shap_values.shape:
        raise ValueError('Shap values from {} have the wrong size. Got {}, expected {}'.format(c, df_processed.shape, shap_values.shape))

    df_shap_values = pd.DataFrame(shap_values, columns=predictor.get_features())

    # Compute importance if second output is selected
    if len(get_output_names_for_role('Shap_imp')):
        # Get description of the dataframe
        df_desc = df_shap_values.describe().transpose()
        df_desc.index.name = 'Variable'

        # Add importance as the mean(abs(shap_values))
        tmp = df_shap_values.agg(lambda x: np.nanmean(np.absolute(x)))
        tmp = tmp.to_frame('Importance')
        df_shap_imp = pd.merge(df_desc, tmp, left_index=True, right_index=True).reset_index()
        
        if is_classification:
            df_shap_imp['TARGET_CLASS'] = c
            
        shap_imp_buffer.append(df_shap_imp)

    # Recipe output for shap values
    # Add copy of variables (if any)
    if idx_variables is not None:
        df_shap_values = pd.merge(df[idx_variables].reset_index(drop=True), df_shap_values.reset_index(drop=True), left_index=True, right_index=True)

    if is_classification:
        df_shap_values['TARGET_CLASS'] = c
    
    shap_values_buffer.append(df_shap_values)

# Write outputs
df_out = pd.concat(shap_values_buffer, axis=0)

if len(get_output_names_for_role('Shap_imp')):
    df_out_imp = pd.concat(shap_imp_buffer, axis=0)

    # Recipe output for importance
    shap_imp_output = dataiku.Dataset(out_imp_name)
    shap_imp_output.write_with_schema(df_out_imp)

shap_values_output = dataiku.Dataset(out_dataset_name)
shap_values_output.write_with_schema(df_out)
