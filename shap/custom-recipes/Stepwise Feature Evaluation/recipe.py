# Code for custom code recipe Stepwise Feature Evaluation (imported from a Python recipe)

# To finish creating your custom recipe from your original PySpark recipe, you need to:
#  - Declare the input and output roles in recipe.json
#  - Replace the dataset names by roles access in your code
#  - Declare, if any, the params of your custom recipe in recipe.json
#  - Replace the hardcoded params values by acccess to the configuration map

# See sample code below for how to do that.
# The code of your original recipe is included afterwards for convenience.
# Please also see the "recipe.json" file for more information.

# import the classes for accessing DSS objects from the recipe
import dataiku
# Import the helpers for custom recipes
from dataiku.customrecipe import *
from feature_evaluation import run

# Inputs and outputs are defined by roles. In the recipe's I/O tab, the user can associate one
# or more dataset to each input and output role.
# Roles need to be defined in recipe.json, in the inputRoles and outputRoles fields.

# To  retrieve the datasets of an input role named 'input_A' as an array of dataset names:
input_name = get_input_names_for_role('main_input')[0].split('.')[1]
print("################################################### {}".format(input_name))
print(get_recipe_config())

# For outputs, the process is the same:
output_name = get_output_names_for_role('main_output')[0].split('.')[1]
output_dataset = dataiku.Dataset(output_name)


# The configuration consists of the parameters set up by the user in the recipe Settings tab.

# Parameters must be added to the recipe.json file so that DSS can prompt the user for values in
# the Settings tab of the recipe. The field "params" holds a list of all the params for wich the
# user will be prompted for values.

# The configuration is simply a map of parameters, and retrieving the value of one of them is simply:
# For optional parameters, you should provide a default value in case the parameter is not present:
config = get_recipe_config()

target_variable   = config['target_variable']

n_start_variables          = int(config.get('n_start_variables', 10))
n_step_variables           = int(config.get('n_step_variables' , 1))
early_stop_rounds          = int(config.get('early_stop_rounds', 10))
early_stop_metric          = config.get('early_stop_metric', 'gini')
early_stop_min_improvement = config.get('early_stop_min_improvement', 0.001)


max_boost_rounds  = int(config.get('max_boost_rounds' , 700))
n_folds           = int(config.get('n_folds'          , 5))
seed              = int(config.get('seed', 0))
num_leaves        = int(config.get('num_leaves', 31))
max_depth         = int(config.get('max_depth', -1))
n_estimators      = int(config.get('n_estimators', 100))
subsample_for_bin = int(config.get('subsample_for_bin', 200000))
n_jobs            = int(config.get('n_jobs', 4))
learning_rate     = config.get('learning_rate', 0.1)
feature_fraction  = config.get('feature_fraction', 1)
bagging_fraction  = config.get('bagging_fraction', 1)


if config.get("use_text_base_variables_toogle", False):
    tmp = config.get('base_variables_text', "")
    if len(tmp):
        base_variables = [x.strip() for x in tmp.split(',')]
    else:
        base_variables = None
else:
    base_variables = config.get('base_variables', None)

if config.get("use_text_skip_variables_toogle", False):
    tmp = config.get('skip_variables_text', "")
    if len(tmp):
        skip_variables = [x.strip() for x in tmp.split(',')]
    else:
        skip_variables = None
else:
    skip_variables = config.get('skip_variables', None)


model_params = get_recipe_config().get('model_params', {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_threads': 1 if n_jobs < 0 else n_jobs,
    'feature_fraction': feature_fraction,
    'seed': seed,
    'n_estimators': n_estimators,
    'num_leaves': num_leaves,
    'max_depth': max_depth,
    'learning_rate': learning_rate,
    'subsample_for_bin': subsample_for_bin,
    'max_depth': max_depth,
    'bagging_fraction': bagging_fraction
})

# Note about typing:
# The configuration of the recipe is passed through a JSON object
# As such, INT parameters of the recipe are received in the get_recipe_config() dict as a Python float.
# If you absolutely require a Python int, use int(get_recipe_config()["my_int_param"])


#############################
# Your original recipe
#############################

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
df_result = run(
    input_name,
    # Variable parameters
    target_variable=target_variable, skip_variables=skip_variables, base_variables=base_variables,
    # Model parameters
    max_boost_rounds=max_boost_rounds, n_folds=n_folds, model_params=model_params,
    # Feature selection parameters
    n_start_variables=n_start_variables, n_step_variables=n_step_variables, early_stop_rounds=early_stop_rounds)

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
# Recipe outputs
output_dataset.write_with_schema(df_result)
