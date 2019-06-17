# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE_MAGIC_CELL
# Automatically replaced inline charts by "no-op" charts
# %pylab inline
import matplotlib
matplotlib.use("Agg")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
####
# Libs
####
import dataiku
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import shap

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CODE
####
# Functions
####
VERBOSE = True

def density_plot(y_true, y_pred, figure=None, curve_name=''):
    if figure is None:
        plt.figure()
    else:
        plt.figure(figure)

    for v in np.unique(y_true):
        _array = y_pred[y_true == v]
        _weights = np.ones_like(_array)/float(len(_array))
        plt.hist(_array, bins=50, label='Class {} {}'.format(v, curve_name), weights=_weights, alpha=0.7)

    plt.xlabel('Predicted probability')
    plt.ylabel('Percentage of observations')
    plt.legend()
    plt.show()


def load_data(tbl_name, target_variable, base_variables=None, skip_variables=None):
    vprint('#### Load Data', include_date=True)
    dku_dataset = dataiku.Dataset(tbl_name)
           
    skip_variables = skip_variables or list()
    base_variables = base_variables or list()
    all_variables = dku_dataset.get_config()['schema']['columns']
        
    keep_variables = [target_variable] + base_variables
    keep_variables += [x['name'] for x in all_variables if x['name'] not in (skip_variables + keep_variables) and x['type'] != 'date']    
    
    if target_variable not in keep_variables:
        raise ValueError('Target variable is not in dataframe')

    # Read dataset
    df = dku_dataset.get_dataframe(columns=keep_variables, limit=None).select_dtypes(exclude=['datetime'])
    
    # Remove_rows_with_null_target
    #df = df[~df[target_variable].isnull()]

    # Raise an error if target variables has rows where is not defined
    if df[target_variable].isnull().sum():
        raise ValueError('Target variable has null values')

    if len(df[target_variable].unique()) > 2:
        raise ValueError('Target variable has  more than 2 modalities. This recipe works only for binary classification tasks')
    
    vprint('Input dataset shape: {}'.format(df.shape))

    # Separate Y from X and misc data
    df_label = df[target_variable]
    df = df.drop(target_variable, axis=1)

    # Detect categorical variables
    categorical_variables = df.select_dtypes(include=[object, 'category']).columns.unique().tolist()
    vprint('There are {} categorical variables'.format(len(categorical_variables)))

    # Label encoding
    for column in categorical_variables:
        df[column] = df[column].astype('category')

    return df, df_label


def compute_model_and_importance_order(X, label, n_folds=5, max_boost_rounds=700, show_plots=False, **kwargs):
    vprint('#### Compute Model & Importance Order', include_date=True)
    cv_res = lgb.cv(kwargs, lgb.Dataset(X, label=label), num_boost_round=max_boost_rounds,
                    nfold=n_folds, early_stopping_rounds=40, verbose_eval=50)
    if show_plots: pd.DataFrame(cv_res)['auc-mean'].plot(style='*-', title="AUC", grid=True);

    # Train full model
    clf = lgb.LGBMClassifier(**kwargs)
    clf.fit(X, label);
    if show_plots: density_plot(label.values, clf.predict_proba(X)[:, 1])

    # Explanation
    explainer = shap.TreeExplainer(clf)
    shap_values = pd.DataFrame(explainer.shap_values(X), columns=X.columns)
    shap_importance = shap_values.abs().mean().sort_values()
    if show_plots: shap_importance.tail(15).plot(kind='barh', grid=True);

    important_variable_order = shap_importance.sort_values(ascending=False).index

    return clf, important_variable_order


def compute_stepwise_cv_score(X, label, important_variable_order, n_folds=5, n_start_variables=10,
                              n_step_variables=1, max_boost_rounds=700, early_stop_rounds=10, early_stop_min_improvement=0.001, early_stop_metric='gini', show_plots=False, **kwargs):
    vprint('#### Compute Stepwise CV Score', include_date=True)
    # Iterate over variables
    previous_metric = 1
    skip_count = 0
    n_offset_variables = 0

    important_variable_order = np.array(important_variable_order)
    
    variables_previous = []
    result = {'gini': [], 'auc': [], 'variables_all': [], 'variables_new': [], 'n_variables': []}
    i = 0
    while i < (len(important_variable_order) - n_start_variables):
        _idx = min((n_start_variables + i * (n_step_variables + n_offset_variables), len(important_variable_order)))
        vprint('Iteration {}:\n  Using first {} variables (from {} until {})'.format(i, _idx, important_variable_order[0], important_variable_order[_idx-1]))
        
        _current_variables = important_variable_order[:_idx]
        _df = X.loc[:, _current_variables]

        new_variables = list(set(_current_variables).difference(variables_previous))
        
        _cv_res = lgb.cv(kwargs, lgb.Dataset(_df, label=label),
                         num_boost_round=max_boost_rounds, nfold=n_folds,
                         verbose_eval=50, early_stopping_rounds=40)

        _auc = max(_cv_res['auc-mean'])
        vprint('  AUC: {}'.format(_auc))

        # Store results
        result['gini'].append(_auc * 2 - 1)
        result['auc'].append(_auc)
        result['variables_all'].append(_current_variables.tolist())
        result['n_variables'].append(_idx)
        result['variables_new'].append(new_variables)

        _improvement = abs(result[early_stop_metric][i] - previous_metric)

        # Check for early stop
        if _improvement <= early_stop_min_improvement:
            skip_count += 1

            if skip_count >= early_stop_rounds:
                vprint('  Early stop!')
                break
            else:
                n_offset_variables += 0
        else:
            skip_count = 0
            
        previous_metric = result[early_stop_metric][i]
        variables_previous = _current_variables
        print("Skip count = {}".format(skip_count))
        
        i += 1
        
    plt.figure()
    if show_plots: pd.DataFrame(result).set_index('n_variables')['gini'].plot(style='*-', title="GINI", grid=True);
    return result


def vprint(msg, include_date=False):
    if VERBOSE:
        print('[{}] {}'.format(str(pd.Timestamp('now')), msg) if include_date else '  {}'.format(msg))


def run(tbl_name,
        # Variable parameters
        target_variable, skip_variables=None, base_variables=None,
        # Model parameters
        max_boost_rounds=700, n_folds=5, model_params=None,
        # Feature selection parameters
        n_start_variables=10, n_step_variables=1, 
        # Early stop
        early_stop_rounds=10, early_stop_metric='gini', early_stop_min_improvement=0.001):
    ####
    # Read data
    ####
    df, df_label = load_data(tbl_name, target_variable, base_variables, skip_variables)

    print("#############################################")
    print(df.columns.tolist())
    print("#############################################")
    ####
    # Model
    ####
    model_params = model_params or {'boosting_type': 'gbdt', 'objective': 'binary', 'metric': 'auc', 'feature_fraction': 1, 'seed': 0, 'n_estimators': 100}
    model_params['is_unbalance'] = (df_label.value_counts()/len(df_label)).min() < 0.4

    ####
    # Build a full model to get SHAP importance ==> Variable importance orde
    ####
    clf, important_variable_order = compute_model_and_importance_order(df, df_label,
                                                                       n_folds=n_folds,
                                                                       max_boost_rounds=max_boost_rounds,
                                                                       **model_params)
    
    # Ensure that the base variables are at the top of the list
    if base_variables is not None:
        n_start_variables += len(base_variables)
        important_variable_order = base_variables + [x for x in important_variable_order if x not in base_variables]
        
    ####
    # Stepwise variable scoring
    ####
    result = compute_stepwise_cv_score(df, df_label, important_variable_order,
                                       n_folds=n_folds,
                                       n_start_variables=n_start_variables,
                                       n_step_variables=n_step_variables,
                                       max_boost_rounds=max_boost_rounds,
                                       early_stop_rounds=early_stop_rounds,
                                       early_stop_min_improvement=early_stop_min_improvement,
                                       early_stop_metric=early_stop_metric,
                                       **model_params)
    ####
    # Build result dataset
    ####
    df_result = pd.DataFrame(result)
    df_result['model_type'] = str(clf.__class__).replace("<class '", "").replace("'>", "")
    df_result['model_params'] = [model_params] * len(df_result)

    return df_result