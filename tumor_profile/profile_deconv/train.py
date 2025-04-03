import pandas as pd
import pickle
import lightgbm as lgb
import time

DEFAULT_BOOSTING_PARAMETERS = {'subsample': 0.9607,
                               'subsample_freq': 9,
                               'colsample_bytree': 0.2933,
                               'reg_alpha': 3.9006,
                               'reg_lambda': 2.938,
                               'learning_rate': 0.05,
                               'max_depth': 11,
                               'min_child_samples': 271,
                               'num_leaves': 9419,
                               'n_estimators': 3000,
                               'n_jobs': 4}

def train_tumor_profile_models(mix_expr, mix_values, train, gene, features=None, boosting_parameters=DEFAULT_BOOSTING_PARAMETERS, 
                save_path=None, path_to_data = '/uftp/COMMON_NOTEBOOKS/Deconvolution/Tumor_profile_reconstruction_models_training/',
                random_state=0):
    """
    Function makes lgb models for cell types from cells_to_mix with
    
    :param mix_expr: dataframe with expressions in TPM with genes as indices
    :param mix_values: dataframe with cell fractions with cell types as indices
    :param train: dataframe with expressions of malignant cells only
    :param gene: str, target gene to predict
    :param features: list, features of the model
    :param boosting_parameters: dict, parameters for boosting for each cell type
    :param random_state: int, random state
    
    :returns: lgbm models
    """

    if features is None:
        features = []
        with open(path_to_data+'ness_genes.txt') as text:
            for i in text:
                features.append(i.strip()[1:-2])
    if gene not in features:
        features += [gene]
        
    start_time = time.time()
    model = lgb.LGBMRegressor(**boosting_parameters, random_state=random_state)
    expr = pd.concat([mix_expr.loc[features].T,mix_values.T], axis = 1)
    matrix_M = pd.read_csv(path_to_data+'Valyas_Matrix_with_monocytes.tsv', sep='\t', index_col=0)
    
    matrix_M['T_cells'] = matrix_M['CD4_T_cells'] + matrix_M['CD8_T_cells']
    common_cell_types = matrix_M.columns.intersection(mix_values.T.columns)
    subtraction_train = mix_values.T.loc[:,common_cell_types].dot(matrix_M.loc[gene, common_cell_types]).clip(lower=0)
    expr.loc[:,'MATRIX_M']=subtraction_train
    model.fit(expr.loc[:, sorted(set(expr.columns))], train.loc[gene])  
    print("--- %s seconds ---" % (time.time() - start_time))
    if save_path is not None: 
        if save_path[-1] != '/':
            save_path+= '/'
        filename = save_path+gene+'.sav'
    else: 
        filename = gene+'.sav'
    pickle.dump(model, open(filename, 'wb'))
    return model
