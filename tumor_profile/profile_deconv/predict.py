from typing import Union, List, Optional
from os.path import isfile, join, exists
from pathlib import Path
from os import listdir
import pandas as pd
import numpy as np
import lightgbm
import logging
import pickle
import yaml
from tumor_profile.profile_deconv.utils import check_expressions, renormalize_expr
from tumor_profile.profile_deconv.old_deconvcall import TumorModel
from mldeconv.models.deconv_model import DeconvModel
from mldeconv.prediction import choose_predict_params_for_diagnosis
from mldeconv.prediction import predict as predict_mldeconv

PATH_TO_AUX_DATA = "/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/"
GENES_IN_EXPRESSION = "/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/models_without_tissues_14_06/genes_v2.tsv"
PATH_TO_M = '/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/product_models/average_TME_profile_okt22.tsv'
PATH_TO_DATA = "/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/"
DEFAULT_CT_FOR_M = ['Monocytes','CD4_T_cells', 'CD8_T_cells', 'B_cells', 'Neutrophils', 'Macrophages', 'Endothelium', 'Fibroblasts', 'NK_cells']
PATH_TO_DECONV_MODEL = '/uftp/Deconvolution/product/models/configs/product/model_v1_all.yaml'
PATH_TO_HELENUS_MODELS = {'SOLID_TUMOR': '/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/models/passed_criteria_article_version/',
              'TCL': '/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/Lymphoma_models/TCL/version1/',
                'BCL': '/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/Lymphoma_models/BCL/version1/'}
FEATURES_PATH = {'SOLID_TUMOR': '/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/models/newdec_nonoise_numav1/features.tsv',
                'TCL': '/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/Lymphoma_models/TCL/version1/',
                'BCL': '/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/Lymphoma_models/BCL/version1/'}
PATH_TO_M_ARTICLE_VERSION = '/uftp/COMMON_NOTEBOOKS/Deconvolution/Tumor_profile_reconstruction_models_training/Valyas_Matrix_with_monocytes.tsv'

def pattern_model_application(expr, path_to_lymphoma_models, mode):
    """Load T or B -cells model_PATTERN model.
    Args:
        expr (pd.DataFrame): initial expressions. Columns - samples, index - genes
        path_to_lymphoma_models (str): Path to directory with model.
        mode (str): (TCL) T-cells pattern model or (BCL) B-cells pattern model.

    Returns:
        pd.Series, fractions of T or B -cells pattern.
    """
    
    path_to_features = f'{path_to_lymphoma_models}''features_'+mode+'.txt'
    with open(path_to_features) as f:
        features = f.read().split('\n')
  
    PATTERN_model = pickle.load(open(f'{path_to_lymphoma_models}''/'+mode[0]+'_CELLS_PATTERN_models.sav', "rb"))
    preds = PATTERN_model.predict(expr.loc[features[:-1]].T)
    
    preds = pd.Series(preds.clip(min=0, max=1), index = expr.columns)
    
    return preds

def run_lymphoma_model(loaded_model, expr, not_passed_thr, gene, features_path):
    """_summary_

    Args:
        loaded_model (LightGBM): Loaded lymphoma model.
        expr (pd.DataFrame): initial expressions. Columns - samples, index - genes
        not_passed_thr (list): list of samples that have purity lower than threshold
        gene (str): Gene name for which we want to run model.
        features_path (str): Path to model features.
        
    Returns:
        dict: tme or tumor profile for one gene.
    """
    prediction = {'TME': None, 'Tumor': None}
    selected_features = pd.read_csv(features_path+'features_for_each_model.tsv', sep="\t", index_col=0)
    features_genes = sorted(list(selected_features[gene].dropna()))
    gene_preds = loaded_model.predict(expr.loc[features_genes].T)
    prediction['TME'] = pd.Series(gene_preds.clip(min=0, max=1000000), index = expr.columns)
    subtracted = expr.loc[gene] - gene_preds.clip(min=0, max=1000000)
    prediction['Tumor'] =  pd.Series(subtracted, index = expr.columns).clip(lower=0)
    
    prediction['TME'].loc[not_passed_thr] = np.nan
    prediction['Tumor'].loc[not_passed_thr] = np.nan
    return prediction

def DECONV_download(version):
    """Load DECONV model.
    Args:
        version (str): version of the model to load. Versions: '0.1.69' or 'old'
    Returns:
        Loaded mldeconv model.
    """
    if version=='0.1.69':
        config_path = '/uftp/Deconvolution/product/models/configs/product/model_v1_all.yaml'
        with open(config_path) as c:
            config = yaml.load(c, Loader=yaml.FullLoader)
        model_deconv = DeconvModel.load(config['model_path'])
        features = model_deconv.genes_features
        
    elif version=='old':
        path_to_olddecmodel = "/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/"
        model_deconv = TumorModel.load(join(path_to_olddecmodel, "deconv_models"))
        features = model_deconv.cell_types.genes
    
    return model_deconv, features


def model_loading(path: str, gene: str)->lightgbm.LGBMRegressor:
    """Load Tumor model.
    Args:
        path (str): Path to directory with model.
        gene (str): Gene name for which we want to load model.

    Returns:
        Loaded tumor profile model: lightgbm.LGBMRegressor.
    """
    if path[-1] != "/":
        path += "/"
    filename = f"{gene}.sav"
    loaded_model = pickle.load(open(join(path, filename), "rb"))

    return loaded_model


def run_model(
    gene: str,
    before_subtraction: pd.DataFrame,
    deconv_prediction: pd.DataFrame,
    subtraction_with_M: pd.DataFrame,
    loaded_model: TumorModel,
    tumfrac: pd.Series,
    features_path: str,
    threshold_frac: float = 0.2,
) -> pd.DataFrame:
    """_summary_

    Args:
        gene (str): Gene name for which we want to run model.
        before_subtraction (pd.DataFrame): Before substraction.
        deconv_prediction (pd.DataFrame): Predictions of deconvolution model.
        subtraction_with_M (pd.DataFrame): Subtraction with M matrix.
        loaded_model (TumorModel): Loaded Tumor model.
        path (str): Path to cell types.
        tumfrac (pd.Series): Tumor fraction.
        features_path (str): Path to model features.
        mode (str): tme or tumor to reconstruct.
        threshold_frac (float, optional): Threshold fraction. Defaults to 0.2.

    Raises:
        ValueError: Mode may be only tme or tumor. Value error will be raised if other model will be given.

    Returns:
        pd.DataFrame: tme or tumor profile.
    """
    selected_features = pd.read_csv(features_path, sep="\t", index_col=0)
    features_genes = list(selected_features[gene].dropna())
    features = pd.concat(
        [
            before_subtraction.loc[list(set(features_genes + [gene]))],
            deconv_prediction
        ],
        axis=0,
    )
    features.loc["MATRIX_M"] = subtraction_with_M.loc[gene]
    features = features.loc[sorted(features.index)]
    tme_pred = pd.Series(loaded_model.predict(features.T))
    less_than_sens = tumfrac[tumfrac < threshold_frac]
    tme_pred.index = tumfrac.index
    tme_pred.loc[less_than_sens.index] = np.nan
    tme_pred = np.clip(tme_pred, 0, 10**6) # TPM clipping

    tumor_pred = (before_subtraction.loc[gene] - tme_pred)
    tumor_pred = np.clip(tumor_pred, 0, 10**6)

    return {'Tumor': tumor_pred,
            'TME': tme_pred}

def predict_tumor_fraction(
    before_subtraction: pd.DataFrame,
    path: str = "/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/",
) -> pd.Series:
    """Predicts tumor fraction.
    Args:
        before_subtraction (pd.DataFrame): Before subtraction.
        path (str, optional): Path to directory with model and its features.
    Returns:
        pd.Series: Predicted tumor fraction.
    """
    other_model = pickle.load(open(join(path, "other_model.p"), "rb"))
    other_model_features = []
    with open(join(path, "genes_for_multiclass_v2.txt"), "r") as filehandle:
        for line in filehandle:
            currentPlace = line[:-1]
            other_model_features.append(currentPlace)
    tumc = pd.Series(
        1 - np.clip(other_model.predict(before_subtraction.loc[other_model_features].T), 0, 1),
        index=before_subtraction.columns,
    )

    return tumc

def get_tme_expression_with_m(
    cell_rna_fractions: pd.DataFrame,
    matrix_m: pd.DataFrame,
    cell_types: list = DEFAULT_CT_FOR_M,
    ) -> pd.DataFrame:
    """Calculate M matrix.
    Args:
        cell_rna_fractions (pd.DataFrame): Cell RNA fractions.
        matrix_m (pd.DataFrame): M matrix.
        cell_types (list, optional): Cell types. Defaults to DEFAULT_CT_FOR_M.
    Raises:
        Exception: Cell fractions > 1.
        Exception: Exclude cells names do not match.
        Exception: Exclude cells profiles names do not match.
    Returns:
        pd.DataFrame: M matrix.
    """

    return cell_rna_fractions[cell_types].dot(matrix_m[cell_types].T)

def matrix_m_reconstruction(
    expr: pd.DataFrame,
    cell_types: Union[List[str], None] = DEFAULT_CT_FOR_M,
    path_to_deconv_model: Union[str, Path] = PATH_TO_DECONV_MODEL,
    path_to_M: Union[str, Path] = PATH_TO_M,
    genes_in_expression: Union[str, Path] = GENES_IN_EXPRESSION,
) -> pd.DataFrame:
    """Reconstruction of microenvironment and tumor expression profile.
    Args:
        expr (pd.DataFrame): DataFrame, expressions of the samples.
        genes (List[str]): List of genes, that should be reconstructed.
        mode (str): tme or tumor to reconstruct.
        genes_in_expression (Union[ str, Path ], optional): Path to directory with a list of genes for TPM normalization.
    Raises:
        ValueError: Mode may be only tme or tumor. Value error will be raised if other model will be given.

    Returns:
        pd.DataFrame: Reconstructed profile of tumor or tme. Depends on mode.
    """
    # loading 

   
    matrix_m = pd.read_csv(path_to_M, sep="\t", index_col=0)

    # checking
    missing_cts = sorted(list(set(cell_types).difference(set(matrix_m.columns))))
    if missing_cts:
        raise Exception(f'Missing cell types in average profiles refference {missing_cts}')
    check_expressions(expr, dec_features=[])
    
    # reconstruction
    expr = renormalize_expr(expr, genes_in_expression)
    tumc_fraction = predict_tumor_fraction(expr)
    if 'Hepatocytes' in  cell_types:
        file = '/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/predict_tissue_with_epithelium_Hepatocytes_M.yaml'
        with open(file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        config["tumor_terms"] = None
        deconv_prediction = predict_mldeconv(expr,config,fraction_type="rna")
        
    else: 
        diagnosis = "All_celltypes"
        deconv_prediction = predict_mldeconv(expr,
                choose_predict_params_for_diagnosis(diagnosis),
                fraction_type="rna")
    subtraction_with_M = get_tme_expression_with_m(deconv_prediction.T.clip(lower=0), matrix_m, cell_types)
    reconstructed_profile = expr.T - subtraction_with_M
    reconstructed_profile = reconstructed_profile.clip(lower=0)
    tumor_reconstructed = reconstructed_profile.T.divide(tumc_fraction)
    tme_reconstructed = subtraction_with_M.T.divide(1 - tumc_fraction)
    
    return {'Tumor': tumor_reconstructed,
            'TME': tme_reconstructed}

def warning_by_genes(path_to_model, genes):
    """processing of input genelist.
    Args:
        path_to_model (str): Path to directory with model and its features.
        genes (list): list of genes.
    Returns:
        list: list of available genes-models.
    """
    model_list = [f[:-4] for f in listdir(path_to_model) if isfile(join(path_to_model , f)) and '.sav' in f and 'PATTERN' not in f]
    if genes is None:
        genes = model_list  
    else:
        missing_genes = set.difference(set(genes), set(model_list))
        if len(missing_genes) > 0:
            logging.warning(f"Following genes are not present in the model and will be skipped: {', '.join(missing_genes)}.")
        genes = set.intersection(set(genes), set(model_list))
    return genes

def predict(
    expr: pd.DataFrame,
    genes: List[str]=None,
    mode: str = 'SOLID_TUMOR',
    path_to_model: Union[str, Path] = PATH_TO_HELENUS_MODELS,
    path_to_data: Union[str, Path] = PATH_TO_AUX_DATA,
    genes_in_expression: Union[str, Path] = GENES_IN_EXPRESSION,
    features_path: Optional[Union[str, Path]] = None,
    path_to_M: Union[str, Path] = PATH_TO_M_ARTICLE_VERSION,
    threshold_frac: float = 0.2,
    deconv_version: str = '0.1.69' 
) -> pd.DataFrame:
    """Reconstruction of microenvironment and tumor expression profile.

    Args:
        expr (pd.DataFrame): DataFrame, expressions of the samples.
        genes (List[str]): List of genes, that should be reconstructed.
        path_to_deconv_model (Union[ str, Path ], optional): Path to directory with models.
        path_to_data (Union[ str, Path ], optional): Path to directory with a list of genes, that should be reconstructed.
        genes_in_expression (Union[ str, Path ], optional): Path to directory with a list of genes for TPM normalization.
        features_path (Union[ str, Path ], optional): Path to features.
        threshold_frac (float, optional): <1. Fraction of tumor below which the model predicts NAN.
        deconv_version (str): version of mldeconv (for features). 
    Raises:
        ValueError: Mode may be only tme or tumor. Value error will be raised if other model will be given.

    Returns:
        dict: pd.DataFrame's with reconstructed profile of Tumor and TME fractions.
    """
   
    if path_to_model[mode][-1] != "/":
        path_to_model[mode] += "/"

    if features_path is None:
        if exists(p := join(path_to_model[mode], 'features.tsv')):
            features_path = p
        else:
            logging.error('Path to features was not passed and features.tsv file does not exist in path_to_model directory.')
            raise FileNotFoundError(f'File with features {p} not found')
            
    genes = warning_by_genes(path_to_model[mode], genes)
    expr = renormalize_expr(expr, path_to_data=genes_in_expression)
   
    if mode == 'SOLID_TUMOR':
        if 'product' in path_to_model[mode]:
            deconv_version='old'
        matrix_m = pd.read_csv(path_to_M, sep="\t", index_col=0)
        tumc_fraction = predict_tumor_fraction(expr)
        model_deconv, dec_features = DECONV_download(version=deconv_version)
        check_expressions(expr, dec_features)
        deconv_prediction = model_deconv.predict_l1(expr)
        common_ct = deconv_prediction.columns.intersection(matrix_m.columns)
        subtraction_with_M = get_tme_expression_with_m(deconv_prediction[common_ct], matrix_m=matrix_m[common_ct])
        run_model_HELENUS = run_model
        input_to_run = {'before_subtraction': expr,
                        'deconv_prediction': deconv_prediction.T,
                        'subtraction_with_M': subtraction_with_M.T,
                        'features_path': features_path,
                        'tumfrac': tumc_fraction, 
                        'threshold_frac': threshold_frac}
       
           
    elif mode == 'TCL' or mode == 'BCL':
        tumc_fraction = pattern_model_application(expr, features_path, mode=mode)
        not_passed_thr = tumc_fraction[tumc_fraction<threshold_frac].index
        run_model_HELENUS = run_lymphoma_model
        input_to_run = {'expr': expr, 'not_passed_thr': not_passed_thr, 'features_path': features_path}
        

    if len(tumc_fraction[tumc_fraction<threshold_frac])>0:
        text = ''
        for i in tumc_fraction[tumc_fraction<threshold_frac].index:
            text+= str(i)+', '
        logging.warning(' Samples(s): '+text[:-1]+ ' tumor purity is lower than threshold '+ str(threshold_frac))
    
    tumor_profile = pd.DataFrame(np.nan, index = genes, columns = expr.columns)
    tme_profile = pd.DataFrame(np.nan, index = genes, columns = expr.columns)
    for gene in genes:

        loaded_model = model_loading(path_to_model[mode], gene)
        gene_preds = run_model_HELENUS(gene=gene, loaded_model=loaded_model,
                                       **input_to_run)
        
        tme_profile.loc[gene] = gene_preds['TME'].values
        tumor_profile.loc[gene] = gene_preds['Tumor'].values

    
    return {"Tumor":tumor_profile.divide(tumc_fraction),
            "TME": tme_profile.divide(1 - tumc_fraction)}
