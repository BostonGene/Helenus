import warnings
from os.path import join
from pathlib import Path
from typing import Union
import numpy as np
import pandas as pd
from loguru import logger


def check_expressions(
    expr: pd.DataFrame,
    dec_features,
    path_to_data: str = "/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/",
) -> None:
    """Checks if expressions have the right format.

    Args:
        expr (pd.DataFrame): Expression data.
        path_to_data (str, optional): Path to directory with features for a current model.
        Defaults to "/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/".

    Raises:
        ValueError: Expression values should not be log normalized.
        ValueError: Expression data should contain all genes which are used as a features for the model.
        ValueError: Expression data should contain all genes which are used as a features for the model.
    """
    ness_genes = []
    with open(join(path_to_data, "ness_genes.txt")) as text:
        for line in text:
            ness_genes.append(line.strip()[1:-2])
    if not any(expr.max(axis=1) > np.log2(10**6)):
        raise ValueError(
            "Current model does not work with log normalized data. Linearize your expression matrix."
        )
    diff = set(ness_genes).difference(set(expr.index))
    if diff:
        raise ValueError(
            "Expression matrix should contain at least all genes that are used as a features."
        )

    diff = set(list(dec_features) + ness_genes).difference(set(expr.index))
    if diff:
        raise ValueError(
            "Expression matrix should contain at least all genes that are used as a features."
        )

    diff = set(ness_genes).symmetric_difference(set(expr.index))
    if not diff:
        warnings.warn(
            "You are using only feature genes. Make sure that normalization is correct.",
            UserWarning,
        )
    else:
        logger.info("Expressions: OK")

def renormalize_expr(
    expr: pd.DataFrame,
    path_to_data: Union[
        str, Path
    ] = "/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/genes_for_TPM_normalization_tcgav2.tsv",
) -> pd.DataFrame:
    """Renormalize expression matrix to TMP.

    Args:
        expr (pd.DataFrame): Expression table.
        path_to_data (Union[ str, Path ], optional): Path to genes from TPM normalization.
        Defaults to "/uftp/COMMON_NOTEBOOKS/Deconvolution/predict_tumor_expression_profile/genes_for_TPM_normalization_tcgav2.tsv".

    Returns:
        pd.DataFrame: TMP normalized expression table.
    """
    TPM_norm = list(pd.read_csv(path_to_data, sep="\t", index_col=0).iloc[:, 0])
    sym_diff = set(TPM_norm).symmetric_difference(set(expr.index))
    if len(sym_diff) > 0:
        expr = expr.loc[TPM_norm]
        expr = (expr / expr.sum()) * 10**6

    return expr
