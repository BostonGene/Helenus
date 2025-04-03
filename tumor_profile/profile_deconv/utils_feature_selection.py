# Data manipulation functions
from functools import partial
import numpy as np
import optuna
import warnings
import pandas as pd
from typing import Callable, Mapping, Union, Iterator, Tuple

from optuna import distributions
from sklearn.model_selection import BaseCrossValidator, cross_val_score
from sklearn.base import BaseEstimator


def get_samples(sr: pd.Series, cts: list) -> list:
    samples = []
    for ct in cts:
        sr_temp = sr.dropna().apply(lambda x: x.split(";"))
        condition = sr_temp.apply(
            lambda x: any(item in ct for item in x) if len(x) > 1 else x[0] == ct
        )
        samples += list(sr_temp[condition].index.values)
    samples = sorted(list(set(samples)))
    return sr[samples]


def get_cell_with_subtypes(config, cell):
    types_structure = config["Types_structure"].copy()
    cells = types_structure[cell].copy() + [cell]
    unchecked_cells = types_structure[cell].copy()
    c = 0
    while unchecked_cells and (c < 1000):
        unchecked_cell = unchecked_cells.pop()
        if unchecked_cell in types_structure.keys():
            cells += types_structure[unchecked_cell]
            unchecked_cells += types_structure[unchecked_cell]
        c += 1
    return sorted(list(set(cells)))


def train_test_split_by_datasets(config, annotation, cells, p):
    train_samples = []
    valid_samples = []

    for cell in cells:
        cell_with_subtypes = get_cell_with_subtypes(config, cell)
        samples = get_samples(annotation["Cell_type"], cell_with_subtypes).index
        datasets = annotation.loc[samples, "Dataset"].unique()

        if len(datasets) < 2:
            print("{} presented in less than 2 datasets! Skipped.".format(cell))
            continue

        n_valid = int(
            max(np.round(p * len(datasets)), 1)
        )  # if len == 2, np.round(0.1 * 2) == 0 => max(0, 1) == 1

        valid = np.random.choice(datasets, n_valid, replace=False)
        train = list(set(datasets) - set(valid))

        train_samples.extend(
            annotation[annotation["Dataset"].isin(train)].index.intersection(samples)
        )
        valid_samples.extend(
            annotation[annotation["Dataset"].isin(valid)].index.intersection(samples)
        )
    return annotation.loc[train_samples], annotation.loc[valid_samples]


class ByDatasetsSplitter(BaseCrossValidator):
    """ByDatasetsSplitter cross-validator

    Provides train/validation indices to split data in train/validation sets.
    """

    def __init__(self, train_size: int, valid_size: int, n_splits: int = 5) -> None:
        """
        Args:
            train_size (int): The number of train points.
            valid_size (int): The number of validation points.
            n_splits (int, optional): The number of fold in cross-validation. Defaults to 5.
        """

        self.n_splits = n_splits
        self.train_size = train_size
        self.valid_size = valid_size
        self.sample_size = train_size + valid_size

    def split(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.DataFrame, np.ndarray, None] = None,
        groups: Union[pd.DataFrame, np.ndarray, None] = None,
    ) -> Iterator[
        Tuple[np.ndarray],
    ]:
        """Provides train/validation indices to split data in train/validation sets.

        Args:
            X (Union[pd.DataFrame, np.ndarray]): array-like of shape `(n_samples, n_features)`
            Training data, where n_samples is the number of samples and n_features is the number of features.
            y (Union[pd.DataFrame, np.ndarray, None], optional): array-like of shape (n_samples,)
            The target variable for supervised learning problems. Defaults to None.
            groups (Union[pd.DataFrame, np.ndarray, None], optional): Always ignored, exists for compatibility. Defaults to None.

        Yields:
            Iterator[Tuple[np.ndarray]]: `train_ids: np.ndarray`. The training set indices for that split. 
            `valid_ids: np.ndarray`. The validating set indices for that split.
        """

        for i in range(self.n_splits):
            start_train = i * self.sample_size
            train_ids = np.arange(start_train, start_train + self.train_size)
            start_valid = start_train + self.train_size
            valid_ids = np.arange(start_valid, start_valid + self.valid_size)

            yield train_ids, valid_ids

    def get_n_splits(
        self, X: object = None, y: object = None, groups: object = None
    ) -> int:
        """Returns the number of splitting iterations in the cross-validator

        Args:
            X (object, optional): Always ignored, exists for compatibility. Defaults to None.
            y (object, optional): Always ignored, exists for compatibility. Defaults to None.
            groups (object, optional): Always ignored, exists for compatibility. Defaults to None.

        Returns:
            int: Returns the number of splitting iterations in the cross-validator.
        """

        return self.n_splits


def optuna_search_experimental(
    all_expression_data: Union[pd.DataFrame, np.ndarray],
    all_cell_proportions: Union[pd.DataFrame, np.ndarray],
    model: BaseEstimator,
    param_distributions: Mapping[str, distributions.BaseDistribution],
    cv: BaseCrossValidator,
    scoring: Union[str, Callable] = None,
    n_trials: int = 10,
    n_jobs: int = -1,
) -> BaseEstimator:
    """Hyperparameter search with cross-validation.

    Args:
        all_expression_data (Union[pd.DataFrame, np.ndarray]): array-like of shape `(n_samples, n_features)`
        Training data, where n_samples is the number of samples and n_features is the number of features.

        all_cell_proportions (Union[pd.DataFrame, np.ndarray]): array-like of shape (n_samples,)
        The target variable for supervised learning problems.

        model (BaseEstimator): Object to use to fit the data. This is assumed to implement the scikit-learn estimator interface. 
        Either this needs to provide score, or scoring must be passed.

        param_distributions (Mapping[str, distributions.BaseDistribution]): Dictionary where keys are parameters and values are distributions. 
        Distributions are assumed to implement the optuna distribution interface.

        cv (BaseCrossValidator): Cross-validation strategy. Possible inputs for cv are:
                1. Integer to specify the number of folds in a CV splitter,
                2. A CV splitter,
                3. An iterable yielding (train, validation) splits as arrays of indices.

        scoring (Union[str, Callable]): String or callable to evaluate the predictions on the validation data. If None, score on the estimator is used.

        n_trials (int, optional): Number of trials. Defaults to 10.

        n_jobs (int, optional): Number of `threading` based parallel jobs. Defaults to None.

    Returns:
        BaseEstimator: Optuna search object with best score and parameters.
    """

    optuna_search = optuna.integration.OptunaSearchCV(
        model,
        param_distributions,
        cv=cv,
        n_trials=n_trials,
        n_jobs=n_jobs,
        scoring=scoring,
    )
    optuna_search.fit(all_expression_data, all_cell_proportions)

    return optuna_search


def optuna_search_classic(
    all_expression_data: Union[pd.DataFrame, np.ndarray],
    all_cell_proportions: Union[pd.DataFrame, np.ndarray],
    model: BaseEstimator,
    param_distributions: Mapping[str, distributions.BaseDistribution],
    cv: BaseCrossValidator,
    scoring: Union[str, Callable] = None,
    n_trials: int = 10,
    n_jobs: int = -1,
) -> BaseEstimator:
    """Hyperparameter search with cross-validation.

    Args:
        all_expression_data (Union[pd.DataFrame, np.ndarray]): array-like of shape `(n_samples, n_features)`
        Training data, where n_samples is the number of samples and n_features is the number of features.

        all_cell_proportions (Union[pd.DataFrame, np.ndarray]): array-like of shape (n_samples,)
        The target variable for supervised learning problems.

        model (BaseEstimator): Object to use to fit the data. This is assumed to implement the scikit-learn estimator interface. 
        Either this needs to provide score, or scoring must be passed.

        param_distributions (Mapping[str, distributions.BaseDistribution]): Dictionary where keys are parameters and values are distributions. 
        Distributions are assumed to implement the optuna distribution interface.

        cv (BaseCrossValidator): Cross-validation strategy. Possible inputs for cv are:
                1. Integer to specify the number of folds in a CV splitter,
                2. A CV splitter,
                3. An iterable yielding (train, validation) splits as arrays of indices.

        scoring (Union[str, Callable]): String or callable to evaluate the predictions on the validation data. If None, score on the estimator is used.

        n_trials (int, optional): Number of trials. Defaults to 10.

        n_jobs (int, optional): Number of `threading` based parallel jobs. Defaults to None.

    Returns:
        BaseEstimator: Optuna search object with best score and parameters.
    """

    def objective(trial):
        methods_mapper = {
            "UniformDistribution": trial.suggest_uniform,
            "LogUniformDistribution": trial.suggest_loguniform,
            "DiscreteUniformDistribution": trial.suggest_discrete_uniform,
            "IntUniformDistribution": trial.suggest_int,
            "IntLogUniformDistribution": partial(trial.suggest_int, log=True),
            "CategoricalDistribution": trial.suggest_categorical,
        }

        model_params = {}
        for param_name, param_dist in param_distributions.items():
            dist_name = param_dist.__class__.__name__
            if dist_name == "CategoricalDistribution":
                suggested_param = methods_mapper[dist_name](
                    name=param_name, choices=param_dist.choices
                )
            else:
                suggested_param = methods_mapper[dist_name](
                    name=param_name, low=param_dist.low, high=param_dist.high
                )

            model_params[param_name] = suggested_param

        model.set_params(**model_params)
        scores = cross_val_score(
            model, all_expression_data, all_cell_proportions, cv=cv, scoring=scoring
        )

        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    return study

def dataset_genes_selection(geneset, target_gene, tcgav2, n=600):
    """
    Return list of features for a target gene
    :param geneset: dict, key(s) - name of a geneset, value(s) - list of genes,
        that compose the geneset. 
    :param target_gene: str.
    :param tcgav2: list,  genes in expression (for normalization)
    :param n: int, amount of features to return
    """
    genesets_with_gene = {}  
    for key in geneset.keys():
        if target_gene in geneset[key]:
            genesets_with_gene[key] = geneset[key]
        
    gene_counts = []
    all_genes = sorted(tcgav2)
    for gene in all_genes:
        c = 0
        for key in genesets_with_gene.keys():
            if gene in genesets_with_gene[key]:
                c += 1 
        gene_counts += [c]

    df = pd.DataFrame({'GENE': all_genes, 'Count': gene_counts})

    features = list(df.loc[df['Count'] > 0].sort_values('Count', ascending=False)[:n]['GENE'].values)
    return features
