from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from pathlib import Path
import yaml
import typing as tp
import pandas as pd
import numpy as np
import gc
from numpy import random
import typing as tp
import pickle
import lightgbm as lgb
import timeit
from mldeconv.cells_utils import DEFAULT_DECONV_EXTERNAL_PATH
from mldeconv.cell_types import CellTypes, get_proportions_series


FP_TYPE = np.float32

# boosting parameters typing
boosting_parameters_dtypes = {
    "learning_rate": float,
    "max_depth": int,
    "min_data_in_leaf": int,
    "num_iterations": int,
    "n_estimators": int,
    "subsample": float,
    "bagging_fraction": float,
    "bagging_freq": int,
    "lambda_l1": float,
    "lambda_l2": float,
    "feature_fraction": float,
    "gamma": float,
    "reg_alpha": float,
    "reg_lambda": float,
    "colsample_bytree": float,
    "colsample_bylevel": float,
    "min_child_weight": int,
    "random_state": int,
    "n_jobs": int,
}

class Mixer(ABC):
    """
    Abstract class for mixers.
    """

    @abstractmethod
    def generate(
        self, modeled_cell: str, make_noise: bool = True, random_seed: int = 0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generating mixes for training deconvolution model of particular cell type.
        :param modeled_cell: str name of modeled cell matched to one of the cells types in type tree
        :param make_noise: if noise to be made on the data
        :param random_seed: int random seed for reproducible data generation
        :return:
        """

    def load(
        self, modeled_cell: str, path: str, level: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loading prepared mixes for model training.
        :param level: level of model
        :param path: path to folder with saved mixes tables in tsv
        :param modeled_cell: str name of modeled cell matched to one of the cells types in type tree
        """
        path = Path(path)
        path = path / f"l{level}" / modeled_cell
        expr = pd.read_csv(path / "expr.tsv", sep="\t", index_col=0)
        values = pd.read_csv(path / "values.tsv", sep="\t", index_col=0)
        return expr, values

    def save(
        self,
        generate_params: dict,
        path: str,
        modeled_cells=None,
        random_seed=0,
        n_levels=2,
    ):
        """
        Generate mixes and saves them for reuse.
        :param modeled_cell: str name of modeled cell matched to one of the cells types in type tree.
        :param generate_params: parameters that are used in generate function.
        :param modeled_cells: list of cells for which mixes are generated.
        :param n_levels: number of levels in model for which sepparate mixes will be generated.
        :param path: str path to folder where mixes are saved.
        """
        pass

        # leaving metadata to check if you loading what you intending to load
        params = {"version": self._version, "mixer_type": self.mixer_type}
        with open(path / "params.yml", "w") as f:
            yaml.dump(params, f)
        print(generate_params)
        for i in range(n_levels):
            print(f"Generating l{i + 1} mixes")
            level_path = path / f"l{i + 1}"
            level_path.mkdir(exist_ok=True)
            for cell in modeled_cells:
                print(cell)
                mixes_path = level_path / cell
                mixes_path.mkdir(exist_ok=True)
                expr, values = self.generate(
                    cell, random_seed=random_seed, **generate_params
                )
                expr.to_csv(mixes_path / "expr.tsv", sep="\t")
                values.to_csv(mixes_path / "values.tsv", sep="\t")

    @property
    def version(self):
        """
        Class version to avoid unentended compatability.
        """
        return self._version

    @property
    def mixer_type(self):
        """
        Mixer type to avoid loading data from wrong Mixer implementation.
        """
        return self.__class__.__name__
    

class Model(ABC):
    """
    Abstact class for deconvolution model.
    """

    @abstractmethod
    def fit(self, mixer: Mixer):
        """
        Model fitting pipeline.
        :param mixer: initialized object of mixer class
        """

    @abstractmethod
    def predict(self, expr: pd.DataFrame) -> pd.DataFrame:
        """
        Model prediction pipeline.
        :param expr: samples expressions in TPM
        :return: pd df of predictions
        """

    @abstractmethod
    def save(self, path: str):
        """
        Saving model.
        :param path: path to where model will be saved.
        """

    @abstractmethod
    def load(self, path: str):
        """
        Loading model.
        :param path: path to saved model.
        """





class CellsMixer(Mixer):
    """
    Base class for mix generation. Handles cells expression mixing and noise adding.
    """

    def __init__(
        self,
        cell_types: CellTypes,
        cells_expr: pd.DataFrame,
        cells_annot: pd.DataFrame,
        num_points: int = 1000,
        rebalance_param: float = 0.3,
        gene_length: str = "/uftp/Deconvolution/training/config/gene_length_values.tsv",
        genes_in_expression_path="/uftp/Deconvolution/product/training/genes/genes_v2.txt",
        num_av: int = 5,
        all_genes: bool = False
    ):
        """
        :param proportions: pandas Series with numbers for proportions for each type
        :param cell_types: Object of class CellTypes
        :param gene_length: path to table with gene lengths values
        :param rebalance_parameter: whether to reduce the weight of large datasets when forming random
                                    samples selection, None or 0 < rebalance_parameter <= 1
                                    rebalance_parameter == 1: equal number of samples from each dataset
        :param poisson_noise_level: coeff for Poisson noise level (larger - higher noise)
        :param uniform_noise_level: coeff for uniform noise level (larger - higher noise)
        :param dirichlet_samples_proportion: fraction of cell mixes that will be formed through
                                            the dirichlet distribution for method 'concat_ratios_with_dirichlet'
                                            Value must be in the range from 0 to 1.
        :param num_av: number of random samples of cell type that will be averaged to form the resulting sample
        :param num_points: number of resulting samples for each cell type
        :param all_genes: genes to consider in mixing. Uses all genes from cells_config if none provided.
        :param random_seed: fixed random state
        """
        self.num_points = num_points
        self.cell_types = cell_types
        self.rebalance_param = rebalance_param
        self.num_av = num_av
        self.proportions = get_proportions_series(cell_types)
        self.gene_length = pd.read_csv(gene_length, sep="\t", index_col=0)
        self.cells_annot = cells_annot

        self.genes_in_expression = []
        with open(genes_in_expression_path, "r") as f:
            for line in f:
                self.genes_in_expression.append(line.strip())

        print("Checking normal cells annotation...")
        self.check_annotation()
        print("Checking normal cells expressions...")
        self.check_expressions(cells_expr)

        # renormalizing expressions if different genes
        cells_expr = cells_expr.loc[self.genes_in_expression]
        cells_expr = (cells_expr / cells_expr.sum()) * np.float32(10) ** np.float32(6)
        self.cells_expr = cells_expr if all_genes else cells_expr.loc[cell_types.genes]
        self.cells_expr = cells_expr.astype('float32')
        self._version = 0.2

    def check_annotation(self):
        """
        Checks if annotation has all subtypes required for models.
        """
        diff = set(self.cell_types.leaves).difference(self.cells_annot["Cell_type"].unique())
        if diff:
            raise ValueError(
                "MISSING CELL TYPES IN ANNOTATION: " + ', '.join(diff)
            )
        print("Annotation OK")

    def check_expressions(self, expr):
        """
        Checks if expressions have the right format.
        """
        if not any(expr.max(axis=1) > np.log2(10 ** 6)):
            raise ValueError(
                "MODEL DOES NOT WORK WITH LOG NORMALIZED DATA. LINEARIZE YOUR EXPRESSION MATRIX."
            )
        diff = set(self.cell_types.genes).difference(set(expr.index))
        if diff:
            raise ValueError(
                f"EXPRESSION MATRIX HAS TO CONTAIN AT LEAST ALL THE GENES THAT ARE USED AS A FEATURES \n {diff}"
            )
        diff = set(self.cell_types.genes).symmetric_difference(set(expr.index))
        if not diff:
            print(
                "WARNING: YOU USING ONLY FEATURE GENES. "
                "MAKE SURE THAT EXPRESSIONS WERE NORMALIZED ON THIS SET OF GENES."
            )
        else:
            print("Expressions OK")

    def generate(
        self, modeled_cell: str, make_noise: bool = True, random_seed: int = 0
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generates mixes for cell model training.
        :modeled_cell: Cell type for which model training mixes is to be assambled
        :random_seed: random seed
        :returns: tuple with dataframes of mixed expressions and rna proportions
        """
        np.random.seed(random_seed)

        mixed_cells_expr = pd.DataFrame(
            np.zeros((len(self.cells_expr.index), self.num_points)),
            index=self.cells_expr.index,
            columns=range(self.num_points),
            dtype=float,
        )

        cells_to_mix = self.get_cells_to_mix(modeled_cell)
        average_cells = {
            **self.generate_pure_cell_expressions(1, cells_to_mix),
            **self.generate_pure_cell_expressions(self.num_av, [modeled_cell]),
        }

        mixed_cells_values = self.concat_ratios_with_dirichlet(cells_to_mix)

        for cell in mixed_cells_values.index:
            mixed_cells_expr += mixed_cells_values.loc[cell] * average_cells[cell]

        modeled_cell_values = self.normal_cell_distribution(
            mean=self.cell_types[modeled_cell].cell_proportion
        )

        other_cells_values = np.int32(1) - modeled_cell_values
        mixed_cells_values *= other_cells_values
        mixed_cells_expr *= other_cells_values
        mixed_cells_expr += modeled_cell_values * average_cells[modeled_cell]
        del average_cells
        mixed_cells_values.loc[modeled_cell] = modeled_cell_values

        if make_noise:
            mixed_cells_expr = self.make_noise(mixed_cells_expr)

        # mixed_cells_expr = round(mixed_cells_expr, 2) # no point in kiping 9 figures after .
        gc.collect()
        return mixed_cells_expr, mixed_cells_values

    @staticmethod
    def rebalance_samples_by_type(annot: pd.DataFrame, k: float) -> pd.Index:
        """
        Function rebalances the annotation dataset: rare types (type is based on column 'col')
        appears more often due to the multiplication of their samples in the dataset.
        All NaN samples will be deleted.

        k == 0: no rebalance
        k == 1: number of samples of each type in 'col' increases to maximum
        0 < k < 1: rebalance based on 'func'

        :param annot: pandas annotation dataframe (samples as indices)
        :param k: rebalance parameter 0 < k < 1
        :return: list of samples
        """
        type_counter = annot["Dataset"].value_counts()

        def func(x, k):
            return x ** (1 - k)

        max_counter = type_counter.max()
        type_counter = np.round(func(x=type_counter / max_counter, k=k) * max_counter).astype(
            int
        )

        samples = []
        for t, counter in type_counter.items():
            samples.extend(
                np.random.choice(annot.loc[annot["Dataset"] == t].index, counter)
            )

        return pd.Index(samples)

    def generate_pure_cell_expressions(
        self, num_av: int, cells_to_mix: List[str]
    ) -> Dict[str, float]:
        """
        Function makes averaged samples of random cellular samples, taking into account the nested structure
        of the subtypes and the desired proportions of the subtypes for cell type.
        :param cells_to_mix: list of cell types for which averaged samples from random selection will be formed
        :param num_av: number of random samples of cell type that will be averaged to form the resulting sample
        :returns: dict with matrix of average of random num_av samples for each cell type with replacement
        """
        average_cells = {}
        for cell in cells_to_mix:
            cells_selection = self.select_cells_with_subtypes(cell)
            expressions_matrix = pd.DataFrame(
                np.zeros((len(self.cells_expr.index), self.num_points)),
                index=self.cells_expr.index,
                columns=range(self.num_points),
                dtype=np.float32,
            )
            for i in range(num_av):
                if self.rebalance_param is not None:
                    cells_index = pd.Index(
                        self.rebalance_samples_by_type(
                            self.cells_annot.loc[cells_selection.index],
                            k=self.rebalance_param,
                        )
                    )
                else:
                    cells_index = cells_selection.index
                if self.proportions is not None:
                    cell_subtypes = self.cell_types.get_all_subtypes(cell)
                    specified_subtypes = set(
                        self.proportions.dropna().index
                    ).intersection(cell_subtypes)
                    if len(specified_subtypes) > 1:
                        cells_index = self.change_subtype_proportions(
                            cell=cell, cells_index=cells_index
                        )
                samples = random.choice(cells_index, self.num_points)
                expressions_matrix += np.self.cells_expr.loc[:, samples].values
            average_cells[cell] = expressions_matrix / np.int32(num_av)
        return average_cells

    def select_cells_with_subtypes(self, cell: str) -> pd.DataFrame:
        """
        Method makes a selection of all cell type samples with all level nested subtypes.
        :param cell: cell type from names in 'Cell_type'
        :returns: pandas Series with samples indexes and cell names
        """
        selected_cells = [cell] + self.cell_types.get_all_subtypes(cell)
        return self.cells_annot[self.cells_annot["Cell_type"].isin(selected_cells)]

    def change_subtype_proportions(self, cell: str, cells_index: pd.Index) -> pd.Index:
        """
        Function changes the proportions of the cell subtypes when they are considered as types for random selection.
        The proportions of the subtypes will be changed including samples of deeper subtypes
        :param cell: string with the name of cell type for which the proportions of the subtypes will be changed
        :param cells_index: pandas index of samples for cell type
        :returns: array of sample indexes oversampled for needed proportions
        """
        cell_subtypes = self.cell_types.get_direct_subtypes(cell)
        specified_subtypes = set(self.proportions.dropna().index).intersection(
            cell_subtypes
        )

        # cell type samples and samples without specified subtype proportion
        unspecified_types = list(set(cell_subtypes).difference(specified_subtypes)) + [
            cell
        ]
        unspecified_samples = cells_index[
            self.cells_annot.loc[cells_index, "Cell_type"].isin(unspecified_types)
        ]
        min_num = min(self.proportions.loc[specified_subtypes])

        subtype_proportions = {cell: dict(self.proportions.loc[specified_subtypes])}

        subtype_samples = {}
        subtype_size = {}
        oversampled_subtypes = {}
        for subtype in specified_subtypes:
            subtype_subtypes = self.cell_types.get_direct_subtypes(subtype)
            subtype_has_subtypes = (
                len(set(self.proportions.dropna().index).intersection(subtype_subtypes))
                > 1
            )

            subtype_samples[subtype] = self.select_cells_with_subtypes(subtype).index

            if subtype_has_subtypes:
                subtype_samples[subtype] = self.change_subtype_proportions(
                    cell=subtype, cells_index=subtype_samples[subtype]
                )
            subtype_size[subtype] = len(subtype_samples[subtype])
        max_size = max(subtype_size.values())
        result_samples = unspecified_samples
        for subtype in specified_subtypes:
            oversampled_subtypes[subtype] = np.random.choice(
                subtype_samples[subtype],
                int(subtype_proportions[cell][subtype] * max_size / min_num + 1),
            )
            result_samples = np.concatenate(
                (result_samples, oversampled_subtypes[subtype])
            )
        return result_samples

    def concat_ratios_with_dirichlet(
        self, cells_to_mix: List[str], dirichlet_samples_proportion=0.4
    ):
        """
        Function generates the values of the proportion of mixed cells by combining simple_ratios and dirichlet methods.
        :param num_points: int number of how many mixes to create
        :param likely_proportions: None for uniform proportions or pandas Series
                                   with numbers for proportions for each type
        :param cells_to_mix: list of cell types to mix
        :param dirichlet_samples_proportion: fraction of cell mixes that will be formed through
                                             the dirichlet distribution for method 'concat_ratios_with_dirichlet'
                                             Value must be in the range from 0 to 1.
        :returns: pandas dataframe with generated cell type fractions
        """
        num_dirichlet_points = int(self.num_points * dirichlet_samples_proportion)
        mix_ratios_values = self.simple_ratios_mixing(
            self.num_points - num_dirichlet_points, cells_to_mix
        )
        mix_dirichlet_values = self.dirichlet_mixing(num_dirichlet_points, cells_to_mix)
        mix_cell_values = pd.concat([mix_ratios_values, mix_dirichlet_values], axis=1)
        mix_cell_values.columns = range(self.num_points)
        return mix_cell_values

    def simple_ratios_mixing(self, num_points: int, cells_to_mix: List[str]):
        """
        Method generates the values of the proportion of mixed cells by simple ratios method.
        In this method, the areas of values in the region of given proportions are well enriched,
        but very few values that far from these areas.
        :param num_points: int number of how many mixes to create
        :param cells_to_mix: list of cell types to mix
        :returns: pandas dataframe with generated cell type fractions
        """
        mixed_cell_values = pd.DataFrame(
            np.random.random(size=(len(cells_to_mix), num_points)),
            index=cells_to_mix,
            columns=range(num_points),
        )
        mixed_cell_values = (mixed_cell_values.T * self.proportions.loc[cells_to_mix]).T
        return mixed_cell_values / mixed_cell_values.sum()

    def dirichlet_mixing(self, num_points: int, cells_to_mix: List[str]):
        """
        Method generates the values of the proportion of mixed cells by dirichlet method.
        The method guarantees a high probability of the the presence of each cell type from 0 to 100%
        at the expense of enrichment of fractions close to zero.
        :param num_points: int number of how many mixes to create
        :param cells_to_mix: list of cell types to mix
        :returns: pandas dataframe with generated cell type fractions
        """
        return pd.DataFrame(
            np.random.dirichlet(
                [1.0 / len(cells_to_mix)] * len(cells_to_mix), size=num_points
            ).T,
            index=cells_to_mix,
            columns=range(num_points),
        )

    def get_cells_to_mix(self, modeled_cell: str) -> List[str]:
        """
        Returns list of cells to mix for modeld cell type.
        """
        cells_to_remove = [modeled_cell]
        cells_to_remove += self.cell_types.get_all_parents(modeled_cell)
        cells_to_remove += self.cell_types.get_all_subtypes(modeled_cell)
        cells_to_mix = []
        for cell in cells_to_remove:
            cells_to_mix += self.cell_types.get_direct_subtypes(cell)

        cells_to_mix = [cell for cell in cells_to_mix if cell not in cells_to_remove]
        return cells_to_mix

    def normal_cell_distribution(self, sd=0.5, mean=0.5) -> pd.Series:
        """
        Generates vector with normal distribution truncated on [0,1] for cell mixing.
        :param sd: Standard deviation
        :param mean: mean
        :returns: np.array with values
        """
        mean, sd = FP_TYPE(mean), FP_TYPE(sd)
        values = sd * np.random.randn(self.num_points).astype(FP_TYPE) + mean
        values[values < 0] = np.random.uniform(size=len(values[values < 0])).astype(FP_TYPE)
        values[values > 1] = np.random.uniform(size=len(values[values > 1])).astype(FP_TYPE)
        return pd.Series(values, index=range(self.num_points))

    def make_noise(
        self, data: pd.DataFrame, poisson_noise_level=0.5, uniform_noise_level=0
    ) -> pd.DataFrame:
        """
        Method adds Poisson noise (very close approximation) and uniform noise for expressions in TPM.
        Uniform noise - proportional to gene expressions noise from a normal distribution.
        :param data: pandas dataframe with expressions in TPM with genes as indexes
        :returns: dataframe data with added noise
        """
        X = data.to_numpy(dtype=FP_TYPE)
        length_normed_data = (
            (X * 1000) /
            np.array(self.gene_length.loc[data.index, "length"])[:, None])
        if uniform_noise_level != 0:
            X += np.sqrt(poisson_noise_level * length_normed_data
                         ) * np.random.normal(size=X.shape) + X * np.random.normal(
                             size=X.shape, scale=uniform_noise_level)
        else:
            X += np.sqrt(poisson_noise_level *
                         length_normed_data) * np.random.normal(size=X.shape)
        x = np.clip(X, a_min=0, a_max=None)
        return pd.DataFrame(x, index=data.index, columns=data.columns)

    def select_datasets_fraction(self, annotation, selector_col, bootstrap_fraction):
        """
        Function selects random datasets for every cell name (without nested subtypes) without replacement
        :param annotation: pandas dataframe with colum 'Dataset' and samples as index
        :param bootstrap_fraction: fraction of datasets to select
        :returns: list of sample indexes for selected datasets
        """
        selected_samples = []
        values = annotation[selector_col].unique()
        for value in values:
            value_inds = annotation.loc[annotation[selector_col] == value].index
            value_inds = value_inds.difference(selected_samples)
            cell_datasets = set(annotation.loc[value_inds, "Dataset"].unique())
            if cell_datasets:
                selected_datasets = np.random.choice(
                    list(cell_datasets),
                    int(len(cell_datasets) * bootstrap_fraction) + 1,
                    replace=False,
                )
                selected_samples.extend(
                    annotation[
                        annotation["Dataset"].isin(selected_datasets)
                    ].index.intersection(value_inds)
                )
        return selected_samples





class TumorMixer(CellsMixer):
    """
    Class for mix generation. Handles cells expression mixing and noise adding.
    """

    def __init__(
        self,
        cell_types: CellTypes,
        cells_expr: pd.DataFrame,
        cells_annot: pd.DataFrame,
        tumor_expr: pd.DataFrame,
        tumor_annot: pd.DataFrame,
        tumor_mean=0.5,
        tumor_sd=0.5,
        hyperexpression_fraction=0.01,
        max_hyperexpr_level=1000,
        num_points: int = 1000,
        rebalance_param: float = 0.3,
        gene_length: str = "/uftp/Deconvolution/training/config/gene_length_values.tsv",
        genes_in_expression_path="/uftp/Deconvolution/product/training/genes/genes_v2.txt",
        num_av: int = 5,
        all_genes=None,
    ):
        """
        :param proportions: pandas Series with numbers for proportions for each type
        :param cell_types: Object of class CellTypes
        :param gene_length: path to table with gene lengths values
        :param rebalance_parameter: whether to reduce the weight of large datasets when forming random samples selection
                                    None or 0 < rebalance_parameter <= 1
                                    rebalance_parameter == 1: equal number of samples from each dataset
        :param poisson_noise_level: coeff for Poisson noise level (larger - higher noise)
        :param uniform_noise_level: coeff for uniform noise level (larger - higher noise)
        :param dirichlet_samples_proportion: fraction of cell mixes that will be formed through
                                             the dirichlet distribution for method 'concat_ratios_with_dirichlet'
                                            Value must be in the range from 0 to 1.
        :param num_av: number of random samples of cell type that will be averaged to form the resulting sample
        :param num_points: number of resulting samples for each cell type
        :param genes: genes to consider in mixing. Uses all genes from cells_config if none provided.
        :param random_seed: fixed random state
        """
        super().__init__(
            cell_types=cell_types,
            cells_expr=cells_expr.astype(FP_TYPE),
            cells_annot=cells_annot,
            num_points=num_points,
            rebalance_param=rebalance_param,
            gene_length=gene_length,
            genes_in_expression_path=genes_in_expression_path,
            num_av=num_av,
            all_genes=all_genes
        )

        print("Checking cancer cells expressions...")
        self.check_expressions(tumor_expr)

        # renormalizing tumor expressions
        tumor_expr = tumor_expr.loc[self.genes_in_expression].astype(FP_TYPE)
        self.tumor_expr = (tumor_expr / tumor_expr.sum()) * FP_TYPE(1e6)

        self.tumor_annot = tumor_annot
        self.tumor_mean = FP_TYPE(tumor_mean)
        self.tumor_sd = FP_TYPE(tumor_sd)
        self.hyperexpression_fraction = FP_TYPE(hyperexpression_fraction)
        self.max_hyperexpr_level = FP_TYPE(max_hyperexpr_level)

    def generate(
        self,
        modeled_cell: str,
        genes: tp.Union[tp.Set, tp.List] = None,
        random_seed: int = 0,
    ) -> tp.Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generates mixes for cell model training.
        :param modeled_cell: cell type for which model training mixes is to be assembled
        :param genes: Subset of genes outputted in resulted mixes expressions.
                      Uses genes for cell type from config if None. Affects execution speed.
        :param random_seed: random seed
        :return: tuple with dataframes of mixed expressions and rna proportions
        """
        np.random.seed(random_seed)

        if not genes:
            genes = self.cell_types[modeled_cell].genes

        cells_to_mix = self.get_cells_to_mix(modeled_cell)
        fractions = self.get_fractions(modeled_cell=modeled_cell, cells_to_mix=cells_to_mix)

        mixed_expr = self.generate_fractions(
            modeled_cell=modeled_cell,
            cells_to_mix=cells_to_mix,
            fractions=fractions,
            genes=genes
        )

        return mixed_expr, fractions

    def get_fractions(self,
                      modeled_cell: str,
                      cells_to_mix: tp.List[str]) -> pd.DataFrame:
        """
        Randomly generate cell composition values for modeled cell, cells to mix and tumor portion.
        Adjust them to 1. These fractions can be used to generate mixes without any readjustments.
        :param modeled_cell: main cell to be modeled in the mixes
        :param cells_to_mix: list of cells to mix with the modeled cell
        :return: dataframe, columns -- mix number, rows -- cell fractions
        """
        cells_to_mix_values = self.dirichlet_mixing(cells_to_mix)
        modeled_cell_values = self.normal_cell_distribution(mean=self.cell_types[modeled_cell].cell_proportion)
        tumor_values = self.normal_cell_distribution(mean=self.tumor_mean, sd=self.tumor_sd)

        cells_fractions = cells_to_mix_values * (1 - modeled_cell_values)
        cells_fractions.loc[modeled_cell] = modeled_cell_values
        fractions = cells_fractions * (1 - tumor_values)
        fractions.loc["Tumor"] = tumor_values
        return fractions

    def generate_fractions(
        self,
        modeled_cell: str,
        cells_to_mix: tp.List[str],
        fractions: tp.Union[pd.Series, pd.DataFrame],
        genes: tp.Union[tp.Set, tp.List] = None
    ) -> pd.DataFrame:
        """
        Generate mixes from known cell fractions.
        :param modeled_cell: main cell to be modeled in the mixes
        :param cells_to_mix: list of cells to mix with the modeled cell
        :param fractions: fractions to be simulated in the mixes, in case of series only one fraction will be simulated
                          in all mixes, in case of dataframe -- corresponding mix will be simulated with its own values
        :param genes: genes to be included in resulting expressions
        :return: table with generated expressions, columns -- mixes, rows -- genes
        """
        average_cells = {
            **self.generate_pure_cell_expressions(genes, 1, cells_to_mix),
            **self.generate_pure_cell_expressions(genes, self.num_av, [modeled_cell]),
            "Tumor": self.generate_tumor_expressions(genes)
        }

        mixed_expr = pd.DataFrame(
            np.zeros((len(genes), self.num_points), dtype=FP_TYPE),
            index=genes,
            columns=range(self.num_points)
        )
        for cell in fractions.index:
            mixed_expr += fractions.loc[cell] * average_cells[cell]
        mixed_expr = self.make_noise(mixed_expr)

        return mixed_expr

    def generate_pure_cell_expressions(
        self, genes: list, num_av: int, cells_to_mix: tp.List[str]
    ) -> tp.Dict[str, float]:
        """
        Function makes averaged samples of random cellular samples, taking into account the nested structure
        of the subtypes and the desired proportions of the subtypes for cell type.
        :param cells_to_mix: list of cell types for which averaged samples from random selection will be formed
        :param num_av: number of random samples of cell type that will be averaged to form the resulting sample
        :returns: dict with matrix of average of random num_av samples for each cell type with replacement
        """
        average_cells = {}
        cells_expr = self.cells_expr.loc[genes]
        for cell in cells_to_mix:
            cells_selection = self.select_cells_with_subtypes(cell)
            expressions_matrix = pd.DataFrame(
                np.zeros((len(cells_expr.index), self.num_points), dtype=FP_TYPE),
                index=cells_expr.index,
                columns=range(self.num_points)
            )
            for i in range(num_av):
                if self.rebalance_param is not None:
                    cells_index = pd.Index(
                        self.rebalance_samples_by_type(
                            self.cells_annot.loc[cells_selection.index],
                            k=self.rebalance_param,
                        )
                    )
                else:
                    cells_index = cells_selection.index
                if self.proportions is not None:
                    cell_subtypes = self.cell_types.get_all_subtypes(cell)
                    specified_subtypes = set(
                        self.proportions.dropna().index
                    ).intersection(cell_subtypes)
                    if len(specified_subtypes) > 1:
                        cells_index = self.change_subtype_proportions(
                            cell=cell, cells_index=cells_index
                        )
                samples = random.choice(cells_index, self.num_points)
                expressions_matrix += cells_expr.loc[:, samples].values
            average_cells[cell] = expressions_matrix / FP_TYPE(num_av)
        return average_cells

    def generate_tumor_expressions(self, genes) -> pd.DataFrame:
        tumor_expr = self.tumor_expr.loc[genes].sample(self.num_points, replace=True, axis=1)
        tumor_expr.columns = range(self.num_points)
        tumor_expr = self.add_tumor_hyperexpression(
            tumor_expr,
            hyperexpression_fraction=self.hyperexpression_fraction,
            max_hyperexpr_level=self.max_hyperexpr_level
        )
        return tumor_expr

    def dirichlet_mixing(self, cells_to_mix: tp.List[str]) -> pd.DataFrame:
        """
        Method generates the values of the proportion of mixed cells by dirichlet method.
        The method guarantees a high probability of the the presence of each cell type from 0 to 100%
        at the expense of enrichment of fractions close to zero.
        :param cells_to_mix: list of cell types to mix
        :returns: pandas dataframe with generated cell type fractions
        """
        return pd.DataFrame(
            np.random.dirichlet(
                [1.0 / len(cells_to_mix)] * len(cells_to_mix), size=self.num_points
            ).astype(FP_TYPE).T,
            index=cells_to_mix,
            columns=range(self.num_points)
        )

    @staticmethod
    def add_tumor_hyperexpression(data, hyperexpression_fraction, max_hyperexpr_level):
        """
        :param data: pandas dataframe with expressions in TPM
        :param hyperexpression_fraction: probability for gene to be hyperexpressed
        :param max_hyperexpr_level: maximum level of tumor expression
        :return:
        """
        tumor_noise = np.random.random(size=data.shape)
        tumor_noise = np.where(
            tumor_noise < hyperexpression_fraction, max_hyperexpr_level, 0
        )
        tumor_noise = np.float32(tumor_noise * np.random.random(size=data.shape))
        data = data + tumor_noise
        return data

    def save(self, path: tp.Union[str, Path]):
        path = Path(path)
        with (path / "cells_annot.tsv").open("w") as f:
            self.cells_annot.to_csv(f, sep="\t")
        with (path / "tumor_annot.tsv").open("w") as f:
            self.tumor_annot.to_csv(f, sep="\t")
        with (path / "tumor_expr.tsv").open("w") as f:
            self.tumor_expr.to_csv(f, sep="\t")
        with (path / "cells_expr.tsv").open("w") as f:
            self.cells_expr.to_csv(f, sep="\t")

        params = {
            "num_points": self.num_points,
            "rebalance_param": self.rebalance_param,
            "num_av": self.num_av,
        }
        with open(path / "mixer_params.yml", "w") as f:
            yaml.dump(params, f)
            
DEFAULT_BOOSTING_PARAMETERS_PATH = {
    "first": Path("/uftp/Deconvolution/product/training/boosting_configs/v1/lgb_parameters_first_step.tsv"),
    "second": Path("/uftp/Deconvolution/product/training/boosting_configs/v1/lgb_parameters_second_step.tsv")
}
DEFAULT_GENES_PATH = Path("/uftp/Deconvolution/product/training/genes/genes_v2.txt")


class TumorModel:
    """
    Base class for model training and prediction.
    """

    def __init__(
        self,
        cell_types: CellTypes,
        boosting_params_first_step: tp.Union[str, Path] = DEFAULT_BOOSTING_PARAMETERS_PATH["first"],
        boosting_params_second_step: tp.Union[str, Path] = DEFAULT_BOOSTING_PARAMETERS_PATH["second"],
        genes_in_expression_path: tp.Union[str, Path] = DEFAULT_GENES_PATH,
        l1_models: tp.Dict[str, lgb.LGBMRegressor] = None,
        l2_models: tp.Dict[str, lgb.LGBMRegressor] = None,
        early_stopping_rounds: int = 1000,
        random_seed: int = 0,
        n_jobs: int = 5
    ):
        """
        :param cell_types: Object of class CellTypes
        :param boosting_params_first_step: path to boosting parameters for the first step
        :param boosting_params_second_step: path to boosting parameters for the second step
        :param random_seed: random seed
        """
        self.cell_types = cell_types
        self.random_seed = random_seed
        self.early_stopping_rounds = early_stopping_rounds
        self.boosting_params_first_step = pd.read_csv(
            boosting_params_first_step,
            sep="\t",
            index_col=0,
            dtype=boosting_parameters_dtypes,
        )
        self.boosting_params_second_step = pd.read_csv(
            boosting_params_second_step,
            sep="\t",
            index_col=0,
            dtype=boosting_parameters_dtypes,
        )
        self.l1_models = {} if l1_models is None else l1_models
        self.l2_models = {} if l2_models is None else l2_models

        self.genes_in_expression = []
        with Path(genes_in_expression_path).open("r") as f:
            for line in f:
                self.genes_in_expression.append(line.strip())
        self.n_jobs = n_jobs

    def fit(
        self,
        mixer: TumorMixer,
        verbose: int = 100,
        early_stopping_rounds: int = 1000,
        train_l2: bool = True,
        n_val_points: int = 3000,
        n_iter: int = 1,
    ):
        """
         Training pipeline for this model.
        :param mixer: object of Mixer/TumorMixer/... class
        :param verbose: see verbose in LGBMRegressor, default is 100
        :param early_stopping_rounds: see early_stopping_rounds in LGBMRegressor, default is 1000
        :param train_l2: if you want to train 2nd level model, default is True
        :param n_val_points: number of validation points, default is 3000
        :param n_iter: number of iterations, default is 1
        :return:
        """
        np.random.seed(self.random_seed)

        self.check_mixer(mixer, n_val_points)

        start = timeit.default_timer()

        print("============== L1 models ==============")
        for i, cell in enumerate(self.cell_types.models):
            for j in range(n_iter):
                print(f"###### Iteration: {j + 1} #####")
                print(f"Generating mixes for {cell} model")
                start1 = timeit.default_timer()
                expr, values = mixer.generate(
                    cell, genes=self.cell_types[cell].genes, random_seed=i + 1 * j
                )
                end1 = timeit.default_timer()
                print(f"Mix generation done in:  {round(end1-start1, 1)} sec.")
                print(f"Fitting {cell} model")
                self.l1_models[cell] = self.train_l1_model(
                    expr, values, cell, verbose, early_stopping_rounds, n_val_points
                )
                print("\n")

        if train_l2:
            print("============== L2 models ==============")
            for i, cell in enumerate(self.cell_types.models):
                for j in range(n_iter):
                    print(f"###### Iteration: {j + 1} #####")
                    print(f"Generating mixes for {cell} model")
                    start1 = timeit.default_timer()
                    expr, values = mixer.generate(
                        cell, genes=self.cell_types.genes, random_seed=i + 1007 * j
                    )
                    end1 = timeit.default_timer()
                    print(f"Mix generation done in:  {round(end1-start1, 1)} sec.")
                    print(f"Fitting {cell} model")
                    self.l2_models[cell] = self.train_l2_model(
                        expr, values, cell, verbose, early_stopping_rounds, n_val_points
                    )
                    print("\n")

        end = timeit.default_timer()
        print(f"Deconv model fitting done in: {round(end-start, 1)} sec.")

    def check_mixer(self, mixer: TumorMixer, n_val_points: int):
        if mixer.num_points < n_val_points * 3:
            raise ValueError(
                f"MIXER num_points (num_points={mixer.num_points}) SHOULD BE AT LEAST 3 TIMES MORE "
                f"THAN n_val_points (n_val_points={n_val_points})."
            )

    def train_l1_model(
        self, expr, values, cell, verbose, early_stopping_rounds, n_val_points
    ):
        """
        Trains L1 model for one cell type.
        :param expr: pd df with samples in columns and genes in rows
        :param values: pd df with true RNA fractions
        :param cell: cell type for which model is trained
        :return: trained model for cell type
        """
        print("Preparing train/validation sets...")
        features = sorted(list(set(self.cell_types[cell].genes)))
        x = expr.T[features]
        x = x.sample(frac=1)
        y = values.loc[cell].loc[x.index]
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        train_x = x[n_val_points:]
        train_y = y[n_val_points:]
        val_x = x[:n_val_points]
        val_y = y[:n_val_points]

        boosting_params = self.boosting_params_first_step.to_dict(orient="index")[cell]
        model = lgb.LGBMRegressor(**boosting_params, random_state=0, n_jobs=self.n_jobs)
        if cell in self.l1_models.keys():
            print(f"Continuing with existing {cell} model...")
            model.fit(
                train_x,
                train_y,
                eval_set=[(val_x, val_y)],
                eval_metric=["l1", "l2"],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
                init_model=self.l1_models[cell]
            )
        else:
            print(f"Fitting new {cell} model...")
            model.fit(
                train_x,
                train_y,
                eval_set=[(val_x, val_y)],
                eval_metric=["l1", "l2"],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose
            )

        return model

    def train_l2_model(
        self, expr, values, cell, verbose, early_stopping_rounds, n_val_points
    ):
        """
        Trains L2 model for one cell type. Uses L1 models as an input features.
        :param expr: pd df with samples in columns and genes in rows
        :param values: pd df with true RNA fractions
        :param cell: cell type for which model is trained
        :return: trained model for cell type
        """
        print("Preparing train/validation sets...")
        features = sorted(list(set(self.cell_types.genes)))
        x = expr.T[features]
        x = x.sample(frac=1)
        l1_preds = self.predict_l1(x.T)
        features = sorted(list(set(self.cell_types[cell].genes)))
        x = x[features]
        x = pd.concat([x, l1_preds], axis=1)
        y = values.loc[cell].loc[x.index]
        x = x.reset_index(drop=True)
        y = y.reset_index(drop=True)

        train_x = x[n_val_points:]
        train_y = y[n_val_points:]
        val_x = x[:n_val_points]
        val_y = y[:n_val_points]

        boosting_params = self.boosting_params_second_step.to_dict(orient="index")[cell]
        model = lgb.LGBMRegressor(**boosting_params, random_state=0, n_jobs=self.n_jobs)
        if cell in self.l2_models.keys():
            print(f"Continuing with existing {cell} model...")
            model.fit(
                train_x,
                train_y,
                eval_set=[(val_x, val_y)],
                eval_metric=["l1", "l2"],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose,
                init_model=self.l2_models[cell]
            )
        else:
            print(f"Fitting new {cell} model...")
            model.fit(
                train_x,
                train_y,
                eval_set=[(val_x, val_y)],
                eval_metric=["l1", "l2"],
                early_stopping_rounds=early_stopping_rounds,
                verbose=verbose
            )
        return model

    def predict(self,
                expr: pd.DataFrame,
                use_l2: bool = False,
                percent_type: str = "cells",
                other_coef: float = 0.073468):
        """
        Prediction pipeline for the model.
        :param expr: pd df with samples in columns and genes in rows
        :param use_l2: whether to use level 2 models or not, default is False
        :param percent_type: "rna" or "cells" fractions to be returned, default is "cells"
        :param other_coef: coefficient for Other, default is 0.073468
        :return: pd df with predictions for cell types in rows and samples in columns
        """
        self.check_expressions(expr)
        expr = self.renormalize_expr(expr)

        preds = self.predict_l2(expr) if use_l2 else self.predict_l1(expr)
        preds = self.adjust_rna_fractions(preds, 0, other_coef)
        preds = self.convert_rna_to_cells_fractions(preds, other_coef) if percent_type == "cells" else preds
        preds = preds.T
        return preds

    def check_expressions(self, expr):
        """
        Checks if expressions have the right format.
        """
        if not any(expr.max(axis=1) > np.log2(10 ** 6)):
            raise ValueError(
                "MODEL DOES NOT WORK WITH LOG NORMALIZED DATA. LINEARIZE YOUR EXPRESSION MATRXI."
            )
        diff = set(self.cell_types.genes).difference(set(expr.index))
        if diff:
            raise ValueError(
                "EXPRESSION MATRIX HAS TO CONTAIN AT LEAST ALL THE GENES THAT ARE USED AS A FEATURES"
            )
        diff = set(self.cell_types.genes).symmetric_difference(set(expr.index))
        if not diff:
            print(
                "WARNING: YOU USING ONLY FEATURE GENES. MAKE SURE THAT NORMALIZATION IS CORRECT"
            )
        else:
            print("Expressions OK")

    def renormalize_expr(self, expr):
        sym_diff = set(self.genes_in_expression).symmetric_difference(set(expr.index))
        if len(sym_diff) > 0:
            expr = expr.loc[self.genes_in_expression]
            expr = (expr / expr.sum()) * 10 ** 6

        return expr

    def adjust_rna_fractions(self, preds, lod, add_other):
        """
        Adjusts predicted fractions based on cell types tree structure. Lower subtypes recalculated to sum up to
        value of its parent type.
        :param preds: pd df with predictions for cell types in columns and samples in rows.
        :add_other: if not None adds Other fraction in case if sum of all general cell types predictors yeilds < 1
        :returns: adjusted preds
        """
        preds[preds < lod] = 0
        cell = self.cell_types.root
        general_types = [
            ct
            for ct in self.cell_types.get_direct_subtypes(cell)
            if ct in self.cell_types.models
        ]
        # adding other
        for sample in preds.index:
            s = preds.loc[sample, general_types].sum()
            if s < 1 and add_other:
                preds.loc[sample, "Other"] = 1 - s
            else:
                preds.loc[sample, general_types] = preds.loc[sample, general_types] / s
                preds.loc[sample, "Other"] = 0

        cells_with_unadjusted_subtypes = general_types

        while cells_with_unadjusted_subtypes:
            cell = cells_with_unadjusted_subtypes.pop()
            subtypes = [
                ct
                for ct in self.cell_types.get_direct_subtypes(cell)
                if ct in self.cell_types.models
            ]
            preds[subtypes] = preds[subtypes].divide(
                preds[subtypes].sum(axis=1), axis=0
            )
            preds = preds.fillna(0)
            preds[subtypes] = preds[subtypes].multiply(preds[cell], axis=0)
            preds = preds.fillna(0)
            cells_with_unadjusted_subtypes = subtypes + cells_with_unadjusted_subtypes

        return preds

    def convert_rna_to_cells_fractions(self, rna_fractions, other_coeff):
        """
        Multiplies RNA fractions predictions for each cell on corresponded rna_per_cell coefficient from cell_config.yaml
        :param preds: pd df with RNA fractions predictions
        :return: pd df with adjusted predictions
        """
        rna_fractions = rna_fractions.T
        terminal_models = []
        for cell in self.cell_types.models:
            subtypes = self.cell_types.get_all_subtypes(cell)
            submodels = [c for c in subtypes if self.cell_types[c].model]
            if not submodels:
                terminal_models.append(cell)

        non_terminal_models = [
            cell for cell in self.cell_types.models if cell not in terminal_models
        ]

        cells_fractions = rna_fractions.loc[["Other"] + terminal_models]
        coefs = pd.Series(
            [other_coeff]
            + [self.cell_types[cell].rna_per_cell for cell in terminal_models]
        )
        terminal_models = ["Other"] + terminal_models
        coefs.index = terminal_models
        cells_fractions = cells_fractions.mul(coefs, axis="rows")
        cells_fractions = cells_fractions / cells_fractions.sum()
        while non_terminal_models:
            m = non_terminal_models.pop()
            submodels = self.cell_types.get_direct_subtypes(
                m
            )  # get all subtypes maybe???
            submodels = [cell for cell in submodels if cell in self.cell_types.models]
            # if its subtypes still unadjusted move it to the end of the queue
            skip = [cell for cell in submodels if cell in non_terminal_models]
            if skip:
                non_terminal_models = [m] + non_terminal_models
            else:
                cells_fractions.loc[m] = cells_fractions.loc[submodels].sum(axis=0)

        return cells_fractions.T

    def predict_l1(self, expr):
        """
        Predicts rna fractions by L1 models.
        :param expr: pd df with samples in columns and genes in rows.
        :return: L1 models predictions.
        """
        preds = {}
        for cell in sorted(self.l1_models.keys()):
            features = sorted(list(set(self.cell_types[cell].genes)))
            x = expr.T[features]
            preds[cell] = self.l1_models[cell].predict(x)
        preds = pd.DataFrame(preds)
        preds.index = x.index
        return preds

    def predict_l2(self, expr):
        """
        Predicts rna fractions by L2 models using L1 models predictions as an input features.
        :param expr: pd df with samples in columns and genes in rows.
        :return: L2 models predictions.
        """
        if not self.l2_models:
            raise ValueError("Level 2 models not found in this model!")

        preds = {}
        l1_preds = self.predict_l1(expr)
        for cell in sorted(self.l2_models.keys()):
            features = sorted(list(set(self.cell_types[cell].genes)))
            x = expr.T[features]
            x = pd.concat([x, l1_preds], axis=1)
            preds[cell] = self.l2_models[cell].predict(x)
        preds = pd.DataFrame(preds)
        preds.index = expr.columns
        return preds

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        l1_models_path = path / "l1_models"
        l1_models_path.mkdir(parents=True, exist_ok=True)
        for key, value in self.l1_models.items():
            with (l1_models_path / f"{key}.p").open("wb") as model_file:
                pickle.dump(value, model_file)

        l2_models_path = path / "l2_models"
        l2_models_path.mkdir(parents=True, exist_ok=True)
        for key, value in self.l2_models.items():
            with (l2_models_path / f"{key}.p").open("wb") as model_file:
                pickle.dump(value, model_file)

        self.cell_types.save(path / "cell_types.yaml")
        self.boosting_params_first_step.to_csv(
            path / "boosting_params_first_step.tsv", sep="\t"
        )
        self.boosting_params_second_step.to_csv(
            path / "boosting_params_second_step.tsv", sep="\t"
        )
        with open(path / "genes_in_expression.txt", "w") as f:
            for g in self.genes_in_expression:
                f.write(str(g) + "\n")

        with (path / "params.yaml").open("w") as f:
            f.write("# Model params \n")
            f.write("random_seed: " + str(self.random_seed) + "\n")

    @classmethod
    def load(cls, path: tp.Union[Path, str]):
        path = Path(path)
        model_files = (path / "l1_models").glob("**/*.p")

        l1_models = {}
        for model_file in model_files:
            with model_file.open("rb") as file:
                l1_models[model_file.stem] = pickle.load(file)

        l2_models = {}
        model_files = (path / "l2_models").glob("**/*.p")
        if model_files:
            for model_file in model_files:
                with model_file.open("rb") as file:
                    l2_models[model_file.stem] = pickle.load(file)

        cell_types = CellTypes.load(path / "cell_types.yaml", show_tree=False)

        return cls(
            cell_types=cell_types,
            boosting_params_first_step=path / "boosting_params_first_step.tsv",
            boosting_params_second_step=path / "boosting_params_second_step.tsv",
            genes_in_expression_path=path / "genes_in_expression.txt",
            l1_models=l1_models,
            l2_models=l2_models,
        )