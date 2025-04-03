import gc
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import yaml


ConfigAnnot = Dict[
    str, Union[Dict[str, Union[str, Dict[str, Union[str, float, int]]]], List[str]]
]


def renormalize_tpm(expr: pd.DataFrame, genes_in_expression: List[str]) -> pd.DataFrame:
    """Renornalize expression values so that expressions of one sample sum up to 10^6.

    Args:
        expr (pd.DataFrame): Expression values.
        genes_in_expression (List[str]): Genes to take into account.

    Returns:
        pd.DataFrame: Renormalized expression values.
    """
    expr = expr.loc[genes_in_expression]
    expr = (expr / expr.sum()) * 10**6

    return expr


def get_samples(sr: pd.Series, cts: List[str]) -> List[str]:
    """Return list of samples which belong to specified cell type.

    Args:
        sr (pd.Series): Series with cells annotation.
        cts (List[str]): Cell types to subset on.

    Returns:
        List[str]: List of samples which belong to specified cell type.
    """
    samples = []
    for ct in cts:
        sr_temp = sr.dropna().apply(lambda x: x.split(";"))
        condition = sr_temp.apply(
            lambda x: any(item in ct for item in x) if len(x) > 1 else x[0] == ct
        )
        samples += list(sr_temp[condition].index.values)
    samples = sorted(list(set(samples)))

    return sr[samples]


class Mixer:
    """
    Class for mix generation. Handles cells expression mixing and noise adding.
    """

    def __init__(
        self,
        config: ConfigAnnot,
        cells_expr: pd.DataFrame,
        cells_annot: pd.DataFrame,
        tumor_expr: pd.DataFrame,
        tumor_annot: pd.DataFrame,
        tumor_mean: float = 0.5,
        tumor_sd: float = 0.5,
        distribution: str = "uniform",
        hyperexpression_fraction: float = 0.01,
        max_hyperexpr_level: int = 1000,
        num_points: int = 1000,
        rebalance_param: float = 0.3,
        gene_length: str = "/uftp/Deconvolution/training/config/gene_length_values.tsv",
        genes_in_expression_path: Union[
            str, Path
        ] = "/uftp/Deconvolution/product/training/genes/genes_v2.txt",
        num_av: int = 3,
        random_seed: int = 42,
    ) -> None:
        self.FP_TYPE = np.float32
        self.num_points = num_points
        self.rebalance_param = rebalance_param
        self.num_av = num_av
        self.gene_length = pd.read_csv(gene_length, sep="\t", index_col=0)
        self.cells_annot = cells_annot
        self.config = config
        self.proportions = None

        self.genes_in_expression = []
        with open(genes_in_expression_path, "r") as f:
            for line in f:
                self.genes_in_expression.append(line.strip())

        # renormalizing expressions if different genes
        self.cells_expr = renormalize_tpm(cells_expr, self.genes_in_expression).astype(
            self.FP_TYPE
        )
        # renormalizing tumor expressions
        self.tumor_expr = renormalize_tpm(tumor_expr, self.genes_in_expression).astype(
            self.FP_TYPE
        )

        self.tumor_annot = tumor_annot
        self.tumor_mean = self.FP_TYPE(tumor_mean)
        self.tumor_sd = self.FP_TYPE(tumor_sd)
        self.distribution = distribution
        self.hyperexpression_fraction = self.FP_TYPE(hyperexpression_fraction)
        self.max_hyperexpr_level = self.FP_TYPE(max_hyperexpr_level)
        self.random_seed = random_seed
        self.rng = np.random.default_rng(self.random_seed)

    def generate(
        self,
        target_gene: str,
        genes: Union[List[str], None] = None,
        make_noise: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate data for tumor profile model training.

        Args:
            target_gene (str): The name of the target gene.
            genes (Union[List[str], None], optional): Genes to use for data generation. Defaults to None.
            If none, genes from config will be taken.
            make_noise (bool, optional): Whether noise should be added to the data. Defaults to True.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The data with structure:
            (mixed_cells_expr, microenv * (1 - fractions.loc["Tumor"]), fractions, tumor)
        """
        if not genes:
            genes = self.config["Model"]["Genes"]

        fractions, tumor, microenv, _ = self.generate_components(genes=genes)
        tumor = self.add_tumor_hyperexpression(
            tumor,
            target_gene,
            hyperexpression_fraction=self.hyperexpression_fraction,
            max_hyperexpr_level=self.max_hyperexpr_level,
            randomizer=self.rng,
            seed_val=self.random_seed,
        )

        mixed_cells_expr = tumor * fractions.loc["Tumor"] + microenv * (
            1 - fractions.loc["Tumor"]
        )

        if make_noise:
            mixed_cells_expr = self.make_noise(mixed_cells_expr, target_gene)
        microenv =  microenv * (1 - fractions.loc["Tumor"])
        return (
            mixed_cells_expr,
            microenv,
            fractions,
            tumor
        )

    def generate_components(
        self, genes: Union[List[str], None] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate fractions tumor and micro environment components.

        Args:
            genes (Union[List[str], None], optional): Genes to use for data generation. Defaults to None.
            If none, genes from config will be taken.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: The data with structure:
            (fractions, tumor_answer, microenv_answer.divide(1 - self.fractions.loc["Tumor"]))
        """
        if not genes:
            genes = self.config["Model"]["Genes"]

        mixed_cells_expr = pd.DataFrame(
            np.zeros((len(genes), self.num_points)),
            index=genes,
            columns=range(self.num_points),
            dtype=self.FP_TYPE,
        )

        cells_to_mix = self.config["Model"]["Mix"]

        microenv_answer = pd.DataFrame(
            np.zeros((len(genes), self.num_points)),
            index=genes,
            columns=range(self.num_points),
            dtype=float,
        )
        pure_cells = self.generate_pure_cell_expressions(genes, self.num_av, cells_to_mix) 
        pure_tumor = self.generate_tumor_expressions(genes, None, add_tumor_hyperexpression=False)
        average_cells = {
            **pure_cells[0],
            "Tumor": pure_tumor[0],
        }
        
        samples_info = pure_cells[1].copy()
        samples_info.loc['Tumor'] = pure_tumor[1]
        
        
        fractions = self.get_fractions(cells_to_mix)
        for cell in sorted(fractions.index):
            if cell != "Tumor":
                microenv_answer += fractions.loc[cell] * average_cells[cell]

        tumor_answer = average_cells["Tumor"]

        del average_cells
        gc.collect()
        microenv_normalized = microenv_answer.divide(1 - fractions.loc["Tumor"])
        return (
            fractions,
            tumor_answer,
            microenv_normalized,
            samples_info
        )

    def generate_tumor_expressions(
        self,
        genes: Union[List[str], None],
        target_gene: str,
        add_tumor_hyperexpression: bool,
    ) -> pd.DataFrame:
        """Method for tumor expression generation.

        Args:
            genes (Union[List[str], None]): Genes to use for data generation.
            target_gene (str): The name of the target gene.
            add_tumor_hyperexpression (bool): Whether tumor hyperexpression should be added.

        Returns:
            pd.DataFrame: Tumor expression.
        """
        tumor_expr = self.tumor_expr.loc[genes].sample(
            self.num_points, replace=True, axis=1, random_state=self.random_seed
        )
        samples_list = tumor_expr.columns
        tumor_expr.columns = range(self.num_points)
        if add_tumor_hyperexpression:
            tumor_expr = self.add_tumor_hyperexpression(
                tumor_expr,
                target_gene,
                hyperexpression_fraction=self.hyperexpression_fraction,
                max_hyperexpr_level=self.max_hyperexpr_level,
                randomizer=self.rng,
                seed_val=self.random_seed,
            ) 

        return tumor_expr, samples_list

    def dirichlet_mixing(self, cells_to_mix: List[str]) -> pd.DataFrame:
        """Method generates the values of the proportion of mixed cells by dirichlet method.
        The method guarantees a high probability of the the presence of each cell type from 0 to 100%
        at the expense of enrichment of frget_cell_with_subtypesactions close to zero.

        Args:
            cells_to_mix (List[str]): Cell types to mix.

        Returns:
            pd.DataFrame: Generated cell type fractions.
        """
        self.rng = np.random.default_rng(self.random_seed)
        fracs = pd.DataFrame(
            self.rng.dirichlet(
                [1.0 / len(cells_to_mix)] * len(cells_to_mix), size=self.num_points
            )
            .astype(self.FP_TYPE)
            .T,
            index=cells_to_mix,
            columns=range(self.num_points),
        )

        return fracs

    @staticmethod
    def add_tumor_hyperexpression(
        data: pd.DataFrame,
        target_gene: str,
        hyperexpression_fraction: np.ndarray,
        max_hyperexpr_level: float,
        randomizer: np.random,
        seed_val: int,
    ) -> pd.DataFrame:
        """Method for tumor hyperexpression addition.

        Args:
            data (pd.DataFrame): Data for which tumor hyperexpression should be added.
            target_gene (str):  The name of the target gene.
            hyperexpression_fraction (np.ndarray): The fraction of hyperexpressed genes.
            max_hyperexpr_level (float): The max hyperexpression value
            randomizer (np.random): The function for random vectors generation
            seed_val (int): Random seed value.

        Returns:
            pd.DataFrame: The data with added hyperexpression.
        """
        randomizer = np.random.default_rng(seed_val)
        tumor_noise = randomizer.random(size=data.shape)
        tumor_noise = np.where(
            tumor_noise < hyperexpression_fraction, max_hyperexpr_level, 0
        )
        randomizer = np.random.default_rng(seed_val)
        tumor_noise = np.float32(tumor_noise * randomizer.random(size=data.shape))
        tumor_noise = pd.DataFrame(tumor_noise, index=data.index, columns=data.columns)
        tumor_noise.loc[target_gene] = 0
        data = data + tumor_noise

        return data

    # @staticmethod
    def rebalance_samples_by_type(self, annot: pd.DataFrame, k: float) -> pd.Index:
        """Function rebalances the annotation dataset: rare types (type is based on column 'col')
        appears more often due to the multiplication of their samples in the dataset.
        All NaN samples will be deleted.

        k == 0: no rebalance
        k == 1: number of samples of each type in 'col' increases to maximum
        0 < k < 1: rebalance based on 'func'

        Args:
            annot (pd.DataFrame): Annotation dataframe (samples as indices).
            k (float): rebalance parameter 0 < k < 1.

        Returns:
            pd.Index: The list of samples as pd.Index.
        """
        type_counter = annot["Dataset"].value_counts()

        def func(x, k):
            return x ** (1 - k)

        max_counter = type_counter.max()
        type_counter = np.round(
            func(x=type_counter / max_counter, k=k) * max_counter
        ).astype(int)

        order = []

        for idx in sorted(set(type_counter))[::-1]:
            samples = type_counter[type_counter == idx].index
            order += sorted(samples)

        type_counter = type_counter.loc[order]
        samples = []
        for t, counter in type_counter.items():
            self.rng = np.random.default_rng(self.random_seed)

            a = self.rng.choice(annot.loc[annot["Dataset"] == t].index, counter)
            samples.extend(a)

        return pd.Index(samples)

    def generate_pure_cell_expressions(
        self, genes: List[str], num_av: int, cells_to_mix: List[str]
    ) -> Dict[str, float]:
        """Function makes averaged samples of random cellular samples, taking into account the nested structure
        of the subtypes and the desired proportions of the subtypes for cell type.

        Args:
            genes (List[str]): Genes for data generation.
            num_av (int): Number of random samples of cell type that will be averaged to form the resulting sample.
            cells_to_mix (List[str]): List of cell types for which averaged samples from random selection will be formed.

        Returns:
            Dict[str, float]: Dictionary with matrix of average of random num_av samples for each cell type with replacement.
        """
        average_cells = {}
        cells_expr = self.cells_expr.loc[genes]
        samples_list = pd.DataFrame([], index = cells_to_mix, columns=range(self.num_points))
        
        for cell in cells_to_mix:
            selected_types = self.get_cell_with_subtypes(cell)

            cells_selection = get_samples(self.cells_annot["Cell_type"], selected_types)
            expressions_matrix = pd.DataFrame(
                np.zeros((len(cells_expr.index), self.num_points), dtype=self.FP_TYPE),
                index=cells_expr.index,
                columns=range(self.num_points),
            )
            
            for i in range(num_av):
                if self.rebalance_param is not None:
                    cells_index = pd.Index(
                        self.rebalance_samples_by_type(
                            self.cells_annot.loc[cells_selection.index],
                            k=self.rebalance_param
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
               
                self.rng = np.random.default_rng(i+self.random_seed)
                samples = self.rng.choice(cells_index, self.num_points)
                
                expressions_matrix += cells_expr.loc[:, samples].values
                if len(samples_list.loc[cell].dropna()) == 0:
                    samples_list.loc[cell] = samples
                else:
                
                    samples_list.loc[cell] += [',' for _ in range(len(samples))] + samples
                
            average_cells[cell] = expressions_matrix / self.FP_TYPE(num_av)
           
           
        return average_cells, samples_list

    def change_subtype_proportions(self, cell: str, cells_index: pd.Index) -> pd.Index:
        """Function changes the proportions of the cell subtypes when they are considered as types for random selection.
        The proportions of the subtypes will be changed including samples of deeper subtypes.

        Args:
            cell (str): Cell type for which the proportions of the subtypes will be changed.
            cells_index (pd.Index): Samples for cell type.

        Returns:
            pd.Index: Array of sample indexes oversampled for needed proportions.
        """
        cell_subtypes = self.cell_types.get_direct_subtypes(cell)
        specified_subtypes = set(self.proportions.dropna().index).intersection(
            cell_subtypes
        )

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
            self.rng = np.random.default_rng(self.random_seed)
            oversampled_subtypes[subtype] = self.rng.choice(
                subtype_samples[subtype],
                int(subtype_proportions[cell][subtype] * max_size / min_num + 1),
            )
            result_samples = np.concatenate(
                (result_samples, oversampled_subtypes[subtype])
            )

        return result_samples

    def normal_cell_distribution(self, sd: float = 0.5, mean: float = 0.5) -> pd.Series:
        """Generates vector with normal distribution truncated on [0,1] for cell mixing.

        Args:
            sd (float, optional): Standard deviation. Defaults to 0.5.
            mean (float, optional): Mean. Defaults to 0.5.

        Returns:
            pd.Series: Array with samples from given normal distribution.
        """
        mean, sd = self.FP_TYPE(mean), self.FP_TYPE(sd)
        self.rng = np.random.default_rng(self.random_seed)
        values = sd * self.rng.randn(self.num_points).astype(self.FP_TYPE) + mean
        self.rng = np.random.default_rng(self.random_seed)
        values[values < 0] = self.rng.uniform(size=len(values[values < 0])).astype(
            self.FP_TYPE
        )
        self.rng = np.random.default_rng(self.random_seed)
        values[values > 1] = self.rng.uniform(size=len(values[values > 1])).astype(
            self.FP_TYPE
        )

        return pd.Series(values, index=range(self.num_points))

    def uniform_cell_distribution(self) -> pd.Series:
        """Generates vector with normal distribution truncated on [0,1] for cell mixing.

        Returns:
            pd.Series: Array with samples from given uniform distribution.
        """
        self.rng = np.random.default_rng(self.random_seed)
        values = self.rng.uniform(size=self.num_points).astype(self.FP_TYPE)
        return pd.Series(values, index=range(self.num_points))

    def exponential_with_uniform_distribution(self) -> pd.Series:
        """Generates vector with of mixed uniform and exponential distrtribution
        truncated on [0,1] for cell mixing.

        Returns:
            pd.Series: Array with samples from given mixed exponential and uniform distribution.
        """
        self.rng = np.random.default_rng(self.random_seed)
        x = self.rng.exponential(size=self.num_points // 2 + (self.num_points % 2))
        x = x / max(x)
        self.rng = np.random.default_rng(self.random_seed)
        x = np.concatenate([x, self.rng.uniform(size=self.num_points // 2)])
        self.rng = np.random.default_rng(self.random_seed)
        self.rng.shuffle(x)
        values = x.astype(self.FP_TYPE)

        return pd.Series(values, index=range(self.num_points))

    def make_noise(
        self,
        data: pd.DataFrame,
        gene: Union[str, None] = None,
        poisson_noise_level: float = 0.5,
        uniform_noise_level: float = 0,
    ) -> pd.DataFrame:
        """Method adds Poisson noise (very close approximation) and uniform noise for expressions in TPM.
        Uniform noise - proportional to gene expressions noise from a normal distribution.

        Args:
            data (pd.DataFrame): Dataframe with expressions in TPM with genes as indexes.
            gene (Union[str, None], optional): Gene for which noise should be added. Defaults to None.
            poisson_noise_level (float, optional): Poisson noise level. Defaults to 0.5.
            uniform_noise_level (float, optional): Uniform noise level. Defaults to 0.

        Returns:
            pd.DataFrame: Dataframe with added noise.
        """
        self.rng = np.random.default_rng(self.random_seed)

        X = data.to_numpy(dtype=self.FP_TYPE)
        length_normed_data = (X * 1000) / np.array(self.gene_length.loc[data.index, "length"])[:, None].astype(self.FP_TYPE)

        self.rng = np.random.default_rng(self.random_seed)
        norms = self.rng.normal(size=X.shape).astype(self.FP_TYPE)
        noise = np.sqrt(poisson_noise_level * length_normed_data) * norms
    
        if uniform_noise_level:
             noise += X * self.rng.normal(size=X.shape, scale=uniform_noise_level)
    
        noise = pd.DataFrame(noise, index=data.index, columns=data.columns, dtype=self.FP_TYPE)
        noise.loc[gene] = 0
        X += noise
        x = np.clip(X, a_min=0, a_max=None)

        return pd.DataFrame(x, index=data.index, columns=data.columns)
                                                     
                                                     
    def select_datasets_fraction(
        self, annotation: pd.DataFrame, selector_col: str, bootstrap_fraction: float
    ) -> List[str]:
        """Function selects random datasets for every cell name (without nested subtypes) without replacement.

        Args:
            annotation (pd.DataFrame): Dataframe with colum 'Dataset' and samples as index.
            selector_col (str): Column name for datasets selection.
            bootstrap_fraction (float): Fraction of datasets to select.

        Returns:
            List[str]: List of sample indexes for selected datasets.
        """
        selected_samples = []
        values = annotation[selector_col].unique()
        for value in values:
            value_inds = annotation.loc[annotation[selector_col] == value].index
            value_inds = value_inds.difference(selected_samples)
            cell_datasets = set(annotation.loc[value_inds, "Dataset"].unique())
            if cell_datasets:
                self.rng = np.random.default_rng(self.random_seed)
                selected_datasets = self.rng.choice(
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

    def get_cell_with_subtypes(self, cell: str) -> List[str]:
        """Function for cell and its subtypes selection.

        Args:
            cell (str): The cell type name.

        Returns:
            List[str]: The list of given cell and its subtypes.
        """
        types_structure = self.config["Types_structure"].copy()
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

    def get_fractions(self, cells_to_mix: List[str]) -> pd.DataFrame:
        """Randomly generate cell composition values for modeled cell, cells to mix and tumor portion.
        Adjust them to 1. These fractions can be used to generate mixes without any readjustments.

        Args:
            cells_to_mix (List[str]): The list of cells to mix with the modeled cell.

        Returns:
            pd.DataFrame: Dataframe, columns - mix number, rows - cell fractions.
        """
        cells_to_mix_values = self.dirichlet_mixing(cells_to_mix)
        tumor_values = self.uniform_cell_distribution()
        cells_fractions = cells_to_mix_values * (1 - tumor_values)
        cells_fractions.loc["Tumor"] = tumor_values

        return cells_fractions

    def save(self, path: Union[str, Path]) -> None:
        """Method for saving generated data.

        Args:
            path (Union[str, Path]): Path to saved data.
        """
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
