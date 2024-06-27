from collections import defaultdict

import numpy as np
import pandas as pd


def _sublist(x: list, indices: list) -> list:
    return [x[index] for index in indices]


def _negate_indices(indices: list[int], n: int) -> list[int]:
    return [
        index
        for index in range(n)
        if index not in indices
    ]


class RelativeFluxState:
    """
    A model state, parameterized one turnover rates for each non-onput metabolite
    This could just be a numpy.array?
    """
    turnover_rates: np.array

    def __init__(self, model: 'RelativeFluxModel', turnover_rates: np.array):
        self.turnover_rates = turnover_rates

    # TODO: we should have a from_pandas methods as well


class RelativeFluxModel:
    # we might want to turn these into numpy arrays for faster indexing
    reactions: list[str]
    metabolites: list[str]
    # stoichiometry matrix (metabolites x reactions)
    stoch_matrix: np.array
    stoch_pos: np.array
    stoch_neg: np.array
    # index of input and internal metabolites
    input_index: list
    internal_index: list

    def __init__(self, stoichiometry: dict):
        default_stoichiometry = defaultdict(int, stoichiometry)
        self.reactions = sorted(list(
            {reaction for reaction, _ in default_stoichiometry.keys()}
        ))
        self.metabolites = sorted(list(
            {metabolite for _, metabolite in default_stoichiometry.keys()}
        ))
        self.stoch_matrix = np.array(
            [
                [
                    default_stoichiometry[reaction, metabolite]
                    for reaction in self.reactions
                ]
                for metabolite in self.metabolites
            ]
        )
        # verify all stoichiometry coefficients are -1, 0, 1
        unique_coefficients = np.unique(self.stoch_matrix)
        if np.any(unique_coefficients != np.array([-1, 0, 1])):
            raise ValueError(f'Stoichiometry includes non-unit coefficients {unique_coefficients}')

        self.stoch_pos = np.clip(self.stoch_matrix, 0, 1)
        self.stoch_neg = -np.clip(self.stoch_matrix, -1, 0)

        # verify each metabolite has a single producing flux
        n_producers = np.sum(self.stoch_pos, axis=1)
        for i, metabolite in enumerate(self.metabolites):
            if n_producers[i] > 1:
                raise ValueError(f'Metabolite {metabolite} has {n_producers} producing fluxes')

        # find input metabolite(s)
        self.input_index = list((n_producers == 0).nonzero()[0])
        if len(self.input_index) == 0:
            raise ValueError('No input metabolites found')
        self.internal_index = _negate_indices(self.input_index, len(self.metabolites))

    def get_stoichiometry_df(self):
        return pd.DataFrame(
            self.stoch_matrix,
            columns=self.reactions, index=self.metabolites
        )

    def internal_metabolites(self):
        return [self.metabolites[i] for i in self.internal_index]

    def get_relative_flux_table(self, flux: RelativeFluxState) -> pd.DataFrame:
        return pd.DataFrame(
            {
                'turnover_rate': flux.turnover_rates,
            },
            index=self.reactions
        )

    def get_incidence_matrix(self):
        return self.stoch_pos @ self.stoch_neg.T
