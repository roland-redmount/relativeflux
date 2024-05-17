from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.integrate import odeint


def _negate_indices(indices: list[int], n: int) -> list[int]:
    return [
        index
        for index in range(n)
        if index not in indices
    ]


class FluxState:
    """
    A flux state for a given model, parameterized by free fluxes
    and exchange coefficients for reversible reactions
    """
    free_fluxes: np.array
    exchanges: np.array
    net_fluxes: np.array
    forward_fluxes: np.array
    reverse_fluxes: np.array

    def __init__(self, model: 'FluxModel', free_fluxes: np.array, exchanges: np.array):
        self.free_fluxes = free_fluxes
        self.exchanges = exchanges

        dep_fluxes = model.dep_flux_matrix @ free_fluxes
        self.net_fluxes = np.zeros(len(model.reactions))
        self.net_fluxes[model.free_index] = free_fluxes
        self.net_fluxes[model.dep_index] = dep_fluxes
        if (self.net_fluxes[model.irreversible_index] < 0).any():
            raise ValueError('Negative fluxes for irreversible reactions')
        if not model.is_balanced(self.net_fluxes):
            raise ValueError('Flux vector is not balanced')

        exchanges_all = np.zeros(len(model.reactions))
        exchanges_all[model.reversible_index] = exchanges
        self.reverse_fluxes = self.net_fluxes * exchanges_all / (1 - exchanges_all)
        self.forward_fluxes = self.net_fluxes + self.reverse_fluxes
        if (self.forward_fluxes < 0).any():
            raise ValueError('Negative forward_fluxes')
        if (self.reverse_fluxes < 0).any():
            raise ValueError('Negative reverse_fluxes')


    @staticmethod
    def from_dict(model: 'FluxModel', free_fluxes: dict[str, float], exchanges: dict[str, float]):
        return FluxState(
            model=model,
            free_fluxes=[free_fluxes[model.reactions[i]] for i in model.free_index],
            exchanges=[exchanges[model.reactions[i]] for i in model.reversible_index]
        )


@dataclass
class ModelState:
    """
    A model state is define by fluxes and concentrations
    """
    flux_state: FluxState
    concentrations: np.array


class FluxModel:
    reactions: list[str]
    reversible_index: list[int]
    irreversible_index: list[int]
    free_index: list[int]
    dep_index: list[int]
    metabolites: list[str]
    # stochiometry matrix, includingg non-balanced metabolites
    stoch_matrix: np.array
    stoch_pos: np.array
    stoch_neg: np.array

    # the flux model now also includes the free flux parametrization, that may be too much
    def __init__(self, stoichiometry: dict, reversible_reactions: list[str], free_reactions: list[str]):
        self.reactions = sorted(list(
            {reaction for reaction, _ in stoichiometry.keys()}
        ))
        self.reversible_index = [
            self.reactions.index(reaction)
            for reaction in sorted(reversible_reactions)
        ]
        self.irreversible_index = _negate_indices(self.reversible_index, len(self.reactions))
        self.metabolites = sorted(list(
            {metabolite for _, metabolite in stoichiometry.keys()}
        ))
        self.stoch_matrix = np.array(
            [
                [
                    stoichiometry[reaction, metabolite]
                    for reaction in self.reactions
                ]
                for metabolite in self.metabolites
            ]
        )
        self.stoch_pos = np.clip(self.stoch_matrix, 0, 1)
        self.stoch_neg = -np.clip(self.stoch_matrix, -1, 0)

        # free reactions
        deg_freedom = len(self.reactions) - np.linalg.matrix_rank(self.stoch_matrix)
        if deg_freedom != len(free_reactions):
            raise ValueError('Matrix rank disagrees with free reactions')
        self.free_index = [
            self.reactions.index(reaction)
            for reaction in sorted(free_reactions)
        ]
        self.dep_index = _negate_indices(self.free_index, len(self.reactions))
        stoch_dep = self.stoch_matrix[:, self.dep_index]
        stoch_free = self.stoch_matrix[:, self.free_index]
        if np.linalg.matrix_rank(stoch_dep) != len(self.dep_index):
            raise ValueError('Matrix rank disagrees with dependent reactions')
        # matrix for computing dependent fluxes
        self.dep_flux_matrix = -np.linalg.inv(stoch_dep) @ stoch_free

    def get_stoichiometry_df(self):
        return pd.DataFrame(
            self.stoch_matrix,
            columns=self.reactions, index=self.metabolites
        )

    def get_exchange_names(self):
        return [self.reactions[i] + '_EX' for i in self.reversible_index]

    def influx_matrix(self, flux: FluxState) -> np.array:
        return (
                (self.stoch_pos * flux.forward_fluxes[None, :]) @ self.stoch_neg.T
                + (self.stoch_neg * flux.reverse_fluxes[None, :]) @ self.stoch_pos.T
        )

    def medium_influx_matrix(self, flux: FluxState, medium_influx_mi: np.array) -> np.array:
        return (self.stoch_pos * flux.forward_fluxes[None, :]) @ medium_influx_mi

    def outflux_vector(self, flux: FluxState) -> np.array:
        return (self.stoch_neg @ flux.forward_fluxes) + (self.stoch_pos @ flux.reverse_fluxes)

    @staticmethod
    def derivatives(heavy_fractions: np.array, t,
                    influx_matrix: np.array, medium_influx_matrix: np.array, outflux_vector: np.array,
                    pool_sizes: np.array) -> np.array:
        return (
            (influx_matrix @ heavy_fractions)
            - (outflux_vector * heavy_fractions)
            + medium_influx_matrix
        ) / pool_sizes

    def get_flux_table(self, flux: FluxState) -> pd.DataFrame:
        return pd.DataFrame(
            {
                'net_flux': flux.net_fluxes,
                'fwd_flux': flux.forward_fluxes,
                'rev_flux': flux.reverse_fluxes,
            },
            index=self.reactions
        )

    def mass_balance(self, net_fluxes: np.array) -> np.array:
        return self.stoch_matrix @ net_fluxes

    def is_balanced(self, net_fluxes: np.array) -> bool:
        return np.allclose(
            self.mass_balance(net_fluxes),
            0
        )


class Experiment:

    model: FluxModel
    medium_influx_mi: np.array

    def __init__(self, model: FluxModel, medium_mi: dict[str, float]):
        self.model = model
        self.medium_influx_mi = np.zeros(len(model.reactions))
        for reaction, mi in medium_mi.items():
            index = model.reactions.index(reaction)
            self.medium_influx_mi[index] = mi

    def simulate(self, state: ModelState, time_points: np.array):
        # add zero data point, expected by odeint()
        time_points_with_zero = np.concatenate((np.array([0]), time_points))
        initial_mi = np.zeros(len(state.concentrations))
        # here odeint() calls `derivatives(y, t, *args)`
        simulated_mi_with_zero = odeint(
            func=self.model.derivatives,
            y0=initial_mi,
            t=time_points_with_zero,
            args=(
                self.model.influx_matrix(state.flux_state),
                self.model.medium_influx_matrix(state.flux_state, self.medium_influx_mi),
                self.model.outflux_vector(state.flux_state),
                state.concentrations
            )
        )
        # remove zero datapoint from result
        return simulated_mi_with_zero[1:, :]
