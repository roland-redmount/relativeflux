import numpy as np
import pandas as pd
from lmfit import Parameters
from scipy.integrate import odeint

from simpleflux.model import FluxState, FluxModel


class ModelState:
    """
    A non-stationary model state is specified by
    a FluxModel, FluxState concentrations and medium labeling state
    """

    model: FluxModel
    flux_state: FluxState
    concentrations: np.array
    medium_influx_mi: np.array      # move this somewhere else?

    influx_matrix: np.array
    medium_influx_matrix: np.array
    outflux_vector: np.array

    def __init__(self, model: FluxModel, flux_state: FluxState,
                 concentrations: np.array, medium_mi: dict[str, float]):
        self.model = model
        # vector of medium substrate heavy fraction for each reaction
        self.medium_influx_mi = np.zeros(len(model.reactions))
        for reaction, mi in medium_mi.items():
            index = model.reactions.index(reaction)
            self.medium_influx_mi[index] = mi
        self.update(flux_state, concentrations)

    def update(self, flux_state: FluxState, concentrations: np.array) -> None:
        """
        Update this model state to new fluxes and concentrations
        """
        self.flux_state = flux_state
        self.concentrations = concentrations
        self._compute_matrices()

    def _compute_matrices(self) -> None:
        self.influx_matrix = self._influx_matrix()
        self.medium_influx_matrix = self._medium_influx_matrix()
        self.outflux_vector = self._outflux_vector()

    def _influx_matrix(self) -> np.array:
        fwd_flux_matrix = self.flux_state.forward_fluxes[None, :]
        rev_flux_matrix = self.flux_state.reverse_fluxes[None, :]
        return (
                (self.model.stoch_pos * fwd_flux_matrix) @ self.model.stoch_neg.T
                + (self.model.stoch_neg * rev_flux_matrix) @ self.model.stoch_pos.T
        )

    def _medium_influx_matrix(self) -> np.array:
        fwd_flux_matrix = self.flux_state.forward_fluxes[None, :]
        return (self.model.stoch_pos * fwd_flux_matrix) @ self.medium_influx_mi

    def _outflux_vector(self) -> np.array:
        fwd_fluxes = self.flux_state.forward_fluxes
        rev_fluxes = self.flux_state.reverse_fluxes
        return (self.model.stoch_neg @ fwd_fluxes) + (self.model.stoch_pos @ rev_fluxes)

    def derivatives(self, heavy_fractions: np.array, t) -> np.array:
        """
        The derivatives dx/dt for all heavy fractions x at the current state x(t)
        """
        return (
                (self.influx_matrix @ heavy_fractions)
                - (self.outflux_vector * heavy_fractions)
                + self.medium_influx_matrix
        ) / self.concentrations

    def simulate(self, time_points: np.array) -> np.array:
        # add zero data point, expected by odeint()
        # NOTE: what happens if time_points already has a zero?
        time_points_with_zero = np.concatenate((np.array([0]), time_points))
        initial_mi = np.zeros(len(self.model.metabolites))
        # here odeint() calls `derivatives(y, t, *args)`
        simulated_mi_with_zero = odeint(
            func=self.derivatives,
            y0=initial_mi,
            t=time_points_with_zero,
        )
        # remove zero datapoint from result

        return simulated_mi_with_zero[1:, :]

    def simulate_to_pandas(self, time_points: np.array) -> pd.DataFrame:
        return pd.DataFrame(
            self.simulate(time_points),
            columns=self.model.metabolites,
            index=time_points
        )

