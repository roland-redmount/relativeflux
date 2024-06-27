import numpy as np
import pandas as pd
from scipy.integrate import odeint

from relativeflux.model import RelativeFluxState, RelativeFluxModel


class ModelState:
    """
    A non-stationary model state is specified by
    a RelativeFluxModel, a RelativeFluxState and labeling state for input metabolites
    """

    model: RelativeFluxModel
    flux_state: RelativeFluxState
    medium_mi: np.array

    diff_matrix: np.array
    input_vector: np.array

    def __init__(self, model: RelativeFluxModel, flux_state: RelativeFluxState,
                 medium_mi: dict[str, float]):
        self.model = model

        incidence_matrix = model.get_incidence_matrix()
        self.diff_matrix = (
            incidence_matrix[model.internal_index][:,  model.internal_index]
            - np.eye(len(model.reactions))
        )

        # vector of medium substrate heavy fraction for each input metabolite
        self.medium_mi = np.zeros(len(model.input_index))
        for i, mi in enumerate(model.input_index):
            self.medium_mi[i] = medium_mi[model.metabolites[mi]]
        # vector of contribution from inputs to internal metabolites
        # print(incidence_matrix[model.internal_index, model.input_index])
        # print(self.medium_mi)
        self.input_vector = incidence_matrix[model.internal_index][:, model.input_index] @ self.medium_mi
        self.update(flux_state)

    def update(self, flux_state: RelativeFluxState) -> None:
        self.flux_state = flux_state

    def derivatives(self, heavy_fractions: np.array, t=0.0) -> np.array:
        """
        The derivatives dx/dt for all heavy fractions x at the current state x(t)
        """
        return (
            (self.diff_matrix @ heavy_fractions) + self.input_vector
        ) * self.flux_state.turnover_rates

    def simulate(self, time_points: np.array) -> np.array:
        # add zero data point, expected by odeint()
        # NOTE: what happens if time_points already has a zero?
        time_points_with_zero = np.concatenate((np.array([0]), time_points))
        initial_mi = np.zeros(len(self.model.internal_index))
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
            index=time_points,
            columns=self.model.internal_metabolites()
        )
