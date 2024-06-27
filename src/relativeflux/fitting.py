import pandas as pd
from lmfit import Parameters
from lmfit.minimizer import Minimizer, MinimizerResult
import numpy as np

from relativeflux.model import RelativeFluxState
from relativeflux.modelstate import ModelState


def _list_index(x: list, y: list):
    return [x.index(t) for t in y]


class ModelFit:

    LOWER_FLUX_BOUND = 0.001

    model_state: ModelState

    # TODO: harmonize these names
    t_measured: np.array
    x_measured: np.array
    x_std_dev: np.array
    x_index_to_fit: list

    minimizer_result: MinimizerResult

    def __init__(self, initial_state: ModelState,
                 measured_mi: pd.DataFrame, measured_mi_std_dev: pd.DataFrame):
        """
        :param initial_state: initial model state
        :param measured_mi: a data frame indexed by time points and
        with measured metabolites in columns
        """
        self.model_state = initial_state
        self.t_measured = measured_mi.index.to_numpy()

        self.x_index_to_fit = _list_index(
            self.model_state.model.internal_metabolites(),
            measured_mi.columns
        )
        self.x_measured = measured_mi.to_numpy()
        self.x_std_dev = measured_mi_std_dev.to_numpy()

    def _parameters_to_turnover(self, parameters: Parameters) -> RelativeFluxState:
        # there is one turnover rate for each reaction
        turnover_rates = np.array([
            parameters[reaction].value
            for reaction in self.model_state.model.reactions
        ])
        return RelativeFluxState(
            model=self.model_state.model,
            turnover_rates=turnover_rates
        )

    def update_state(self, parameters: Parameters) -> None:
        self.model_state.update(
            flux_state=self._parameters_to_turnover(parameters)
        )

    def update_and_compute_residual(self, parameters: Parameters) -> np.array:
        try:
            self.update_state(parameters)
            return self.model_residual()
        except ValueError:
            print(parameters)
            raise

    def model_residual(self) -> np.array:
        return np.concatenate([
            self.isotope_residual().flatten(),
            # we may want to add additional terms to the objective
        ])

    def isotope_residual(self) -> np.array:
        x_simulated = self.model_state.simulate(self.t_measured)
        return (self.x_measured - x_simulated[:, self.x_index_to_fit]) / self.x_std_dev

    @classmethod
    def state_to_parameters(cls, state: ModelState, flux_bound=1000) -> Parameters:
        parameters = Parameters()
        for i, reaction in enumerate(state.model.reactions):
            parameters.add(
                name=reaction,
                value=state.flux_state.turnover_rates[i],
                min=ModelFit.LOWER_FLUX_BOUND,
                max=flux_bound
            )
        return parameters

    def fit(self) -> None:
        # here minimize calls userfcn(params, *args, **kws)
        minimizer = Minimizer(
            userfcn=self.update_and_compute_residual,
            params=self.state_to_parameters(self.model_state),
        )
        self.minimizer_result = minimizer.minimize(method='leastsq')

    def fitted_fluxes_std_err(self) -> np.array:
        turnover_cov = self.minimizer_result.covar
        return np.sqrt(turnover_cov.diagonal())
