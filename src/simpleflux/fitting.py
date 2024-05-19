import pandas as pd
from lmfit import Parameters
from lmfit.minimizer import Minimizer, MinimizerResult
import numpy as np
from simpleflux.model import FluxState
from simpleflux.modelstate import ModelState


# barrier function to be used for for constraining to positive fluxes,
# not used now. Barrier does not strictly prevent evaluating the objective
# at negative fluxes, inlt makes it less likely
def barrier_function(x: np.array, epsilon: float, a: float, b: float) -> float:
    """
    A smooth function f(x) such that f(epsilon) = a > f(0) = b
    and f(x) is linear for x < 0 with f'(x) = f'(b)
    """
    if x > 0:
        return np.power(b/a, 1 - x/epsilon).sum()
    else:
        return b + a * np.log(b/a) * x


def _list_index(x: list, y: list):
    return [x.index(t) for t in y]

class ModelFit:

    LOWER_FLUX_BOUND = 0.01
    MAX_CONC_RATIO = 10
    MAX_EXCHANGE_BOUND = 0.999

    model_state: ModelState

    # TODO: harmonize these names
    t_measured: np.array
    x_measured: np.array
    x_std_dev: np.array
    x_index_to_fit: list
    flux_measured: np.array
    flux_std_dev: np.array
    flux_index_to_fit: list
    pool_sizes_measured: np.array
    pool_sizes_std_dev: np.array
    conc_index_to_fit: list

    minimizer_result: MinimizerResult

    def __init__(self, initial_state: ModelState,
                 measured_mi: pd.DataFrame, measured_mi_std_dev: pd.DataFrame,
                 measured_fluxes: pd.DataFrame,
                 measured_conc: pd.DataFrame):
        """
        :param initial_state: initial model state
        :param measured_mi: a data frame index by time points and
        with measured metabolites in columns
        :param measured_fluxes: a data frame indexed by reactions
        :param measured_conc:
        """
        self.model_state = initial_state
        self.t_measured = measured_mi.index.to_numpy()

        self.x_index_to_fit = _list_index(
            self.model_state.model.metabolites,
            measured_mi.columns
        )
        self.x_measured = measured_mi.to_numpy()
        self.x_std_dev = measured_mi_std_dev.to_numpy()

        self.flux_index_to_fit = _list_index(
            self.model_state.model.reactions,
            measured_fluxes.index
        )
        self.flux_measured = measured_fluxes['mean'].to_numpy()
        self.flux_std_dev = measured_fluxes['std_dev'].to_numpy()

        self.conc_index_to_fit = _list_index(
            self.model_state.model.metabolites,
            measured_conc.index
        )
        self.pool_sizes_measured = measured_conc['mean'].to_numpy()
        self.pool_sizes_std_dev = measured_conc['std_dev'].to_numpy()

    def _parameters_to_flux(self, parameters: Parameters) -> FluxState:
        free_fluxes = np.array([
            parameters[reaction].value
            for reaction in self.model_state.model.free_reactions()
        ])
        exchange = np.array([
            parameters[exchange].value
            for exchange in self.model_state.model.exchange_names()
        ])
        return FluxState(
            model=self.model_state.model,
            free_fluxes=free_fluxes,
            exchanges=exchange
        )

    def _parameters_to_concentrations(self, parameters: Parameters) -> np.array:
        return np.array([
            parameters[metabolite].value
            for metabolite in self.model_state.model.metabolites
        ])

    def update_state(self, parameters: Parameters) -> None:
        self.model_state.update(
            flux_state=self._parameters_to_flux(parameters),
            concentrations=self._parameters_to_concentrations(parameters),
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
            self.flux_residual().flatten(),
            self.pool_size_residual().flatten()
        ])

    def isotope_residual(self) -> np.array:
        x_simulated = self.model_state.simulate(self.t_measured)
        return (self.x_measured - x_simulated[:, self.x_index_to_fit]) / self.x_std_dev

    def flux_residual(self) -> np.array:
        current_fluxes = self.model_state.flux_state.net_fluxes[self.flux_index_to_fit]
        return (self.flux_measured - current_fluxes) / self.flux_std_dev

    def pool_size_residual(self) -> np.array:
        current_pool_size = self.model_state.concentrations[self.conc_index_to_fit]
        return (self.pool_sizes_measured - current_pool_size) / self.pool_sizes_std_dev

    @classmethod
    def state_to_parameters(cls, state: ModelState, flux_bound=1000) -> Parameters:
        parameters = Parameters()
        for i, reaction in enumerate(state.model.free_reactions()):
            parameters.add(
                name=reaction,
                value=state.flux_state.free_fluxes[i],
                min=(-flux_bound if state.model.is_reversible(reaction) else cls.LOWER_FLUX_BOUND),
                max=flux_bound
            )
        for i, exchange_name in enumerate(state.model.exchange_names()):
            parameters.add(
                name=exchange_name,
                value=state.flux_state.exchanges[i],
                min=0,
                max=cls.MAX_EXCHANGE_BOUND
            )
        for i, exchange_name in enumerate(state.model.metabolites):
            parameters.add(
                exchange_name,
                value=state.concentrations[i],
                min=state.concentrations[i] / cls.MAX_CONC_RATIO,
                max=state.concentrations[i] * cls.MAX_CONC_RATIO
            )
        return parameters

    def fit(self) -> None:
        # here minimize calls userfcn(params, *args, **kws)
        minimizer = Minimizer(
            userfcn=self.update_and_compute_residual,
            params=self.state_to_parameters(self.model_state),
        )
        self.minimizer_result = minimizer.minimize(method='leastsq')

    def _dep_flux_cov(self, free_flux_cov: np.array) -> np.array:
        # project covariance matrix for X onto Y = AX as Cov(Y) = A Cov(X) A^t
        return (
            self.model_state.model.dep_flux_matrix @ free_flux_cov @
            self.model_state.model.dep_flux_matrix.T
        )

    def fitted_fluxes_std_err(self) -> np.array:
        n_reactions = len(self.model_state.model.reactions)
        n_free_fluxes = len(self.model_state.model.free_index)
        # NOTE: this assumes the free fluxes are placed first
        free_flux_cov = self.minimizer_result.covar[:n_free_fluxes, :n_free_fluxes]
        dep_flux_cov = self._dep_flux_cov(free_flux_cov)
        net_flux_std_err = np.zeros(n_reactions)
        net_flux_std_err[self.model_state.model.free_index] = np.sqrt(free_flux_cov.diagonal())
        net_flux_std_err[self.model_state.model.dep_index] = np.sqrt(dep_flux_cov.diagonal())
        return net_flux_std_err
