import pandas as pd
from lmfit import Parameters
from lmfit.minimizer import Minimizer, MinimizerResult
import numpy as np
from simpleflux.model import FluxState
from simpleflux.modelstate import ModelState


class ModelFit:

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

    def __init__(self, initial_state: ModelState,
                 measured_mi: pd.DataFrame, measured_mi_std_dev: pd.DataFrame,
                 measured_flux: pd.DataFrame, measured_flux_std_dev: pd.DataFrame,
                 measured_conc: np.array, measured_conc_std_dev: np.array):
        """
        :param initial_state: initial model state
        :param measured_mi: a data frame index by time points and
        with measured metabolites in columns
        :param measured_flux: a data frame indexed by reactions
        :param measured_conc:
        """
        self.model_state = initial_state
        self.t_measured = measured_mi.index.to_numpy()

        self.x_index_to_fit = [
            self.model_state.model.metabolites.index(metabolite)
            for metabolite in measured_mi.columns
        ]
        self.x_measured = measured_mi.to_numpy()
        self.x_std_dev = measured_mi_std_dev.to_numpy()

        self.flux_index_to_fit = [
            self.model_state.model.reactions.index(reaction)
            for reaction in measured_flux.index
        ]
        self.flux_measured = measured_flux.to_numpy()
        self.flux_std_dev = measured_flux_std_dev.to_numpy()
        # currently we must specify measurements for all pool sizes
        self.pool_sizes_measured = measured_conc
        self.pool_sizes_std_dev = measured_conc_std_dev

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
        self.update_state(parameters)
        return self.model_residual()

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
        return (self.pool_sizes_measured - self.model_state.concentrations) / self.pool_sizes_std_dev

    def fit(self) -> MinimizerResult:
        # here minimize calls userfcn(params, *args, **kws)
        minimizer = Minimizer(
            userfcn=self.update_and_compute_residual,
            params=self.model_state.to_parameters(),
        )
        return minimizer.minimize(method='leastsq')
