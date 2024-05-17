from collections import defaultdict

import numpy

from simpleflux.model import FluxModel, FluxState, Experiment, ModelState
from pytest import fixture
import numpy as np


TEST_STOICHIOMETRY = defaultdict(int, {
    ('MET_IN', 'met'): 1,
    ('PROT_SYNTH', 'met'): -1, ('PROT_SYNTH', 'metp'): 1,
    ('AHCY', 'sah'): -1, ('AHCY', 'hcys'): 1,
    ('METS', 'hcys'): -1, ('METS', 'met'): 1,
    ('MAT', 'met'): -1, ('MAT', 'sam'): 1,
    ('SAM_METH', 'sam'): -1, ('SAM_METH', 'sah'): 1,
    ('CYSTS', 'hcys'): -1, ('CYSTS', 'cyst'): 1,
    ('CYSTL', 'cyst'): -1, ('CYSTL', 'akb'): 1,
    ('HCYS_OUT', 'hcys'): -1,
    ('PROT_OUT', 'metp'): -1,
    ('AKB_OUT', 'akb'): -1,
})
TEST_REVERSIBLE = [
    'MET_IN',
    'PROT_SYNTH',
    'AHCY',
    'CYSTL',
]
TEST_FREE_REACTIONS = ['CYSTS', 'METS', 'PROT_OUT', 'SAM_METH']


@fixture
def example_model():
    return FluxModel(
        stoichiometry=TEST_STOICHIOMETRY,
        reversible_reactions=TEST_REVERSIBLE,
        free_reactions=TEST_FREE_REACTIONS,
    )


@fixture
def example_flux_state_1(example_model):
    return FluxState(
        model=example_model,
        free_fluxes=[10, 80, 30, 120],
        exchanges=[0.9, 0.4, 0.2, 0.8]
    )


@fixture
def example_flux_state_2(example_model):
    return FluxState.from_dict(
        model=example_model,
        free_fluxes={
            'CYSTS' : 1.0,
            'METS': 1.0,
            'SAM_METH': 2.0,
            'PROT_OUT': 3.0,
        },
        exchanges={
            'MET_IN' : 0.0,
            'PROT_SYNTH': 0.0,
            'AHCY': 0.0,
            'CYSTL': 0.0,
        }
    )


@fixture
def example_experiment(example_model):
    return Experiment(
        model=example_model,
        medium_mi={'MET_IN' : 0.99}
    )


def test_stoichiometry(example_model):
    assert example_model.reactions == [
        'AHCY',
        'AKB_OUT',
        'CYSTL',
        'CYSTS',
        'HCYS_OUT',
        'MAT',
        'METS',
        'MET_IN',
        'PROT_OUT',
        'PROT_SYNTH',
        'SAM_METH',
    ]
    assert example_model.metabolites == [
        'akb',
        'cyst',
        'hcys',
        'met',
        'metp',
        'sah',
        'sam',
    ]
    assert example_model.reversible_index == [0, 2, 7, 9]
    assert example_model.free_index == [3, 6, 8, 10]
    assert example_model.dep_index == [0, 1, 2, 4, 5, 7, 9]
    assert example_model.get_exchange_names() == ['AHCY_EX', 'CYSTL_EX', 'MET_IN_EX', 'PROT_SYNTH_EX']


def test_flux_state_1(example_model, example_flux_state_1):
    assert (example_flux_state_1.net_fluxes == [120, 10, 10, 10, 30, 120, 80, 70, 30, 30, 120]).all()
    assert np.allclose(
        example_flux_state_1.forward_fluxes,
        [1200, 10, 50/3, 10, 30, 120, 80, 87.5, 30, 150, 120]
    )
    assert np.allclose(
        example_flux_state_1.reverse_fluxes,
        [1080, 0, 20/3, 0, 0, 0, 0, 17.5, 0, 120, 0]
    )
    assert example_model.is_balanced(example_flux_state_1.net_fluxes)


def test_flux_state_2(example_model, example_flux_state_2):
    assert np.allclose(
        example_flux_state_2.net_fluxes,
        [2, 1, 1, 1, 0, 2, 1, 4, 3, 3, 2]
    )
    assert np.allclose(
        example_flux_state_2.forward_fluxes,
        example_flux_state_2.net_fluxes
    )
    assert np.allclose(
        example_flux_state_2.reverse_fluxes,
        0
    )
    assert example_model.is_balanced(example_flux_state_2.net_fluxes)


def test_experiment(example_experiment):
    assert (example_experiment.medium_influx_mi == [0, 0, 0, 0, 0, 0, 0, 0.99, 0, 0, 0]).all()


def test_simulate_1(example_experiment, example_flux_state_1):
    concentrations = [1000, 30, 10, 500, 12_000, 15, 65]
    time_points = np.arange(0, 3, 0.5)
    simulated_mi = example_experiment.simulate(
        state=ModelState(
            flux_state=example_flux_state_1,
            concentrations=concentrations
        ),
        time_points=time_points
    )
    assert simulated_mi.shape == (len(time_points), len(example_experiment.model.metabolites))
    assert np.allclose(simulated_mi[0], 0)
    assert (simulated_mi >= 0).all()
    assert (simulated_mi <= 1).all()


def test_simulate_2(example_experiment, example_flux_state_2):
    concentrations = [1000, 30, 10, 500, 12_000, 15, 65]
    time_points = np.arange(0, 3, 0.5)
    simulated_mi = example_experiment.simulate(
        state=ModelState(
            flux_state=example_flux_state_2,
            concentrations=concentrations
        ),
        time_points=time_points
    )
    assert simulated_mi.shape == (len(time_points), len(example_experiment.model.metabolites))
    assert np.allclose(simulated_mi[0], 0)
    assert (simulated_mi >= 0).all()
    assert (simulated_mi <= 1).all()
