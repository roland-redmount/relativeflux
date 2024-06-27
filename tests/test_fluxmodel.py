import numpy as np
import pandas as pd

from relativeflux.model import RelativeFluxModel, RelativeFluxState
from relativeflux.modelstate import ModelState
from relativeflux.fitting import ModelFit

from pytest import fixture


TEST_STOICHIOMETRY = {
    ('MAT', 'met'): -1, ('MAT', 'sam'): 1,
    ('SAMDC_POLYA', 'sam'): -1, ('SAMDC_POLYA', 'mta'): 1,
    ('SAM_METH', 'sam'): -1, ('SAM_METH', 'sah'): 1,
    ('AHCY_CYSTS', 'sah'): -1, ('AHCY_CYSTS', 'cyst'): 1,
}


@fixture
def example_model():
    return RelativeFluxModel(
        stoichiometry=TEST_STOICHIOMETRY
    )


EXAMPLE_TURNOVER_RATES = [0.9, 0.4, 0.2, 0.8]


@fixture
def example_flux_state_1(example_model):
    return RelativeFluxState(
        model=example_model,
        turnover_rates=EXAMPLE_TURNOVER_RATES
    )


@fixture
def example_model_state_1(example_model, example_flux_state_1):
    return ModelState(
        model=example_model,
        flux_state=example_flux_state_1,
        medium_mi={'met': 0.99}
    )


def test_stoichiometry(example_model):
    assert example_model.reactions == [
        'AHCY_CYSTS',
        'MAT',
        'SAMDC_POLYA',
        'SAM_METH',
    ]
    assert example_model.metabolites == [
        'cyst',
        'met',
        'mta',
        'sah',
        'sam',
    ]
    assert example_model.input_index == [1]
    assert example_model.internal_index == [0, 2, 3, 4]
    assert example_model.internal_metabolites() == [
        'cyst',
        'mta',
        'sah',
        'sam',
    ]
    expected_incidence_matrix = np.array(
        [
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0],
        ]
    )
    assert np.all(example_model.get_incidence_matrix() == expected_incidence_matrix)


def test_model_state_1(example_model_state_1):
    assert np.all(example_model_state_1.input_vector == [0, 0, 0, 0.99])
    expected_diff_matrix = np.array(
        [
            [-1, 0, 1, 0],
            [0, -1, 0, 1],
            [0, 0, -1, 1],
            [0, 0, 0, -1],
        ]
    )
    assert np.all(example_model_state_1.diff_matrix == expected_diff_matrix)

    heavy_fractions = np.array([0.03, 0.45, 0.7, 0.83])
    print(example_model_state_1.derivatives(heavy_fractions))

    time_points = np.arange(0, 3, 0.5)
    simulated_mi = example_model_state_1.simulate(
        time_points=time_points
    )
    assert simulated_mi.shape == (len(time_points), len(example_model_state_1.model.internal_index))
    assert np.allclose(simulated_mi[0], 0)
    assert np.all(simulated_mi >= 0)
    assert np.all(simulated_mi <= 1)


def test_model_fit(example_model_state_1):
    time_points = np.arange(0, 3, 0.5)
    mi_data = example_model_state_1.simulate_to_pandas(
        time_points=time_points
    )
    std_dev = 0.03
    mi_data_noisy = mi_data + np.random.normal(0, std_dev, mi_data.shape)
    mi_data_stdev = pd.DataFrame(
        np.ones(mi_data.shape) * std_dev,
        index=time_points,
        columns=example_model_state_1.model.internal_metabolites()
    )

    initial_state = ModelState(
        model=example_model_state_1.model,
        flux_state=RelativeFluxState(
            example_model_state_1.model, [0.6, 0.2, 0.5, 0.5]),
        medium_mi={'met': 0.99}
    )

    fitter = ModelFit(initial_state, mi_data_noisy, mi_data_stdev)
    assert fitter.x_index_to_fit == [0, 1, 2, 3]

    fitter.isotope_residual()


# def test_parameters(example_model_state_1):
#     pprint(ModelFit.state_to_parameters(example_model_state_1))
