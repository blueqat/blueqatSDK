import random

import pytest
import torch

from blueqat import BlueqatGlobalSetting

DEFAULT_BACKEND = BlueqatGlobalSetting.get_default_backend_name()


def pytest_addoption(parser):
    parser.addoption('--add-backend', default=[DEFAULT_BACKEND], action='append')


def pytest_generate_tests(metafunc):
    if 'backend' in metafunc.fixturenames:
        metafunc.parametrize('backend', metafunc.config.getoption('--add-backend'))


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Seed every RNG before each test.

    Several tests assert shot statistics within 3-sigma bounds; with an
    unseeded RNG each such assertion fails by pure chance ~0.3% of the time,
    which across the CI matrix (3 Python versions x PR + push) made spurious
    red builds routine. Fixed seeds make the sampled outcomes deterministic
    while keeping the statistical assertions meaningful.
    """
    torch.manual_seed(1234)
    random.seed(1234)
    try:
        import numpy as np
        np.random.seed(1234)
    except ImportError:
        pass
