from blueqat import BlueqatGlobalSetting

DEFAULT_BACKEND = BlueqatGlobalSetting.get_default_backend_name()


def pytest_addoption(parser):
    parser.addoption('--add-backend', default=[DEFAULT_BACKEND], action='append')


def pytest_generate_tests(metafunc):
    if 'backend' in metafunc.fixturenames:
        metafunc.parametrize('backend', metafunc.config.getoption('--add-backend'))
