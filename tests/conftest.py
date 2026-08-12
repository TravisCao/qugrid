import matplotlib
import pytest

matplotlib.use("Agg")


@pytest.fixture()
def case9():
    import qugrid as qg

    return qg.cases.case9()


@pytest.fixture()
def toy3():
    import qugrid as qg

    return qg.cases.toy3()
