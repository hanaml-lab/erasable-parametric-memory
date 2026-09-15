import pytest

from slate.toy import make_world


@pytest.fixture(scope="session")
def world():
    return make_world(n_contacts=10, n_public=24, weeks=3, seed=3)
