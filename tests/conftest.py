import pytest


@pytest.fixture(autouse=True)
def reset_to_default_backend():
    """Keep legacy Von tests isolated without importing its optional stack at collection."""
    yield
    try:
        import von
    except ModuleNotFoundError:
        return
    von.set_backend("von-1.0")
