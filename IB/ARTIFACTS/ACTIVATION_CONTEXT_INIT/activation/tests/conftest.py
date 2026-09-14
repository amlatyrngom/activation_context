import pytest

GATED_MARKERS = {
    "manual": "--manual",
    "slow": "--slow",
    "gpu": "--gpu",
    "ddp": "--ddp",
}

def pytest_addoption(parser):
    group = parser.getgroup("activation test selection")
    group.addoption("--slow", action="store_true", help="run slow tests")
    group.addoption("--manual", action="store_true", help="run manual tests")
    group.addoption("--gpu", action="store_true", help="run GPU tests")
    group.addoption("--ddp", action="store_true", help="run multi-GPU tests")


def pytest_collection_modifyitems(config, items):
    for item in items:
        missing_options = [
            option
            for marker, option in GATED_MARKERS.items()
            if item.get_closest_marker(marker) is not None
            and not config.getoption(option)
        ]

        if missing_options:
            item.add_marker(
                pytest.mark.skip(
                    reason=f"requires {' and '.join(missing_options)}"
                )
            )