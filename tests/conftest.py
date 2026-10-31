import os
import pytest


@pytest.fixture(scope="session")
def dataset_available():
    return os.path.exists(
        "/media/master/2tbkup/MySpace/projects/dataset_kiti/sequences/00/calib.txt"
    )