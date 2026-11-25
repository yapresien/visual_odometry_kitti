import os
import pytest


@pytest.fixture(scope="session")
def dataset_available():
    return os.path.exists(
        "/home/master/myspace/projects/dataset_kitti/sequences/00/calib.txt"
    )