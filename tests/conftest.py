import pytest
import numpy as np
from pathplan.core.map import GridMap

@pytest.fixture
def empty_map_20x20():
    """Generates an open 20x20 grid map containing zero obstacles."""
    data = np.zeros((20, 20))
    return GridMap(data)

@pytest.fixture
def wall_map_20x20():
    """
    Generates a 20x20 grid divided by a vertical obstacle wall 
    with a single open passage gap at index (10, 10).
    """
    data = np.zeros((20, 20))
    data[5:15, 10] = 1.0
    data[10, 10] = 0.0  # Open navigation gate
    return GridMap(data)

@pytest.fixture
def blocked_map_5x5():
    """Generates a highly constrained 5x5 map with an unpassable center barrier."""
    data = np.zeros((5, 5))
    data[:, 2] = 1.0  # solid vertical block slice
    return GridMap(data)