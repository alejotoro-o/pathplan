import numpy as np

class GridMap:
    """
    A normalized grid representation for path planning.
    
    Values are normalized between 0.0 (completely free) and 1.0 (lethal obstacle).
    """
    def __init__(self, data: np.ndarray):
        """
        Initializes the GridMap with a 2D numpy array.
        Values inside the array will be clipped between 0.0 and 1.0.
        """
        if data.ndim != 2:
            raise ValueError("GridMap data must be a 2D matrix.")
        
        # Ensure values stay normalized between 0.0 and 1.0
        self.data = np.clip(data.astype(np.float32), 0.0, 1.0)
        self.shape = self.data.shape
        self.height, self.width = self.shape

    def is_valid_index(self, row: int, col: int) -> bool:
        """Checks if index coordinates reside within bounds."""
        return 0 <= row < self.height and 0 <= col < self.width

    def is_occupied(self, row: int, col: int, threshold: float = 0.5) -> bool:
        """Determines if a cell is untraversable based on an occupancy threshold."""
        if not self.is_valid_index(row, col):
            return True  # Out of bounds is treated as occupied
        return self.data[row, col] >= threshold

    def get_cost(self, row: int, col: int) -> float:
        """Returns the cost value of the cell."""
        return float(self.data[row, col])