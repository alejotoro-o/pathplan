from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Any
from pathplan.core.map import GridMap

class BaseSolver(ABC):
    """Abstract Base Class for all path planning solvers within PathPlan."""
    
    def __init__(self, grid_map: GridMap):
        self.grid_map = grid_map

    @abstractmethod
    def plan(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[Optional[List[Tuple[int, int]]], Any]:
        """
        Executes path tracking configuration.
        
        Args:
            start: Tuple representing (row, col) starting indices.
            goal: Tuple representing (row, col) goal indices.
            
        Returns:
            Tuple containing:
                - Path as a list of coordinates (or None if no path found)
                - Diagnostic data metadata dictionary or list (e.g., explored spaces)
        """
        pass