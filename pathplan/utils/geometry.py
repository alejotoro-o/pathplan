from typing import Tuple
from pathplan.core.map import GridMap

def edge_collision_free(p1: Tuple[float, float], p2: Tuple[float, float], grid_map: GridMap, threshold: float = 0.5) -> bool:
    """
    Robust grid collision check using a supercover Bresenham line.
    Ensures every grid cell the line segment passes through is evaluated.
    """
    x0, y0 = int(round(p1[0])), int(round(p1[1]))
    x1, y1 = int(round(p2[0])), int(round(p2[1]))

    if not grid_map.is_valid_index(x0, y0) or not grid_map.is_valid_index(x1, y1):
        return False

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx - dy

    x, y = x0, y0
    
    if grid_map.is_occupied(x, y, threshold=threshold):
        return False

    while (x != x1) or (y != y1):
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
            if not grid_map.is_valid_index(x, y) or grid_map.is_occupied(x, y, threshold=threshold):
                return False
        if e2 < dx:
            err += dx
            y += sy
            if not grid_map.is_valid_index(x, y) or grid_map.is_occupied(x, y, threshold=threshold):
                return False

    return True