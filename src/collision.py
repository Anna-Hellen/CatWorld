"""
collision.py – Detecção de Colisão (RT-06)
Implementa AABB (Axis-Aligned Bounding Box) e colisão por esfera.
"""

import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AABB:
    """Axis-Aligned Bounding Box 3D."""
    min_pt: np.ndarray  # (x_min, y_min, z_min)
    max_pt: np.ndarray  # (x_max, y_max, z_max)

    def intersects(self, other: "AABB") -> bool:
        """
        Dois AABBs colidem quando, em TODOS os eixos,
        os intervalos se sobrepõem:
          A.min[i] <= B.max[i]  AND  A.max[i] >= B.min[i]
        """
        return (
            self.min_pt[0] <= other.max_pt[0] and self.max_pt[0] >= other.min_pt[0] and
            self.min_pt[1] <= other.max_pt[1] and self.max_pt[1] >= other.min_pt[1] and
            self.min_pt[2] <= other.max_pt[2] and self.max_pt[2] >= other.min_pt[2]
        )

    def center(self) -> np.ndarray:
        return (self.min_pt + self.max_pt) * 0.5


@dataclass
class Sphere:
    """Bounding Sphere 3D."""
    center: np.ndarray
    radius: float

    def intersects_sphere(self, other: "Sphere") -> bool:
        """
        Colisão esfera-esfera:
        dist(A.center, B.center) <= A.radius + B.radius
        """
        dist = np.linalg.norm(self.center - other.center)
        return dist <= (self.radius + other.radius)

    def intersects_aabb(self, box: AABB) -> bool:
        """
        Ponto mais próximo do AABB à esfera, clamped a [min, max]:
        colide se dist(closest, center) <= radius.
        """
        closest = np.clip(self.center, box.min_pt, box.max_pt)
        dist = np.linalg.norm(closest - self.center)
        return dist <= self.radius


def make_aabb(center: Tuple[float, float, float],
              half_size: Tuple[float, float, float]) -> AABB:
    """Cria AABB a partir do centro e meio-tamanho (half-extents)."""
    c = np.array(center, dtype=np.float64)
    h = np.array(half_size, dtype=np.float64)
    return AABB(c - h, c + h)


def make_sphere(center: Tuple[float, float, float], radius: float) -> Sphere:
    return Sphere(np.array(center, dtype=np.float64), radius)
