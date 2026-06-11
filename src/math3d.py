"""
math3d.py – Utilitários de matemática 3D
Transformações geométricas, projeções e operações com matrizes 4x4 homogêneas.
"""

import numpy as np
import math


# ── Transformações Geométricas ────────────────────────────────────────────────

def translation_matrix(tx: float, ty: float, tz: float) -> np.ndarray:
    """
    Matriz de translação 4×4 homogênea.

        | 1  0  0  tx |
    T = | 0  1  0  ty |
        | 0  0  1  tz |
        | 0  0  0   1 |
    """
    m = np.eye(4, dtype=np.float64)
    m[0, 3] = tx
    m[1, 3] = ty
    m[2, 3] = tz
    return m


def scale_matrix(sx: float, sy: float, sz: float) -> np.ndarray:
    """
    Matriz de escala 4×4 homogênea.

        | sx  0   0   0 |
    S = |  0 sy   0   0 |
        |  0  0  sz   0 |
        |  0  0   0   1 |
    """
    m = np.eye(4, dtype=np.float64)
    m[0, 0] = sx
    m[1, 1] = sy
    m[2, 2] = sz
    return m


def rotation_x(angle: float) -> np.ndarray:
    """
    Rotação em torno do eixo X (ângulo em radianos).

         | 1    0       0    0 |
    Rx = | 0  cos θ  -sin θ  0 |
         | 0  sin θ   cos θ  0 |
         | 0    0       0    1 |
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1],
    ], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    """
    Rotação em torno do eixo Y (ângulo em radianos).

         |  cos θ  0  sin θ  0 |
    Ry = |    0    1    0    0 |
         | -sin θ  0  cos θ  0 |
         |    0    0    0    1 |
    """
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [ c, 0, s, 0],
        [ 0, 1, 0, 0],
        [-s, 0, c, 0],
        [ 0, 0, 0, 1],
    ], dtype=np.float64)


def rotation_z(angle: float) -> np.ndarray:
    """Rotação em torno do eixo Z (ângulo em radianos)."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([
        [c, -s, 0, 0],
        [s,  c, 0, 0],
        [0,  0, 1, 0],
        [0,  0, 0, 1],
    ], dtype=np.float64)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Aplica uma transformação 4×4 a um array de pontos (N×3).
    Converte para coordenadas homogêneas, multiplica e converte de volta.
    """
    n = points.shape[0]
    h = np.ones((n, 4), dtype=np.float64)
    h[:, :3] = points
    result = (matrix @ h.T).T  # (4×4) @ (4×N) → (N×4)
    return result[:, :3]


# ── Câmera / View Matrix ──────────────────────────────────────────────────────

def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Constrói a View Matrix (LookAt).

    Normaliza os vetores da base da câmera:
      f = normalize(target - eye)   (frente)
      r = normalize(f × up)         (direita)
      u = r × f                     (cima real)

    A matriz final é:
        | rx  ry  rz  -dot(r,eye) |
    V = | ux  uy  uz  -dot(u,eye) |
        |-fx -fy -fz   dot(f,eye) |
        |  0   0   0       1      |
    """
    f = target - eye
    f = f / np.linalg.norm(f)

    r = np.cross(f, up)
    r = r / np.linalg.norm(r)

    u = np.cross(r, f)

    view = np.array([
        [ r[0],  r[1],  r[2], -np.dot(r, eye)],
        [ u[0],  u[1],  u[2], -np.dot(u, eye)],
        [-f[0], -f[1], -f[2],  np.dot(f, eye)],
        [    0,      0,     0,               1],
    ], dtype=np.float64)
    return view


# ── Projeção Perspectiva ──────────────────────────────────────────────────────

def perspective_projection(fov_y: float, aspect: float,
                            near: float, far: float) -> np.ndarray:
    """
    Matriz de projeção perspectiva (OpenGL convention).

    fov_y  : ângulo vertical do campo de visão (radianos)
    aspect : largura / altura da janela
    near   : plano near (> 0)
    far    : plano far  (> near)

    f = 1 / tan(fov_y / 2)

        | f/aspect   0         0              0        |
    P = |    0       f         0              0        |
        |    0       0  (f+n)/(n-f)   (2·f·n)/(n-f)   |
        |    0       0        -1              0        |
    """
    f = 1.0 / math.tan(fov_y / 2.0)
    return np.array([
        [f / aspect, 0,                          0,                         0],
        [0,          f,                          0,                         0],
        [0,          0,  (far + near) / (near - far), 2*far*near/(near - far)],
        [0,          0,                         -1,                         0],
    ], dtype=np.float64)


def project_to_screen(points_3d: np.ndarray, view: np.ndarray,
                       proj: np.ndarray, width: int, height: int):
    """
    Transforma pontos 3D → pixels 2D via pipeline MVP.
    Retorna (pixels, depths, mask_visible).

    Pipeline:
      1. View transform  : aplica look_at
      2. Proj transform  : divide perspectiva (clip space)
      3. Divisão por w   : NDC  [-1, 1]
      4. Viewport        : NDC → pixels
    """
    n = points_3d.shape[0]
    h = np.ones((n, 4))
    h[:, :3] = points_3d

    # View
    h = (view @ h.T).T
    depths = h[:, 2].copy()

    # Projection
    h = (proj @ h.T).T

    # Clipping: descarta pontos atrás da câmera
    w = h[:, 3]
    visible = w > 0.01

    # Divisão perspectiva → NDC
    with np.errstate(divide='ignore', invalid='ignore'):
        x_ndc = np.where(visible, h[:, 0] / w, 0)
        y_ndc = np.where(visible, h[:, 1] / w, 0)

    # NDC → viewport (0…width, 0…height)
    px = ((x_ndc + 1) * 0.5 * width).astype(np.float64)
    py = ((1 - y_ndc) * 0.5 * height).astype(np.float64)

    pixels = np.stack([px, py], axis=1)
    return pixels, depths, visible
