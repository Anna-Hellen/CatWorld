"""
lighting.py – Iluminação suave e quente (Bônus)

Modelo difuso com luz quente de fim de tarde:
  I = Ia·Ka + Id·Kd·max(0, dot(N̂, L̂))

Ajustes "cozy":
  - Ambiente alto (0.62) → sombras claras, nada fica escuro demais
  - Difusa moderada (0.42) → contraste suave
  - Tinte quente: canal R levemente reforçado, B levemente reduzido
"""

import numpy as np

LIGHT_DIR = np.array([0.4, 1.0, 0.45], dtype=np.float64)
LIGHT_DIR = LIGHT_DIR / np.linalg.norm(LIGHT_DIR)

AMBIENT  = 0.62
DIFFUSE  = 0.42

# Tinte quente da luz (multiplica por canal)
WARM_R = 1.04
WARM_G = 1.00
WARM_B = 0.94


def face_normal(v0, v1, v2):
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n)
    if norm < 1e-10:
        return np.array([0.0, 1.0, 0.0])
    return n / norm


def shade_color(base, normal, light_dir=LIGHT_DIR):
    d = max(0.0, float(np.dot(normal, light_dir)))
    intensity = min(1.0, AMBIENT + DIFFUSE * d)
    r = int(min(255, base[0] * intensity * WARM_R))
    g = int(min(255, base[1] * intensity * WARM_G))
    b = int(min(255, base[2] * intensity * WARM_B))
    return (r, g, b)


def shade_mesh_faces(vertices_world, faces):
    shaded = []
    for indices, base_color, draw_border in faces:
        if len(indices) < 3:
            shaded.append((indices, base_color, draw_border))
            continue
        n = face_normal(vertices_world[indices[0]],
                        vertices_world[indices[1]],
                        vertices_world[indices[2]])
        shaded.append((indices, shade_color(base_color, n), draw_border))
    return shaded
