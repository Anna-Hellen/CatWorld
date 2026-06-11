"""
renderer.py – Renderizador 3D com iluminação e projeção dinâmica (RT-01, RT-03, Bônus)
Agora aceita proj_matrix externo (para FOV dinâmico da câmera).
"""

import pygame
import numpy as np
import math
from typing import List, Tuple, Optional
from math3d import project_to_screen, perspective_projection
from lighting import shade_mesh_faces

Face = Tuple[List[int], Tuple[int, int, int], bool]


class Mesh:
    def __init__(self, vertices: np.ndarray, faces: List[Face]):
        self.vertices = np.array(vertices, dtype=np.float64)
        self.faces = faces


class Renderer:
    def __init__(self, surface: pygame.Surface):
        self.surface = surface
        self.width   = surface.get_width()
        self.height  = surface.get_height()
        # Proj padrão (sobrescrita pelo camera.proj_matrix() a cada frame)
        self.proj = perspective_projection(
            math.radians(60), self.width/self.height, 0.5, 500.0
        )
        self.lighting_enabled = True

    def set_proj(self, proj: np.ndarray):
        """Atualiza a matriz de projeção (FOV dinâmico)."""
        self.proj = proj

    def render_mesh(self, mesh: Mesh, model_matrix: np.ndarray,
                    view_matrix: np.ndarray,
                    tint: Optional[Tuple] = None,
                    wireframe: bool = False):
        verts_world = self._apply_model(mesh.vertices, model_matrix)

        if self.lighting_enabled and not wireframe and tint is None:
            shaded_faces = shade_mesh_faces(verts_world, mesh.faces)
        else:
            shaded_faces = mesh.faces

        pixels, depths, visible = project_to_screen(
            verts_world, view_matrix, self.proj, self.width, self.height
        )

        face_data = []
        for indices, color, draw_border in shaded_faces:
            if not all(visible[i] for i in indices):
                continue
            face_pixels = [pixels[i] for i in indices]
            mean_depth  = np.mean([depths[i] for i in indices])
            face_data.append((mean_depth, face_pixels, color, draw_border))

        face_data.sort(key=lambda x: x[0])

        for _, face_pixels, color, draw_border in face_data:
            pts = [(int(p[0]), int(p[1])) for p in face_pixels]
            if len(pts) < 3:
                continue
            draw_color = tint if tint else color
            if wireframe:
                pygame.draw.polygon(self.surface, draw_color, pts, 1)
            else:
                pygame.draw.polygon(self.surface, draw_color, pts)
                if draw_border:
                    bc = tuple(max(0, c - 45) for c in draw_color)
                    pygame.draw.polygon(self.surface, bc, pts, 1)

    def _apply_model(self, vertices, model):
        n = vertices.shape[0]
        h = np.ones((n, 4))
        h[:, :3] = vertices
        return ((model @ h.T).T)[:, :3]
