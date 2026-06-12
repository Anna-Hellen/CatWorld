"""
renderer.py – Renderizador 3D otimizado (RT-01, RT-03)

Pipeline de alto desempenho (tudo vetorizado em NumPy):
  1. PRÉ-PROCESSAMENTO (uma vez): faces agrupadas em arrays de índices
     (triângulos N×3, quads N×4), normais e centros pré-calculados.
  2. BATCH ESTÁTICO: todo o cenário fixo (chão, muros, árvores, casas...)
     é fundido em UM conjunto de arrays no carregamento, com transformações
     e ILUMINAÇÃO JÁ APLICADAS (baked). Por frame: 1 multiplicação de matriz.
  3. MALHAS DINÂMICAS (gato, peixes): sombreamento vetorizado por frame
     usando normais pré-calculadas rotacionadas em bloco.
  4. CULLING vetorizado: descarta faces atrás da câmera, fora da tela,
     menores que ~1.5 px e voltadas para trás (backface, ~50% das faces).
  5. ORDENAÇÃO GLOBAL: todas as faces visíveis do frame são ordenadas
     por profundidade de uma vez (np.argsort) → painter's algorithm correto
     entre objetos, não só dentro de cada um.
  6. Único loop Python restante: as chamadas pygame.draw.polygon (em C).
"""

import pygame
import numpy as np
from typing import List

from lighting import AMBIENT, DIFFUSE, LIGHT_DIR, WARM_R, WARM_G, WARM_B

_WARM = np.array([WARM_R, WARM_G, WARM_B])


class Mesh:
    """Malha bruta: vértices (N×3) e faces [(indices, cor, borda[, cull])]."""
    def __init__(self, vertices, faces):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        self.faces = faces


class PreparedMesh:
    """
    Malha pré-processada para renderização vetorizada.
    Faces separadas por aridade (tri/quad) em arrays de índices,
    com cores-base, flags e normais/centros em object space.
    """
    __slots__ = ("verts", "tri_idx", "quad_idx", "tri_col", "quad_col",
                 "tri_bord", "quad_bord", "tri_cull", "quad_cull",
                 "tri_nrm", "quad_nrm", "tri_ctr", "quad_ctr")

    def __init__(self, mesh: Mesh):
        self.verts = mesh.vertices.copy()
        tris, quads = [], []
        t_col, q_col, t_b, q_b, t_cu, q_cu = [], [], [], [], [], []
        for face in mesh.faces:
            if len(face) == 4:
                idx, col, bord, cull = face
            else:
                idx, col, bord = face
                cull = False          # faces avulsas: 2 lados visíveis
            if len(idx) == 3:
                tris.append(idx);  t_col.append(col)
                t_b.append(bord);  t_cu.append(cull)
            elif len(idx) == 4:
                quads.append(idx); q_col.append(col)
                q_b.append(bord);  q_cu.append(cull)

        self.tri_idx  = np.array(tris,  dtype=np.int32).reshape(-1, 3)
        self.quad_idx = np.array(quads, dtype=np.int32).reshape(-1, 4)
        self.tri_col  = np.array(t_col, dtype=np.float64).reshape(-1, 3)
        self.quad_col = np.array(q_col, dtype=np.float64).reshape(-1, 3)
        self.tri_bord  = np.array(t_b,  dtype=bool)
        self.quad_bord = np.array(q_b,  dtype=bool)
        self.tri_cull  = np.array(t_cu, dtype=bool)
        self.quad_cull = np.array(q_cu, dtype=bool)

        v = self.verts
        self.tri_nrm,  self.tri_ctr  = _face_geom(v, self.tri_idx)
        self.quad_nrm, self.quad_ctr = _face_geom(v, self.quad_idx)


def _face_geom(verts, idx):
    """Normais (unitárias) e centros das faces, vetorizado."""
    if len(idx) == 0:
        return (np.zeros((0, 3)), np.zeros((0, 3)))
    p0 = verts[idx[:, 0]]
    p1 = verts[idx[:, 1]]
    p2 = verts[idx[:, 2]]
    n = np.cross(p1 - p0, p2 - p0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    n /= ln
    ctr = verts[idx].mean(axis=1)
    return n, ctr


def shade_colors(base, normals):
    """Iluminação difusa vetorizada: cores (F×3) já tonalizadas."""
    d = np.clip(normals @ LIGHT_DIR, 0.0, None)
    inten = np.minimum(1.0, AMBIENT + DIFFUSE * d)[:, None]
    return np.clip(base * inten * _WARM, 0, 255)


class StaticBatch:
    """
    Cenário estático inteiro fundido: vértices em world space,
    iluminação baked, normais/centros em world space.
    """
    def __init__(self, pairs):
        """pairs: lista de (Mesh, model_matrix 4×4)."""
        all_v, all_t, all_q = [], [], []
        t_col, q_col, t_b, q_b, t_cu, q_cu = [], [], [], [], [], []
        off = 0
        for mesh, M in pairs:
            pm = PreparedMesh(mesh)
            R3, T3 = M[:3, :3], M[:3, 3]
            vw = pm.verts @ R3.T + T3
            all_v.append(vw)
            if len(pm.tri_idx):
                all_t.append(pm.tri_idx + off)
                # Baka iluminação com normais em world space
                nrm = pm.tri_nrm @ R3.T
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                t_col.append(shade_colors(pm.tri_col, nrm))
                t_b.append(pm.tri_bord); t_cu.append(pm.tri_cull)
            if len(pm.quad_idx):
                all_q.append(pm.quad_idx + off)
                nrm = pm.quad_nrm @ R3.T
                nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
                q_col.append(shade_colors(pm.quad_col, nrm))
                q_b.append(pm.quad_bord); q_cu.append(pm.quad_cull)
            off += len(vw)

        self.verts = np.concatenate(all_v) if all_v else np.zeros((0, 3))
        z3 = np.zeros((0, 3), dtype=np.int32)
        z4 = np.zeros((0, 4), dtype=np.int32)
        self.tri_idx  = np.concatenate(all_t) if all_t else z3
        self.quad_idx = np.concatenate(all_q) if all_q else z4
        self.tri_col  = (np.concatenate(t_col) if t_col
                         else np.zeros((0, 3))).astype(np.int16)
        self.quad_col = (np.concatenate(q_col) if q_col
                         else np.zeros((0, 3))).astype(np.int16)
        self.tri_bord  = np.concatenate(t_b)  if t_b  else np.zeros(0, bool)
        self.quad_bord = np.concatenate(q_b)  if q_b  else np.zeros(0, bool)
        self.tri_cull  = np.concatenate(t_cu) if t_cu else np.zeros(0, bool)
        self.quad_cull = np.concatenate(q_cu) if q_cu else np.zeros(0, bool)
        # Normais/centros world (para backface culling por frame)
        self.tri_nrm,  self.tri_ctr  = _face_geom(self.verts, self.tri_idx)
        self.quad_nrm, self.quad_ctr = _face_geom(self.verts, self.quad_idx)


class Renderer:
    """
    Renderizador com superfície interna escalável (render scale).
    Uso por frame:
        renderer.begin(view, proj, eye)
        renderer.submit_static(batch)
        renderer.submit_mesh(prep, model)   # dinâmicos
        renderer.flush()                    # ordena tudo e desenha
    """

    def __init__(self, screen: pygame.Surface, render_scale: float = 0.8):
        self.screen = screen
        self.W = screen.get_width()
        self.H = screen.get_height()
        self.set_scale(render_scale)
        self._frame_pts:  List = []
        self._frame_cols: List = []
        self._frame_bord: List = []
        self._frame_depth: List = []

    def set_scale(self, scale: float):
        """Define a resolução interna de renderização."""
        self.scale = scale
        self.rw = max(320, int(self.W * scale))
        self.rh = max(180, int(self.H * scale))
        self.surface = pygame.Surface((self.rw, self.rh))

    # ── Frame ────────────────────────────────────────────────────────────────

    def begin(self, view: np.ndarray, proj: np.ndarray, eye: np.ndarray):
        self._VP  = proj @ view
        self._eye = eye
        self._frame_pts.clear()
        self._frame_cols.clear()
        self._frame_bord.clear()
        self._frame_depth.clear()

    def _project(self, verts_w):
        """Projeta vértices world→pixels. Retorna (pix int32, depth, w_ok)."""
        n = len(verts_w)
        h = np.empty((n, 4))
        h[:, :3] = verts_w
        h[:, 3] = 1.0
        clip = h @ self._VP.T
        w = clip[:, 3]
        ok = w > 0.05
        w_safe = np.where(ok, w, 1.0)
        x = clip[:, 0] / w_safe
        y = clip[:, 1] / w_safe
        px = (x + 1.0) * (0.5 * self.rw)
        py = (1.0 - y) * (0.5 * self.rh)
        pix = np.stack([np.clip(px, -4000, 4000),
                        np.clip(py, -4000, 4000)], axis=1)
        pix = np.rint(pix).astype(np.int32)
        # Profundidade p/ ordenação: -w (maior w = mais perto)
        return pix, -w, ok

    def _submit_group(self, pix, depth, ok, idx, cols, bord,
                      cull_mask, nrm_w, ctr_w):
        """Processa um grupo de faces (tri ou quad) e acumula no frame."""
        if len(idx) == 0:
            return
        # 1. Todos os vértices da face com w válido
        vis = ok[idx].all(axis=1)
        # 2. Backface culling (apenas faces de sólidos fechados)
        if cull_mask.any():
            facing = ((ctr_w - self._eye) * nrm_w).sum(axis=1) < 0.0
            vis &= np.where(cull_mask, facing, True)
        if not vis.any():
            return
        sel = np.nonzero(vis)[0]
        fpts = pix[idx[sel]]                     # (F, k, 2)
        # 3. Bounding box: fora da tela ou minúscula (<1.5 px)
        mn = fpts.min(axis=1); mx = fpts.max(axis=1)
        on = ((mx[:, 0] >= 0) & (mn[:, 0] < self.rw) &
              (mx[:, 1] >= 0) & (mn[:, 1] < self.rh))
        size_ok = ((mx[:, 0] - mn[:, 0] > 1) | (mx[:, 1] - mn[:, 1] > 1))
        keep = on & size_ok
        if not keep.any():
            return
        sel = sel[keep]
        fpts = fpts[keep]
        fdep = depth[idx[sel]].mean(axis=1)

        self._frame_pts.extend(fpts.tolist())
        self._frame_cols.extend(map(tuple, cols[sel].tolist()))
        self._frame_bord.extend(bord[sel].tolist())
        self._frame_depth.append(fdep)

    def submit_static(self, b: StaticBatch):
        pix, depth, ok = self._project(b.verts)
        self._submit_group(pix, depth, ok, b.tri_idx,  b.tri_col,
                           b.tri_bord,  b.tri_cull,  b.tri_nrm,  b.tri_ctr)
        self._submit_group(pix, depth, ok, b.quad_idx, b.quad_col,
                           b.quad_bord, b.quad_cull, b.quad_nrm, b.quad_ctr)

    def submit_mesh(self, pm: PreparedMesh, model: np.ndarray,
                    tint=None):
        """Malha dinâmica: transforma, sombreia (vetorizado) e acumula."""
        R3, T3 = model[:3, :3], model[:3, 3]
        vw = pm.verts @ R3.T + T3
        pix, depth, ok = self._project(vw)

        for idx, base, bord, cull, nrm_o, ctr_o in (
            (pm.tri_idx,  pm.tri_col,  pm.tri_bord,  pm.tri_cull,
             pm.tri_nrm,  pm.tri_ctr),
            (pm.quad_idx, pm.quad_col, pm.quad_bord, pm.quad_cull,
             pm.quad_nrm, pm.quad_ctr),
        ):
            if len(idx) == 0:
                continue
            nrm_w = nrm_o @ R3.T
            nrm_w /= np.maximum(
                np.linalg.norm(nrm_w, axis=1, keepdims=True), 1e-12)
            ctr_w = ctr_o @ R3.T + T3
            if tint is not None:
                cols = np.full((len(idx), 3), tint, dtype=np.int16)
            else:
                cols = shade_colors(base, nrm_w).astype(np.int16)
            self._submit_group(pix, depth, ok, idx, cols, bord, cull,
                               nrm_w, ctr_w)

    def flush(self):
        """Ordena TODAS as faces do frame por profundidade e desenha."""
        if not self._frame_pts:
            return
        depths = np.concatenate(self._frame_depth)
        order = np.argsort(depths)            # mais distante primeiro
        pts  = self._frame_pts
        cols = self._frame_cols
        bord = self._frame_bord
        surf = self.surface
        draw = pygame.draw.polygon
        for i in order:
            p = pts[i]
            c = cols[i]
            draw(surf, c, p)
            if bord[i]:
                draw(surf, (max(0, c[0]-45), max(0, c[1]-45),
                            max(0, c[2]-45)), p, 1)

    def blit_to_screen(self):
        """Escala a superfície interna para a janela."""
        if self.scale >= 0.999:
            self.screen.blit(self.surface, (0, 0))
        else:
            pygame.transform.scale(self.surface, (self.W, self.H),
                                   self.screen)
