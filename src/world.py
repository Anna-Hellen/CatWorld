"""
world.py – Mundo cozy com BATCH ESTÁTICO (otimização principal)

Todo o cenário fixo (chão, muros, cercas, flores, cogumelos, arbustos,
lanternas, casas, árvores, caixotes) é fundido em um único StaticBatch
no carregamento: transformações aplicadas e iluminação pré-calculada.
Por frame, o custo do cenário inteiro = 1 multiplicação de matriz +
culling/ordenação vetorizados, em vez de ~280 chamadas separadas.
"""

import numpy as np
import math
import random
from typing import List

from renderer import Renderer, StaticBatch
from math3d import translation_matrix, rotation_y, scale_matrix
from geometry import (floor_tile_mesh, wall_segment_mesh, crate_mesh,
                      stone_mesh, tree_mesh, house_mesh, flower_mesh,
                      mushroom_mesh, lantern_mesh, bush_mesh,
                      fence_segment_mesh,
                      C_GRASS_A, C_GRASS_B, C_GRASS_C, C_PATH)
from entities import Fish
from collision import AABB, make_aabb

WORLD_X = (-22, 22)
WORLD_Z = (-22, 22)
WALL_H  = 2.2


def _M(x, y, z, yaw=0.0, s=1.0):
    """Helper: matriz model T·Ry·S."""
    M = translation_matrix(x, y, z) @ rotation_y(yaw)
    if s != 1.0:
        M = M @ scale_matrix(s, s, s)
    return M


class World:
    TILE_SIZE = 4.0

    def __init__(self, seed=42, n_fish=15, n_stars=3, density=1.0):
        """density: 0..1 — fator de quantidade de decoração (qualidade)."""
        random.seed(seed)
        self.density = density
        pairs = []            # (Mesh, model) para o batch estático
        self._aabbs: List[AABB] = []

        self._build_floor(pairs)
        self._build_walls(pairs)
        self._build_obstacles(pairs)
        self._build_decorations(pairs)
        self._spawn_collectibles(n_fish, n_stars)

        # ★ Funde tudo em um único batch com iluminação baked
        self.static_batch = StaticBatch(pairs)

    # ── Construção ───────────────────────────────────────────────────────────

    def _build_floor(self, pairs):
        t = self.TILE_SIZE
        meshes = {
            'a': floor_tile_mesh(t, C_GRASS_A),
            'b': floor_tile_mesh(t, C_GRASS_B),
            'c': floor_tile_mesh(t, C_GRASS_C),
            'p': floor_tile_mesh(t, C_PATH),
        }
        rng = random.Random(7)
        xs = np.arange(WORLD_X[0], WORLD_X[1], t)
        zs = np.arange(WORLD_Z[0], WORLD_Z[1], t)
        for xi, x in enumerate(xs):
            for zi, z in enumerate(zs):
                cx, cz = x + t/2, z + t/2
                if abs(cx) < t*0.6 or abs(cz) < t*0.6:
                    key = 'p'
                else:
                    r = rng.random()
                    key = 'c' if r < 0.18 else ('a' if (xi+zi)%2==0 else 'b')
                pairs.append((meshes[key], _M(cx, 0, cz)))

    def _build_walls(self, pairs):
        lx = WORLD_X[1]-WORLD_X[0]; lz = WORLD_Z[1]-WORLD_Z[0]
        hw = WALL_H/2
        mx = wall_segment_mesh(lx, WALL_H)
        mz = wall_segment_mesh(lz, WALL_H)
        pairs.append((mx, _M(0, 0, WORLD_Z[0])))
        pairs.append((mx, _M(0, 0, WORLD_Z[1])))
        pairs.append((mz, _M(WORLD_X[1], 0, 0, yaw=math.pi/2)))
        pairs.append((mz, _M(WORLD_X[0], 0, 0, yaw=math.pi/2)))
        self._aabbs += [
            make_aabb((0, hw, WORLD_Z[0]), (lx/2, hw, 0.9)),
            make_aabb((0, hw, WORLD_Z[1]), (lx/2, hw, 0.9)),
            make_aabb((WORLD_X[1], hw, 0), (0.9, hw, lz/2)),
            make_aabb((WORLD_X[0], hw, 0), (0.9, hw, lz/2)),
        ]

    def _build_obstacles(self, pairs):
        positions = [
            (-8,-8,1.4,1.5,1.4), ( 8,-8,1.6,1.3,1.6),
            (-8, 8,1.3,1.6,1.3), ( 8, 8,1.7,1.2,1.7),
            ( 0,-12,1.5,1.4,1.5),( 0,12,1.3,1.7,1.3),
            (-14,0,2.0,0.8,2.0), (14,0,1.8,0.9,1.8),
            (-5, 4,1.3,1.2,1.3), ( 5,-4,1.4,1.3,1.4),
            (-11,7,1.6,0.9,1.6), (11,-7,1.5,0.8,1.5),
            (-3,15,1.2,1.6,1.2), ( 3,-15,1.3,1.5,1.3),
            (16,10,1.5,1.4,1.5),(-16,-10,1.4,1.3,1.4),
        ]
        self._obstacle_centers = []
        for x, z, w, h, d in positions:
            mesh = stone_mesh(w,h,d) if h < 1.0 else crate_mesh(w,h,d)
            pairs.append((mesh, _M(x, 0, z)))
            self._aabbs.append(make_aabb((x, h/2, z), (w/2, h/2, d/2)))
            self._obstacle_centers.append((x, z))

    def _build_decorations(self, pairs):
        d = self.density
        # Árvores (verdes + cerejeiras)
        tree_data = [
            (-18,-18,1.2,True),( 18,-18,1.0,False),(-18,18,1.3,False),
            ( 18,18,0.95,True),(-20,0,1.1,False), ( 20,0,1.15,True),
            ( 0,-20,1.0,True), ( 0,20,1.1,False), (-15,-12,0.85,False),
            (15,12,0.9,True),  (-12,15,1.05,True),(12,-15,0.85,False),
            (-19,8,1.0,False), (19,-8,1.0,True),  (-7,-19,0.9,True),
            ( 7,19,0.95,False),
        ]
        mesh_green = tree_mesh(pink=False)
        mesh_pink  = tree_mesh(pink=True)
        for x, z, s, pink in tree_data:
            pairs.append((mesh_pink if pink else mesh_green,
                          _M(x, 0, z, yaw=random.uniform(0, 6.28), s=s)))
            self._aabbs.append(make_aabb((x, 1.5, z), (0.55*s, 3.5, 0.55*s)))

        # Casas
        mesh_house = house_mesh()
        for x, z, yaw in [(-19,-14,0.5),(19,14,math.pi),
                          (-16,19,1.0),(16,-19,2.5)]:
            pairs.append((mesh_house, _M(x, 0, z, yaw=yaw)))
            self._aabbs.append(make_aabb((x, 1.5, z), (1.9, 3.5, 1.6)))

        obs_c = self._obstacle_centers
        def clear(x, z, dist=2.0):
            return all(math.hypot(x-ox,z-oz) > dist for ox,oz in obs_c)

        # Flores (3 meshes compartilhadas por cor; quantidade × densidade)
        FLOWER_COLORS = [(245,170,185),(255,215,130),(205,170,235),
                         (252,245,235),(255,180,150),(180,215,250)]
        flower_meshes = [flower_mesh(c) for c in FLOWER_COLORS]
        for _ in range(int(55 * d)):
            for _t in range(20):
                x = random.uniform(-20, 20); z = random.uniform(-20, 20)
                if clear(x,z) and (abs(x)>2.2 or abs(z)>2.2):
                    pairs.append((random.choice(flower_meshes),
                                  _M(x, 0, z, yaw=random.uniform(0,6.28),
                                     s=random.uniform(0.85, 1.3))))
                    break

        # Cogumelos perto das árvores (compartilha 2 variantes)
        mush = [mushroom_mesh(0.9), mushroom_mesh(1.3)]
        for x, z, s, pink in tree_data[:int(10 * d)]:
            a = random.uniform(0, 2*math.pi)
            r = random.uniform(1.2, 2.2)
            mx, mz = x + math.cos(a)*r, z + math.sin(a)*r
            if abs(mx) < 20.5 and abs(mz) < 20.5:
                pairs.append((random.choice(mush),
                              _M(mx, 0, mz, yaw=random.uniform(0,6.28))))

        # Lanternas no caminho
        mesh_lant = lantern_mesh()
        for dd in [-15, -9, 9, 15]:
            pairs.append((mesh_lant, _M(dd, 0, 2.6)))
            pairs.append((mesh_lant, _M(2.6, 0, dd)))

        # Arbustos (2 variantes compartilhadas)
        bushes = [bush_mesh(0.9), bush_mesh(1.4)]
        for _ in range(int(14 * d)):
            for _t in range(20):
                x = random.uniform(-19, 19); z = random.uniform(-19, 19)
                if clear(x, z, 2.6) and (abs(x)>3.5 or abs(z)>3.5):
                    pairs.append((random.choice(bushes),
                                  _M(x, 0, z, yaw=random.uniform(0,6.28))))
                    break

        # Cerquinha branca
        mesh_fence = fence_segment_mesh(3.0)
        for x in range(-15, 16, 6):
            pairs.append((mesh_fence, _M(x, 0, WORLD_Z[0]+2.2)))
            pairs.append((mesh_fence, _M(x, 0, WORLD_Z[1]-2.2)))
        for z in range(-15, 16, 6):
            pairs.append((mesh_fence, _M(WORLD_X[0]+2.2, 0, z, yaw=math.pi/2)))
            pairs.append((mesh_fence, _M(WORLD_X[1]-2.2, 0, z, yaw=math.pi/2)))

    def _spawn_collectibles(self, n_fish, n_stars):
        self.fishes: List[Fish] = []
        obs_c = self._obstacle_centers
        def safe(x, z):
            if any(math.hypot(x-ox,z-oz) < 2.8 for ox,oz in obs_c):
                return False
            return abs(x)<=19.5 and abs(z)<=19.5 and not (abs(x)<2.5 and abs(z)<2.5)
        for is_star, count in [(False,n_fish),(True,n_stars)]:
            placed = tries = 0
            while placed < count and tries < 600:
                tries += 1
                x = random.uniform(-19.5, 19.5); z = random.uniform(-19.5, 19.5)
                if safe(x, z):
                    self.fishes.append(Fish(x, z, is_star=is_star)); placed += 1

    # ── Acesso ───────────────────────────────────────────────────────────────

    @property
    def all_obstacle_aabbs(self) -> List[AABB]:
        return self._aabbs       # cacheado (era recriado todo frame)

    def render(self, renderer: Renderer):
        renderer.submit_static(self.static_batch)

    def render_collectibles(self, renderer: Renderer):
        for fish in self.fishes:
            if not fish.collected or fish.collect_anim < 1.2:
                tint = (255,255,255) if fish.collect_anim > 0 else None
                renderer.submit_mesh(fish.prep, fish.model_matrix(), tint=tint)
