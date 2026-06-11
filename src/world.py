"""
world.py – Jardim aconchegante: cerejeiras, cogumelos, lanternas, arbustos
"""

import numpy as np
import math
import random
from typing import List

from renderer import Renderer
from math3d import translation_matrix, rotation_y
from geometry import (floor_tile_mesh, wall_segment_mesh,
                      C_GRASS_A, C_GRASS_B, C_GRASS_C, C_PATH)
from entities import (Fish, Obstacle, Tree, House, Flower, Fence,
                      Mushroom, Lantern, Bush)
from collision import AABB, make_aabb

WORLD_X = (-22, 22)
WORLD_Z = (-22, 22)
WALL_H  = 2.2


class World:
    TILE_SIZE = 4.0

    def __init__(self, seed=42, n_fish=15, n_stars=3):
        random.seed(seed)
        self._build_floor()
        self._build_walls()
        self._build_obstacles()
        self._build_decorations()
        self._spawn_collectibles(n_fish, n_stars)

    def _build_floor(self):
        """Grama em 3 tons pastel + caminho de areia em cruz."""
        self._floor_tiles = []
        t = self.TILE_SIZE
        meshes = {
            'a': floor_tile_mesh(t, C_GRASS_A),
            'b': floor_tile_mesh(t, C_GRASS_B),
            'c': floor_tile_mesh(t, C_GRASS_C),
            'p': floor_tile_mesh(t, C_PATH),
        }
        xs = np.arange(WORLD_X[0], WORLD_X[1], t)
        zs = np.arange(WORLD_Z[0], WORLD_Z[1], t)
        rng = random.Random(7)
        for xi, x in enumerate(xs):
            for zi, z in enumerate(zs):
                cx, cz = x + t/2, z + t/2
                if abs(cx) < t*0.6 or abs(cz) < t*0.6:
                    key = 'p'
                else:
                    r = rng.random()
                    key = 'c' if r < 0.18 else ('a' if (xi+zi)%2==0 else 'b')
                self._floor_tiles.append(
                    (meshes[key], translation_matrix(cx, 0, cz)))

    def _build_walls(self):
        lx = WORLD_X[1]-WORLD_X[0]; lz = WORLD_Z[1]-WORLD_Z[0]
        cx = 0.0; cz = 0.0; hw = WALL_H/2
        mx = wall_segment_mesh(lx, WALL_H)
        mz = wall_segment_mesh(lz, WALL_H)
        R90 = rotation_y(math.pi/2)
        self._wall_data = [
            (mx, translation_matrix(cx, 0, WORLD_Z[0])),
            (mx, translation_matrix(cx, 0, WORLD_Z[1])),
            (mz, translation_matrix(WORLD_X[1], 0, cz) @ R90),
            (mz, translation_matrix(WORLD_X[0], 0, cz) @ R90),
        ]
        self.wall_aabbs: List[AABB] = [
            make_aabb((cx, hw, WORLD_Z[0]), (lx/2, hw, 0.9)),
            make_aabb((cx, hw, WORLD_Z[1]), (lx/2, hw, 0.9)),
            make_aabb((WORLD_X[1], hw, cz), (0.9, hw, lz/2)),
            make_aabb((WORLD_X[0], hw, cz), (0.9, hw, lz/2)),
        ]

    def _build_obstacles(self):
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
        self.obstacles = [Obstacle(x,z,w,h,d) for x,z,w,h,d in positions]

    def _build_decorations(self):
        # Árvores – mistura de verdes e CEREJEIRAS rosas
        self.trees: List[Tree] = []
        tree_data = [
            (-18,-18,1.2,True),( 18,-18,1.0,False),(-18,18,1.3,False),
            ( 18,18,0.95,True),(-20,0,1.1,False), ( 20,0,1.15,True),
            ( 0,-20,1.0,True), ( 0,20,1.1,False), (-15,-12,0.85,False),
            (15,12,0.9,True),  (-12,15,1.05,True),(12,-15,0.85,False),
            (-19,8,1.0,False), (19,-8,1.0,True),  (-7,-19,0.9,True),
            ( 7,19,0.95,False),
        ]
        for x, z, s, pink in tree_data:
            self.trees.append(Tree(x, z, s, pink=pink))

        self.houses = [House(x,z,yaw) for x,z,yaw in
                       [(-19,-14,0.5),(19,14,math.pi),
                        (-16,19,1.0),(16,-19,2.5)]]

        obs_c = [(o.pos[0], o.pos[2]) for o in self.obstacles]
        def clear(x, z, d=2.0):
            return all(math.hypot(x-ox,z-oz) > d for ox,oz in obs_c)

        # Flores (muitas! jardim florido)
        self.flowers: List[Flower] = []
        for _ in range(55):
            for _try in range(20):
                x = random.uniform(-20, 20); z = random.uniform(-20, 20)
                if clear(x,z) and (abs(x)>2.2 or abs(z)>2.2):
                    self.flowers.append(Flower(x, z)); break

        # Cogumelos perto das árvores
        self.mushrooms: List[Mushroom] = []
        for tree in self.trees[:10]:
            for _ in range(random.randint(1, 2)):
                a = random.uniform(0, 2*math.pi)
                r = random.uniform(1.2, 2.2)
                mx = tree.pos[0] + math.cos(a)*r
                mz = tree.pos[2] + math.sin(a)*r
                if abs(mx) < 20.5 and abs(mz) < 20.5:
                    self.mushrooms.append(Mushroom(mx, mz))

        # Lanternas ao longo do caminho central
        self.lanterns: List[Lantern] = []
        for d in [-15, -9, 9, 15]:
            self.lanterns.append(Lantern(d, 2.6))
            self.lanterns.append(Lantern(2.6, d))

        # Arbustos redondinhos
        self.bushes: List[Bush] = []
        for _ in range(14):
            for _try in range(20):
                x = random.uniform(-19, 19); z = random.uniform(-19, 19)
                if clear(x, z, 2.6) and (abs(x)>3.5 or abs(z)>3.5):
                    self.bushes.append(Bush(x, z)); break

        # Cerquinha branca decorativa (esparsa, perto dos muros)
        self.fences: List[Fence] = []
        for x in range(-15, 16, 6):
            self.fences.append(Fence(x, WORLD_Z[0]+2.2, 0))
            self.fences.append(Fence(x, WORLD_Z[1]-2.2, 0))
        for z in range(-15, 16, 6):
            self.fences.append(Fence(WORLD_X[0]+2.2, z, math.pi/2))
            self.fences.append(Fence(WORLD_X[1]-2.2, z, math.pi/2))

    def _spawn_collectibles(self, n_fish, n_stars):
        self.fishes: List[Fish] = []
        obs_c = [(o.pos[0], o.pos[2]) for o in self.obstacles]
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

    @property
    def all_obstacle_aabbs(self) -> List[AABB]:
        return ([o.aabb for o in self.obstacles] +
                [t.aabb for t in self.trees] +
                [h.aabb for h in self.houses] +
                self.wall_aabbs)

    def update(self, dt):
        for f in self.flowers:
            f.update(dt)

    def render(self, renderer: Renderer, view):
        for mesh, M in self._floor_tiles:
            renderer.render_mesh(mesh, M, view)
        for fence in self.fences:
            renderer.render_mesh(fence.mesh, fence.model_matrix(), view)
        for fl in self.flowers:
            renderer.render_mesh(fl.mesh, fl.model_matrix(), view)
        for m in self.mushrooms:
            renderer.render_mesh(m.mesh, m.model_matrix(), view)
        for b in self.bushes:
            renderer.render_mesh(b.mesh, b.model_matrix(), view)
        for l in self.lanterns:
            renderer.render_mesh(l.mesh, l.model_matrix(), view)
        for h in self.houses:
            renderer.render_mesh(h.mesh, h.model_matrix(), view)
        for t in self.trees:
            renderer.render_mesh(t.mesh, t.model_matrix(), view)
        for mesh, M in self._wall_data:
            renderer.render_mesh(mesh, M, view)
        for o in self.obstacles:
            renderer.render_mesh(o.mesh, o.model_matrix(), view)

    def render_collectibles(self, renderer: Renderer, view):
        for fish in self.fishes:
            if not fish.collected or fish.collect_anim < 1.2:
                tint = (255,255,255) if fish.collect_anim > 0 else None
                renderer.render_mesh(fish.mesh, fish.model_matrix(),
                                     view, tint=tint)
