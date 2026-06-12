"""
entities.py – Entidades dinâmicas com meshes COMPARTILHADAS (RT-02, RF-01)

Otimização: todos os peixes compartilham UMA malha pré-processada
(antes: cada peixe criava sua própria cópia — 18× a memória e o
tempo de inicialização). O gato pré-processa sua malha uma vez.
A movimentação arcade precisa com deslizamento é mantida da v3.
"""

import numpy as np
import math
import random
from typing import List

from math3d import (translation_matrix, rotation_y, rotation_z,
                    rotation_x, scale_matrix)
from geometry import cat_body_mesh, fish_mesh, star_mesh
from collision import AABB, make_sphere
from renderer import PreparedMesh

# ── Meshes compartilhadas (criadas uma única vez) ─────────────────────────────
_FISH_PREP = None
_STAR_PREP = None

def get_fish_prep():
    global _FISH_PREP
    if _FISH_PREP is None:
        _FISH_PREP = PreparedMesh(fish_mesh())
    return _FISH_PREP

def get_star_prep():
    global _STAR_PREP
    if _STAR_PREP is None:
        _STAR_PREP = PreparedMesh(star_mesh())
    return _STAR_PREP


class Cat:
    """Agente com controle arcade preciso (deslizamento em colisões)."""

    MAX_SPEED   = 9.0
    ACCEL_TIME  = 0.18
    BRAKE_TIME  = 0.12
    TURN_SPEED  = math.pi * 1.6
    TURN_SMOOTH = 18.0
    DASH_SPEED  = 17.0
    DASH_TIME   = 0.16
    DASH_CD     = 1.0

    def __init__(self, x=0.0, z=0.0):
        self.pos      = np.array([x, 0.80, z], dtype=np.float64)
        self.vel      = np.array([0.0, 0.0, 0.0])
        self.yaw      = 0.0
        self.yaw_vel  = 0.0
        self.radius   = 0.58
        self.bob_time = 0.0
        self.lean     = 0.0
        self.pitch    = 0.0
        self.squash   = 1.0
        self.dash_t   = 0.0
        self.dash_cd  = 0.0
        self.dashing  = False
        self.bump     = 0.0
        self.prep     = PreparedMesh(cat_body_mesh())
        self.scale    = 0.58

    @property
    def sphere(self):
        return make_sphere(tuple(self.pos), self.radius)

    @property
    def speed(self):
        return math.hypot(self.vel[0], self.vel[2])

    @property
    def dash_cooldown(self):
        return self.dash_cd

    @property
    def DASH_COOLDOWN(self):
        return self.DASH_CD

    @property
    def yaw_rate(self):
        return self.yaw_vel

    @property
    def impact_strength(self):
        return self.bump

    def model_matrix(self):
        spd_t = min(1.0, self.speed / self.MAX_SPEED)
        bob_y = math.sin(self.bob_time * 10.0) * 0.05 * spd_t
        breath = math.sin(self.bob_time * 2.0) * 0.015 * (1.0 - spd_t)
        T  = translation_matrix(self.pos[0], self.pos[1] + bob_y + breath,
                                self.pos[2])
        Ry = rotation_y(self.yaw)
        Rz = rotation_z(self.lean)
        Rx = rotation_x(self.pitch)
        sy = self.squash
        sxz = 1.0 / math.sqrt(max(0.6, sy))
        S  = scale_matrix(self.scale*sxz, self.scale*sy, self.scale*sxz)
        return T @ Ry @ Rz @ Rx @ S

    def collect_bounce(self):
        self.squash = 1.22

    def update(self, dt, keys, world_bounds, obstacles: List[AABB]):
        import pygame

        turn = 0.0
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: turn += 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: turn -= 1.0
        fwd = 0.0
        if keys[pygame.K_UP]   or keys[pygame.K_w]: fwd += 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: fwd -= 0.55
        dash_key = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        target_yaw_vel = turn * self.TURN_SPEED
        blend = min(1.0, self.TURN_SMOOTH * dt)
        self.yaw_vel += (target_yaw_vel - self.yaw_vel) * blend
        self.yaw += self.yaw_vel * dt

        self.dash_cd = max(0.0, self.dash_cd - dt)
        if self.dash_t > 0:
            self.dash_t -= dt
            self.dashing = self.dash_t > 0
        elif dash_key and self.dash_cd <= 0 and abs(fwd) > 0:
            self.dash_t  = self.DASH_TIME
            self.dash_cd = self.DASH_CD
            self.dashing = True
            self.squash  = 0.80

        fwd_dir = np.array([math.sin(self.yaw), 0.0, math.cos(self.yaw)])
        max_spd = self.DASH_SPEED if self.dashing else self.MAX_SPEED
        target_vel = fwd_dir * (max_spd if self.dashing else fwd * max_spd)

        cur_sq = self.vel[0]**2 + self.vel[2]**2
        tgt_sq = target_vel[0]**2 + target_vel[2]**2
        response = self.ACCEL_TIME if tgt_sq > cur_sq else self.BRAKE_TIME
        alpha = min(1.0, dt / response)
        self.vel[0] += (target_vel[0] - self.vel[0]) * alpha
        self.vel[2] += (target_vel[2] - self.vel[2]) * alpha

        new_pos = self.pos + self.vel * dt

        xmin, xmax, zmin, zmax = world_bounds
        margin = 0.85
        if new_pos[0] < xmin + margin:
            new_pos[0] = xmin + margin
            if self.vel[0] < 0:
                self.bump = max(self.bump, abs(self.vel[0])*0.4); self.vel[0] = 0.0
        elif new_pos[0] > xmax - margin:
            new_pos[0] = xmax - margin
            if self.vel[0] > 0:
                self.bump = max(self.bump, abs(self.vel[0])*0.4); self.vel[0] = 0.0
        if new_pos[2] < zmin + margin:
            new_pos[2] = zmin + margin
            if self.vel[2] < 0:
                self.bump = max(self.bump, abs(self.vel[2])*0.4); self.vel[2] = 0.0
        elif new_pos[2] > zmax - margin:
            new_pos[2] = zmax - margin
            if self.vel[2] > 0:
                self.bump = max(self.bump, abs(self.vel[2])*0.4); self.vel[2] = 0.0

        # Deslizamento: testa eixos separadamente
        test = make_sphere((new_pos[0], self.pos[1], self.pos[2]), self.radius)
        blocked_x = any(test.intersects_aabb(o) for o in obstacles)
        test = make_sphere((self.pos[0], self.pos[1], new_pos[2]), self.radius)
        blocked_z = any(test.intersects_aabb(o) for o in obstacles)
        if blocked_x:
            self.bump = max(self.bump, abs(self.vel[0]) * 0.3)
            new_pos[0] = self.pos[0]; self.vel[0] = 0.0
        if blocked_z:
            self.bump = max(self.bump, abs(self.vel[2]) * 0.3)
            new_pos[2] = self.pos[2]; self.vel[2] = 0.0

        test = make_sphere(tuple(new_pos), self.radius)
        for o in obstacles:
            if test.intersects_aabb(o):
                c = o.center()
                diff = new_pos - c; diff[1] = 0
                d = np.linalg.norm(diff)
                if d > 0.001:
                    new_pos += (diff/d) * 0.08

        self.pos = new_pos
        self.bump *= math.pow(0.05, dt)

        spd_t = min(1.0, self.speed / self.MAX_SPEED)
        self.bob_time += dt * (0.4 + spd_t)
        target_lean = -self.yaw_vel * 0.05 * spd_t
        self.lean += (target_lean - self.lean) * min(1.0, dt * 14)
        target_pitch = spd_t * 0.06 + (0.12 if self.dashing else 0.0)
        self.pitch += (target_pitch - self.pitch) * min(1.0, dt * 10)
        self.squash += (1.0 - self.squash) * min(1.0, dt * 12)


class Fish:
    """Coletável. Compartilha a malha pré-processada com todos os peixes."""

    def __init__(self, x, z, is_star=False):
        self.pos          = np.array([x, 1.25, z], dtype=np.float64)
        self.rot_y        = random.uniform(0, math.pi*2)
        self.time         = random.uniform(0, math.pi*2)
        self.collected    = False
        self.is_star      = is_star
        self.points       = 50 if is_star else 10
        self.collect_anim = 0.0
        self.prep         = get_star_prep() if is_star else get_fish_prep()
        self.base_scale   = 1.0 if is_star else 0.95

    def model_matrix(self):
        y_bob = self.pos[1] + math.sin(self.time * 1.8) * 0.16
        wobble = math.sin(self.time * 1.1) * 0.10
        T  = translation_matrix(self.pos[0], y_bob, self.pos[2])
        Ry = rotation_y(self.rot_y)
        Rz = rotation_z(wobble)
        sc = self.base_scale
        if self.collect_anim > 0:
            sc *= (1.0 + self.collect_anim * 2.2)
        S = scale_matrix(sc, sc, sc)
        return T @ Ry @ Rz @ S

    @property
    def sphere(self):
        return make_sphere(tuple(self.pos), 0.78)

    def update(self, dt):
        if not self.collected:
            self.time  += dt
            self.rot_y += dt * (2.0 if self.is_star else 1.3)
        else:
            self.collect_anim += dt * 3.0
