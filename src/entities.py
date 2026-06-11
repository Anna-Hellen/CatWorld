"""
entities.py – Entidades com movimentação arcade precisa (RT-02, RF-01)

Filosofia da movimentação (responsiva e fluida):
  - Aceleração rápida (~0.18s até velocidade máxima) → resposta imediata
  - Frenagem rápida (~0.12s até parar) → controle preciso
  - Giro direto com leve suavização → sem "deriva" de inércia angular
  - Colisão por DESLIZAMENTO: remove só o componente da velocidade
    na direção do obstáculo → o gato desliza ao longo de paredes
    em vez de quicar ou travar
  - Dash curto (SHIFT) com cooldown
  - Animações sutis: bob de caminhada, inclinação na curva, squash na coleta
"""

import numpy as np
import math
import random
from typing import List

from math3d import (translation_matrix, rotation_y, rotation_z,
                    rotation_x, scale_matrix)
from geometry import (cat_body_mesh, fish_mesh, star_mesh, tree_mesh,
                      house_mesh, crate_mesh, stone_mesh, flower_mesh,
                      fence_segment_mesh, mushroom_mesh, lantern_mesh,
                      bush_mesh)
from collision import AABB, Sphere, make_aabb, make_sphere


class Cat:
    """
    Agente com controle arcade preciso.

    Transformação composta (RT-02):
      M = T(pos+bob) · Ry(yaw) · Rz(lean) · Rx(pitch) · S(squash)
    """

    MAX_SPEED   = 9.0
    ACCEL_TIME  = 0.18    # segundos até velocidade máxima
    BRAKE_TIME  = 0.12    # segundos até parar
    TURN_SPEED  = math.pi * 1.6   # rad/s
    TURN_SMOOTH = 18.0    # suavização do giro (alto = responsivo)
    DASH_SPEED  = 17.0
    DASH_TIME   = 0.16
    DASH_CD     = 1.0

    def __init__(self, x=0.0, z=0.0):
        self.pos      = np.array([x, 0.80, z], dtype=np.float64)
        self.vel      = np.array([0.0, 0.0, 0.0])
        self.yaw      = 0.0
        self.yaw_vel  = 0.0      # taxa de giro suavizada
        self.radius   = 0.58

        # Animação
        self.bob_time = 0.0
        self.lean     = 0.0
        self.pitch    = 0.0
        self.squash   = 1.0

        # Dash
        self.dash_t   = 0.0
        self.dash_cd  = 0.0
        self.dashing  = False

        # Para feedback (câmera)
        self.bump     = 0.0   # intensidade de impacto recente

        self.mesh  = cat_body_mesh()
        self.scale = 0.58

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
        # Respiração sutil quando parado
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
        """Pequeno pulo de alegria ao coletar (squash & stretch)."""
        self.squash = 1.22

    def update(self, dt, keys, world_bounds, obstacles: List[AABB]):
        import pygame

        # ── Entrada ───────────────────────────────────────────────────────────
        turn = 0.0
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: turn += 1.0
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: turn -= 1.0

        fwd = 0.0
        if keys[pygame.K_UP]   or keys[pygame.K_w]: fwd += 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]: fwd -= 0.55

        dash_key = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]

        # ── Giro: direto com leve suavização (responsivo, sem deriva) ─────────
        target_yaw_vel = turn * self.TURN_SPEED
        blend = min(1.0, self.TURN_SMOOTH * dt)
        self.yaw_vel += (target_yaw_vel - self.yaw_vel) * blend
        self.yaw += self.yaw_vel * dt

        # ── Dash ─────────────────────────────────────────────────────────────
        self.dash_cd = max(0.0, self.dash_cd - dt)
        if self.dash_t > 0:
            self.dash_t -= dt
            self.dashing = self.dash_t > 0
        elif dash_key and self.dash_cd <= 0 and abs(fwd) > 0:
            self.dash_t  = self.DASH_TIME
            self.dash_cd = self.DASH_CD
            self.dashing = True
            self.squash  = 0.80

        # ── Velocidade: aceleração/frenagem rápidas (arcade) ─────────────────
        fwd_dir = np.array([math.sin(self.yaw), 0.0, math.cos(self.yaw)])
        max_spd = self.DASH_SPEED if self.dashing else self.MAX_SPEED
        target_vel = fwd_dir * fwd * max_spd
        if self.dashing:
            target_vel = fwd_dir * max_spd  # dash sempre pra frente

        # Tempo de resposta diferente para acelerar vs frear
        accelerating = np.dot(target_vel, target_vel) > np.dot(
            np.array([self.vel[0],0,self.vel[2]]),
            np.array([self.vel[0],0,self.vel[2]]))
        response = self.ACCEL_TIME if accelerating else self.BRAKE_TIME
        alpha = min(1.0, dt / response)

        self.vel[0] += (target_vel[0] - self.vel[0]) * alpha
        self.vel[2] += (target_vel[2] - self.vel[2]) * alpha

        # ── Movimento com colisão por DESLIZAMENTO ────────────────────────────
        step = self.vel * dt
        new_pos = self.pos + step

        # Paredes do mundo: desliza (zera só o eixo que colide)
        xmin, xmax, zmin, zmax = world_bounds
        margin = 0.85
        if new_pos[0] < xmin + margin:
            new_pos[0] = xmin + margin
            if self.vel[0] < 0:
                self.bump = max(self.bump, abs(self.vel[0])*0.4)
                self.vel[0] = 0.0
        elif new_pos[0] > xmax - margin:
            new_pos[0] = xmax - margin
            if self.vel[0] > 0:
                self.bump = max(self.bump, abs(self.vel[0])*0.4)
                self.vel[0] = 0.0
        if new_pos[2] < zmin + margin:
            new_pos[2] = zmin + margin
            if self.vel[2] < 0:
                self.bump = max(self.bump, abs(self.vel[2])*0.4)
                self.vel[2] = 0.0
        elif new_pos[2] > zmax - margin:
            new_pos[2] = zmax - margin
            if self.vel[2] > 0:
                self.bump = max(self.bump, abs(self.vel[2])*0.4)
                self.vel[2] = 0.0

        # Obstáculos: resolve por eixo separado → desliza naturalmente
        # (testa movimento em X e Z independentemente)
        test = make_sphere((new_pos[0], self.pos[1], self.pos[2]), self.radius)
        blocked_x = any(test.intersects_aabb(o) for o in obstacles)
        test = make_sphere((self.pos[0], self.pos[1], new_pos[2]), self.radius)
        blocked_z = any(test.intersects_aabb(o) for o in obstacles)

        if blocked_x:
            self.bump = max(self.bump, abs(self.vel[0]) * 0.3)
            new_pos[0] = self.pos[0]
            self.vel[0] = 0.0
        if blocked_z:
            self.bump = max(self.bump, abs(self.vel[2]) * 0.3)
            new_pos[2] = self.pos[2]
            self.vel[2] = 0.0

        # Caso raro: preso na diagonal → empurra para fora
        test = make_sphere(tuple(new_pos), self.radius)
        for o in obstacles:
            if test.intersects_aabb(o):
                c = o.center()
                diff = new_pos - c; diff[1] = 0
                d = np.linalg.norm(diff)
                if d > 0.001:
                    new_pos += (diff/d) * 0.08

        self.pos = new_pos
        self.bump *= math.pow(0.05, dt)   # decai rápido

        # ── Animações ─────────────────────────────────────────────────────────
        spd_t = min(1.0, self.speed / self.MAX_SPEED)
        self.bob_time += dt * (0.4 + spd_t)

        # Inclina na curva (sutil)
        target_lean = -self.yaw_vel * 0.05 * spd_t
        self.lean += (target_lean - self.lean) * min(1.0, dt * 14)

        # Inclina o nariz pra baixo levemente ao correr (gato focado!)
        target_pitch = spd_t * 0.06 + (0.12 if self.dashing else 0.0)
        self.pitch += (target_pitch - self.pitch) * min(1.0, dt * 10)

        # Squash volta ao normal
        self.squash += (1.0 - self.squash) * min(1.0, dt * 12)


class Fish:
    def __init__(self, x, z, is_star=False):
        self.pos          = np.array([x, 1.25, z], dtype=np.float64)
        self.rot_y        = random.uniform(0, math.pi*2)
        self.time         = random.uniform(0, math.pi*2)
        self.collected    = False
        self.is_star      = is_star
        self.points       = 50 if is_star else 10
        self.collect_anim = 0.0
        self.mesh         = star_mesh() if is_star else fish_mesh()
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


class Obstacle:
    def __init__(self, x, z, w=1.3, h=1.4, d=1.3):
        self.pos = np.array([x, h/2, z], dtype=np.float64)
        self.w, self.h, self.d = w, h, d
        self.mesh = stone_mesh(w,h,d) if h < 1.0 else crate_mesh(w,h,d)

    @property
    def aabb(self):
        return make_aabb(tuple(self.pos), (self.w/2, self.h/2, self.d/2))

    def model_matrix(self):
        return translation_matrix(self.pos[0], 0, self.pos[2])


class Tree:
    def __init__(self, x, z, scale=1.0, pink=False):
        self.pos     = np.array([x, 0, z], dtype=np.float64)
        self.scale_v = scale
        self.mesh    = tree_mesh(pink=pink)
        self.yaw     = random.uniform(0, math.pi*2)

    def model_matrix(self):
        T = translation_matrix(self.pos[0], 0, self.pos[2])
        return T @ rotation_y(self.yaw) @ scale_matrix(
            self.scale_v, self.scale_v, self.scale_v)

    @property
    def aabb(self):
        return make_aabb((self.pos[0], 1.5, self.pos[2]),
                         (0.55*self.scale_v, 3.5, 0.55*self.scale_v))


class House:
    def __init__(self, x, z, yaw=0):
        self.pos, self.yaw = np.array([x,0,z],dtype=np.float64), yaw
        self.mesh = house_mesh()

    def model_matrix(self):
        return translation_matrix(*self.pos) @ rotation_y(self.yaw)

    @property
    def aabb(self):
        return make_aabb((self.pos[0], 1.5, self.pos[2]), (1.9, 3.5, 1.6))


class Flower:
    COLORS = [(245,170,185),(255,215,130),(205,170,235),
              (252,245,235),(255,180,150),(180,215,250)]

    def __init__(self, x, z):
        self.pos  = np.array([x, 0, z], dtype=np.float64)
        self.yaw  = random.uniform(0, math.pi*2)
        self.time = random.uniform(0, math.pi*2)
        self.mesh = flower_mesh(random.choice(self.COLORS))
        self.scale_v = random.uniform(0.85, 1.3)

    def model_matrix(self):
        sway = math.sin(self.time * 0.9) * 0.06
        return (translation_matrix(*self.pos) @ rotation_y(self.yaw)
                @ rotation_z(sway)
                @ scale_matrix(self.scale_v, self.scale_v, self.scale_v))

    def update(self, dt):
        self.time += dt


class Mushroom:
    def __init__(self, x, z):
        self.pos  = np.array([x, 0, z], dtype=np.float64)
        self.yaw  = random.uniform(0, math.pi*2)
        self.mesh = mushroom_mesh(random.uniform(0.8, 1.4))

    def model_matrix(self):
        return translation_matrix(*self.pos) @ rotation_y(self.yaw)


class Lantern:
    def __init__(self, x, z):
        self.pos  = np.array([x, 0, z], dtype=np.float64)
        self.mesh = lantern_mesh()

    def model_matrix(self):
        return translation_matrix(*self.pos)


class Bush:
    def __init__(self, x, z):
        self.pos  = np.array([x, 0, z], dtype=np.float64)
        self.yaw  = random.uniform(0, math.pi*2)
        self.mesh = bush_mesh(random.uniform(0.8, 1.5))

    def model_matrix(self):
        return translation_matrix(*self.pos) @ rotation_y(self.yaw)


class Fence:
    def __init__(self, x, z, yaw=0):
        self.pos, self.yaw = np.array([x,0,z],dtype=np.float64), yaw
        self.mesh = fence_segment_mesh(3.0)

    def model_matrix(self):
        return translation_matrix(*self.pos) @ rotation_y(self.yaw)
