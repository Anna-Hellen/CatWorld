"""
camera.py – Câmera em terceira pessoa estável e suave (RT-04)

Seguimento exponencial simples e bem calibrado:
  - Sem oscilação de mola → estável e previsível
  - Velocidades de seguimento separadas (posição rápida, rotação suave)
  - FOV dinâmico sutil no dash
  - Shake apenas em momentos-chave (curto e suave)
"""

import numpy as np
import math
import random
from math3d import look_at, perspective_projection


class Camera:
    UP = np.array([0.0, 1.0, 0.0])
    DIST   = 6.8
    HEIGHT = 4.4

    # Velocidades de seguimento (quanto maior, mais rápido converge)
    FOLLOW_POS  = 9.0
    FOLLOW_TGT  = 14.0

    def __init__(self, width=1280, height=720):
        self.eye    = np.array([0.0, self.HEIGHT, -self.DIST])
        self.target = np.array([0.0, 0.6, 0.0])
        self.width, self.height = width, height

        self.fov_base = math.radians(58)
        self.fov      = self.fov_base

        self.shake_mag = 0.0
        self._shake    = np.zeros(3)

    def add_shake(self, mag):
        self.shake_mag = max(self.shake_mag, mag)

    def update(self, agent_pos, agent_yaw, agent_speed,
               agent_yaw_rate, dashing, impact_strength, dt):

        # Posição ideal: atrás do gato na direção do yaw
        ideal = np.array([
            agent_pos[0] - math.sin(agent_yaw) * self.DIST,
            agent_pos[1] + self.HEIGHT,
            agent_pos[2] - math.cos(agent_yaw) * self.DIST,
        ])

        # Seguimento exponencial estável (frame-rate independent)
        a_pos = 1.0 - math.exp(-self.FOLLOW_POS * dt)
        self.eye += (ideal - self.eye) * a_pos

        # Alvo: mira no gato (um pouco acima e à frente)
        look_ahead = 0.8 * min(1.0, agent_speed / 9.0)
        t_ideal = agent_pos + np.array([
            math.sin(agent_yaw) * look_ahead,
            0.6,
            math.cos(agent_yaw) * look_ahead,
        ])
        a_tgt = 1.0 - math.exp(-self.FOLLOW_TGT * dt)
        self.target += (t_ideal - self.target) * a_tgt

        # FOV: sutil no dash
        fov_target = self.fov_base + (math.radians(6) if dashing else 0)
        self.fov += (fov_target - self.fov) * min(1.0, dt * 8)

        # Shake (curto)
        if impact_strength > 2.0:
            self.shake_mag = max(self.shake_mag, min(0.10, impact_strength*0.02))
        self.shake_mag *= math.pow(0.0003, dt)   # decai muito rápido
        if self.shake_mag > 0.004:
            self._shake = np.array([
                (random.random()-0.5)*2*self.shake_mag,
                (random.random()-0.5)*self.shake_mag,
                0.0])
        else:
            self._shake = np.zeros(3)

    def view_matrix(self):
        return look_at(self.eye + self._shake, self.target, self.UP)

    def proj_matrix(self):
        return perspective_projection(self.fov, self.width/self.height,
                                      0.5, 500.0)
