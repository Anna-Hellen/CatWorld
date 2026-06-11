"""
main.py – Game Loop Principal (RT-05) com melhorias gráficas e de movimentação
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pygame
import numpy as np
import math
import random

from renderer import Renderer
from camera import Camera
from entities import Cat
from world import World, WORLD_X, WORLD_Z
from hud import HUD
from sound import SoundManager

WIDTH, HEIGHT = 1280, 720
FPS_TARGET    = 60
TITLE         = "CatWorld – Computação Gráfica"

PHASES = [
    {"time": 120, "fish": 15, "stars": 3, "seed": 42,   "name": "Fase 1 – O Jardim"},
    {"time":  90, "fish": 18, "stars": 4, "seed": 1337, "name": "Fase 2 – O Labirinto"},
    {"time":  70, "fish": 22, "stars": 5, "seed": 9999, "name": "Fase 3 – O Desafio Final"},
]


class GameState:
    START = "start"; PLAYING = "playing"; PAUSE = "pause"
    END = "end";     WIN_ALL = "win_all"


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen   = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock    = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.hud      = HUD(self.screen)
        self.sound    = SoundManager()
        self.sound.init()
        self.phase_idx   = 0
        self.total_score = 0
        self._init_phase()

    def _init_phase(self):
        cfg = PHASES[self.phase_idx]
        self.state       = GameState.START if self.phase_idx == 0 else GameState.PLAYING
        self.time_left   = float(cfg["time"])
        self.time_used   = 0.0
        self.score       = 0
        self.combo       = 1
        self.combo_timer = 0.0
        self.won         = False
        self.step_timer  = 0.0
        self.world  = World(seed=cfg["seed"], n_fish=cfg["fish"], n_stars=cfg["stars"])
        self.cat    = Cat(x=0, z=0)
        self.camera = Camera(WIDTH, HEIGHT)
        self.total_fishes = len(self.world.fishes)
        if self.state == GameState.PLAYING:
            self.sound.start_ambient()

    def run(self):
        while True:
            dt = self.clock.tick(FPS_TARGET) / 1000.0
            dt = min(dt, 0.05)
            self.handle_events()
            self.update(dt)
            self.render()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state == GameState.PLAYING:
                        self.state = GameState.PAUSE
                        self.sound.stop_ambient()
                    elif self.state == GameState.PAUSE:
                        self.state = GameState.PLAYING
                        self.sound.start_ambient()
                    else:
                        pygame.quit(); sys.exit()
                if event.key == pygame.K_SPACE:
                    if self.state == GameState.START:
                        self.state = GameState.PLAYING
                        self.sound.start_ambient()
                    elif self.state == GameState.PAUSE:
                        self.state = GameState.PLAYING
                        self.sound.start_ambient()
                    elif self.state == GameState.END:
                        self.sound.stop_ambient()
                        self._init_phase()
                        self.state = GameState.PLAYING
                        self.sound.start_ambient()
                    elif self.state == GameState.WIN_ALL:
                        self.phase_idx = 0
                        self.total_score = 0
                        self.sound.stop_ambient()
                        self._init_phase()
                if event.key == pygame.K_RETURN:
                    if self.state == GameState.END and self.won:
                        self._advance_phase()

    def _advance_phase(self):
        self.total_score += self.score
        self.phase_idx   += 1
        self.sound.stop_ambient()
        if self.phase_idx >= len(PHASES):
            self.state = GameState.WIN_ALL
        else:
            self._init_phase()

    def update(self, dt):
        self.hud.update(dt)
        if self.state == GameState.PLAYING:
            self._update_playing(dt)

    def _update_playing(self, dt):
        self.time_left  -= dt
        self.time_used  += dt
        if self.time_left <= 0:
            self.time_left = 0
            self.state = GameState.END
            self.won   = False
            self.sound.stop_ambient()
            self.sound.play('lose')
            return

        self.combo_timer -= dt
        if self.combo_timer <= 0:
            self.combo = 1

        # Gato
        keys = pygame.key.get_pressed()
        self.cat.update(dt, keys,
                        (WORLD_X[0], WORLD_X[1], WORLD_Z[0], WORLD_Z[1]),
                        self.world.all_obstacle_aabbs)

        # Som de passos
        if self.cat.speed > 1.0:
            self.step_timer -= dt
            if self.step_timer <= 0:
                self.step_timer = 0.26 - self.cat.speed * 0.01
                self.sound.play('step')

        # Câmera com física (passa todos os parâmetros necessários)
        self.camera.update(
            self.cat.pos, self.cat.yaw,
            self.cat.speed, self.cat.yaw_rate,
            self.cat.dashing, self.cat.impact_strength,
            dt
        )

        # Mundo (flores)
        self.world.update(dt)

        # Coletáveis
        for fish in self.world.fishes:
            fish.update(dt)

        # Colisão gato × peixes
        cat_sphere = self.cat.sphere
        for fish in self.world.fishes:
            if fish.collected: continue
            if cat_sphere.intersects_sphere(fish.sphere):
                fish.collected = True
                self.cat.collect_bounce()
                pts = fish.points * self.combo
                self.score      += pts
                self.combo      += 1
                self.combo_timer = 2.5
                self.camera.add_shake(0.06 if not fish.is_star else 0.12)
                if fish.is_star:
                    self.sound.play('star')
                else:
                    self.sound.play('fish')
                if self.combo > 2:
                    self.sound.play('combo')

        # Vitória
        if all(f.collected for f in self.world.fishes):
            self.state = GameState.END
            self.won   = True
            self.score += int(self.time_left * 5)
            self.sound.stop_ambient()
            self.sound.play('win')
            self.camera.add_shake(0.2)

    def render(self):
        self._draw_sky()
        cfg = PHASES[self.phase_idx]

        # Atualiza proj no renderer a cada frame (FOV dinâmico)
        self.renderer.set_proj(self.camera.proj_matrix())

        if self.state == GameState.START:
            self._render_scene()
            self.hud.draw_start_screen(cfg["name"])
        elif self.state == GameState.PLAYING:
            self._render_scene()
            remaining = sum(1 for f in self.world.fishes if not f.collected)
            self.hud.draw_game(self.score, remaining, self.total_fishes,
                               self.time_left, self.combo,
                               self.phase_idx+1, len(PHASES))
            self.hud.draw_minimap(self.cat.pos, self.world.fishes, WORLD_X, WORLD_Z)
            # Indicador de dash
            if self.cat.dash_cooldown > 0:
                self._draw_dash_bar()
        elif self.state == GameState.PAUSE:
            self._render_scene()
            self.hud.draw_pause()
        elif self.state == GameState.END:
            self._render_scene()
            self.hud.draw_end_screen(self.score, self.won, self.time_used,
                                     self.phase_idx+1, len(PHASES))
        elif self.state == GameState.WIN_ALL:
            self._render_scene()
            self.hud.draw_win_all(self.total_score + self.score)

        pygame.display.flip()

    def _render_scene(self):
        view = self.camera.view_matrix()
        self.world.render(self.renderer, view)
        self.world.render_collectibles(self.renderer, view)
        self.renderer.render_mesh(self.cat.mesh, self.cat.model_matrix(), view)

    def _draw_dash_bar(self):
        """Barra de cooldown do dash – canto inferior."""
        pct = 1.0 - self.cat.dash_cooldown / self.cat.DASH_COOLDOWN
        bx, by, bw, bh = 10, HEIGHT - 50, 120, 12
        pygame.draw.rect(self.screen, (40, 40, 60), (bx, by, bw, bh), border_radius=4)
        if pct > 0:
            fill_c = (80, 160, 255) if pct < 1.0 else (150, 220, 255)
            pygame.draw.rect(self.screen, fill_c,
                             (bx, by, int(bw*pct), bh), border_radius=4)
        pygame.draw.rect(self.screen, (100, 120, 180), (bx, by, bw, bh), 1, border_radius=4)
        font = pygame.font.SysFont("Segoe UI", 14)
        lbl = font.render("DASH [SHIFT]" if pct >= 1.0 else "...", True,
                          (200,220,255) if pct >= 1.0 else (120,140,180))
        self.screen.blit(lbl, (bx, by - 16))

    def _draw_sky(self):
        """Gradiente de céu com nuvens simples."""
        SKY_TOP = (255, 205, 170); SKY_BOT = (190, 222, 240)  # pôr-do-sol pastel
        # Gradiente (passo 3 linhas para performance)
        for y in range(0, HEIGHT, 3):
            t = y / HEIGHT
            r = int(SKY_TOP[0] + (SKY_BOT[0]-SKY_TOP[0]) * t)
            g = int(SKY_TOP[1] + (SKY_BOT[1]-SKY_TOP[1]) * t)
            b = int(SKY_TOP[2] + (SKY_BOT[2]-SKY_TOP[2]) * t)
            pygame.draw.line(self.screen, (r,g,b), (0,y), (WIDTH,y))
            pygame.draw.line(self.screen, (r,g,b), (0,y+1), (WIDTH,y+1))
            pygame.draw.line(self.screen, (r,g,b), (0,y+2), (WIDTH,y+2))

        # Nuvens (elipses simples na parte superior)
        cloud_c = (255, 240, 230)
        for cx, cy, rw, rh in [
            (180, 70, 75, 28), (320, 50, 55, 22),
            (700, 80, 90, 32), (950, 55, 65, 24),
            (1150,72, 70, 26),
        ]:
            pygame.draw.ellipse(self.screen, cloud_c,
                                (cx-rw, cy-rh, rw*2, rh*2))
            pygame.draw.ellipse(self.screen, (255,248,242),
                                (cx-rw+8, cy-rh+6, rw*2-16, rh*2-12))


if __name__ == "__main__":
    Game().run()
