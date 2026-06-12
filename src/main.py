"""
main.py – Game Loop (RT-05) com otimizações de desempenho

Otimizações neste arquivo:
  - Céu + nuvens PRÉ-RENDERIZADOS em uma Surface (antes: ~480 draw.line
    + 10 ellipses por frame; agora: 1 blit)
  - Resolução interna escalável (render scale): o 3D é rasterizado em
    resolução menor e ampliado — o custo de preenchimento de polígonos
    cai com o quadrado da escala
  - Presets de qualidade (F1): Baixa 60% / Média 80% / Alta 100%,
    afetando resolução interna e densidade de decoração
  - Contador de FPS (F3) para demonstrar o desempenho na apresentação
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import pygame
import numpy as np
import math

from renderer import Renderer
from camera import Camera
from entities import Cat
from world import World, WORLD_X, WORLD_Z
from hud import HUD
from sound import SoundManager

WIDTH, HEIGHT = 1280, 720
FPS_TARGET    = 60
TITLE         = "CatWorld – Computação Gráfica"

# (nome, escala de resolução interna, densidade de decoração)
QUALITIES = [
    ("Baixa", 0.60, 0.55),
    ("Média", 0.80, 0.85),
    ("Alta",  1.00, 1.00),
]

PHASES = [
    {"time": 120, "fish": 15, "stars": 3, "seed": 42,   "name": "Fase 1 – O Jardim"},
    {"time":  90, "fish": 18, "stars": 4, "seed": 1337, "name": "Fase 2 – O Labirinto"},
    {"time":  70, "fish": 22, "stars": 5, "seed": 9999, "name": "Fase 3 – O Desafio Final"},
]


class GameState:
    START = "start"; PLAYING = "playing"; PAUSE = "pause"
    END = "end";     WIN_ALL = "win_all"


def bake_sky(w, h):
    """Pré-renderiza o gradiente de céu + nuvens (1× por mudança de escala)."""
    SKY_TOP = (255, 205, 170); SKY_BOT = (190, 222, 240)
    surf = pygame.Surface((w, h))
    for y in range(h):
        t = y / h
        surf.fill((int(SKY_TOP[0] + (SKY_BOT[0]-SKY_TOP[0])*t),
                   int(SKY_TOP[1] + (SKY_BOT[1]-SKY_TOP[1])*t),
                   int(SKY_TOP[2] + (SKY_BOT[2]-SKY_TOP[2])*t)),
                  (0, y, w, 1))
    sx = w / 1280
    for cx, cy, rw, rh in [(180,70,75,28),(320,50,55,22),(700,80,90,32),
                           (950,55,65,24),(1150,72,70,26)]:
        cx, cy, rw, rh = int(cx*sx), int(cy*sx), int(rw*sx), int(rh*sx)
        pygame.draw.ellipse(surf, (255,240,230), (cx-rw, cy-rh, rw*2, rh*2))
        pygame.draw.ellipse(surf, (255,248,242),
                            (cx-rw+8, cy-rh+6, rw*2-16, rh*2-12))
    return surf


class Game:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption(TITLE)
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock  = pygame.time.Clock()

        self.quality_idx = 1                       # Média por padrão
        _, scale, _ = QUALITIES[self.quality_idx]
        self.renderer = Renderer(self.screen, render_scale=scale)
        self.sky = bake_sky(self.renderer.rw, self.renderer.rh)

        self.hud   = HUD(self.screen)
        self.sound = SoundManager()
        self.sound.init()
        self.show_fps    = True
        self.phase_idx   = 0
        self.total_score = 0
        self._init_phase()

    def _apply_quality(self):
        """Troca de qualidade em tempo real (F1)."""
        name, scale, _dens = QUALITIES[self.quality_idx]
        self.renderer.set_scale(scale)
        self.sky = bake_sky(self.renderer.rw, self.renderer.rh)

    def _init_phase(self):
        cfg = PHASES[self.phase_idx]
        _, _, density = QUALITIES[self.quality_idx]
        self.state       = GameState.START if self.phase_idx == 0 else GameState.PLAYING
        self.time_left   = float(cfg["time"])
        self.time_used   = 0.0
        self.score       = 0
        self.combo       = 1
        self.combo_timer = 0.0
        self.won         = False
        self.step_timer  = 0.0
        self.world  = World(seed=cfg["seed"], n_fish=cfg["fish"],
                            n_stars=cfg["stars"], density=density)
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

    # ── 1. Entrada ───────────────────────────────────────────────────────────

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_F1:
                    self.quality_idx = (self.quality_idx + 1) % len(QUALITIES)
                    self._apply_quality()
                if event.key == pygame.K_F3:
                    self.show_fps = not self.show_fps
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

    # ── 2. Atualização ───────────────────────────────────────────────────────

    def update(self, dt):
        self.hud.update(dt)
        if self.state == GameState.PLAYING:
            self._update_playing(dt)

    def _update_playing(self, dt):
        self.time_left -= dt
        self.time_used += dt
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

        keys = pygame.key.get_pressed()
        self.cat.update(dt, keys,
                        (WORLD_X[0], WORLD_X[1], WORLD_Z[0], WORLD_Z[1]),
                        self.world.all_obstacle_aabbs)

        if self.cat.speed > 1.0:
            self.step_timer -= dt
            if self.step_timer <= 0:
                self.step_timer = 0.26 - self.cat.speed * 0.01
                self.sound.play('step')

        self.camera.update(self.cat.pos, self.cat.yaw,
                           self.cat.speed, self.cat.yaw_rate,
                           self.cat.dashing, self.cat.impact_strength, dt)

        for fish in self.world.fishes:
            fish.update(dt)

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
                self.sound.play('star' if fish.is_star else 'fish')
                if self.combo > 2:
                    self.sound.play('combo')

        if all(f.collected for f in self.world.fishes):
            self.state = GameState.END
            self.won   = True
            self.score += int(self.time_left * 5)
            self.sound.stop_ambient()
            self.sound.play('win')
            self.camera.add_shake(0.2)

    # ── 3. Renderização ──────────────────────────────────────────────────────

    def render(self):
        # Cena 3D na superfície interna (escalada)
        r = self.renderer
        r.surface.blit(self.sky, (0, 0))         # céu: 1 blit
        r.begin(self.camera.view_matrix(), self.camera.proj_matrix(),
                self.camera.eye)
        self.world.render(r)
        self.world.render_collectibles(r)
        r.submit_mesh(self.cat.prep, self.cat.model_matrix())
        r.flush()
        r.blit_to_screen()

        # HUD em resolução cheia
        cfg = PHASES[self.phase_idx]
        if self.state == GameState.START:
            self.hud.draw_start_screen(cfg["name"])
        elif self.state == GameState.PLAYING:
            remaining = sum(1 for f in self.world.fishes if not f.collected)
            self.hud.draw_game(self.score, remaining, self.total_fishes,
                               self.time_left, self.combo,
                               self.phase_idx+1, len(PHASES))
            self.hud.draw_minimap(self.cat.pos, self.world.fishes,
                                  WORLD_X, WORLD_Z)
            if self.cat.dash_cooldown > 0:
                self._draw_dash_bar()
        elif self.state == GameState.PAUSE:
            self.hud.draw_pause()
        elif self.state == GameState.END:
            self.hud.draw_end_screen(self.score, self.won, self.time_used,
                                     self.phase_idx+1, len(PHASES))
        elif self.state == GameState.WIN_ALL:
            self.hud.draw_win_all(self.total_score + self.score)

        if self.show_fps:
            qname = QUALITIES[self.quality_idx][0]
            self.hud.draw_fps(self.clock.get_fps(), qname)

        pygame.display.flip()

    def _draw_dash_bar(self):
        pct = 1.0 - self.cat.dash_cooldown / self.cat.DASH_COOLDOWN
        bx, by, bw, bh = 10, HEIGHT - 50, 120, 12
        pygame.draw.rect(self.screen, (120, 95, 80), (bx, by, bw, bh),
                         border_radius=4)
        if pct > 0:
            fill_c = (170, 215, 250) if pct < 1.0 else (210, 235, 255)
            pygame.draw.rect(self.screen, fill_c,
                             (bx, by, int(bw*pct), bh), border_radius=4)


if __name__ == "__main__":
    Game().run()
