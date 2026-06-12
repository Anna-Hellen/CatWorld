"""
hud.py – Interface do Usuário (RF-04, RF-05, Bônus: UI polida + múltiplas fases)
"""

import pygame
import math

UI_BG     = (90, 65, 55, 190)
UI_ACCENT = (255, 195, 120)
UI_ACCENT2= (170, 215, 250)
UI_TEXT   = (255, 250, 242)
UI_GREEN  = (150, 225, 160)
UI_RED    = (245, 140, 130)
UI_ORANGE = (255, 175, 110)


class HUD:
    def __init__(self, screen):
        self.screen = screen
        self.W = screen.get_width()
        self.H = screen.get_height()
        pygame.font.init()
        self.font_big   = pygame.font.SysFont("Segoe UI", 52, bold=True)
        self.font_mid   = pygame.font.SysFont("Segoe UI", 32, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 22)
        self.font_tiny  = pygame.font.SysFont("Segoe UI", 18)
        self._pulse = 0.0
        # Cache de superfícies de texto: fontes renderizadas 1x por
        # (texto, fonte, cor) em vez de a cada frame
        self._text_cache = {}
        self._minimap_bg = None
        self._fps_acc = 0.0
        self._fps_surf = None

    def update(self, dt):
        self._pulse += dt * 3

    # ── Jogo ─────────────────────────────────────────────────────────────────

    def draw_game(self, score, fishes_left, total_fishes,
                  time_left, combo, phase, total_phases):
        # Pontuação
        self._panel(8, 8, 200, 56)
        self._text(f"🐟 {score}", 20, 18, self.font_mid, UI_ACCENT)

        # Objetivos
        self._panel(8, 72, 200, 50)
        self._text(f"{fishes_left}/{total_fishes} peixes", 20, 82, self.font_small, UI_TEXT)

        # Fase
        self._panel(self.W//2 - 80, 8, 160, 40)
        self._text(f"Fase {phase}/{total_phases}", self.W//2, 16,
                   self.font_small, UI_ACCENT2, center=True)

        # Tempo
        t_color = UI_GREEN
        if time_left < 30: t_color = UI_ORANGE
        if time_left < 10:
            t_color = (255, 60, 60) if math.sin(self._pulse * 3) > 0 else UI_RED
        self._panel(self.W - 150, 8, 142, 56)
        m, s = int(time_left)//60, int(time_left)%60
        self._text(f"⏱ {m}:{s:02d}", self.W - 138, 18, self.font_mid, t_color)

        # Combo
        if combo > 1:
            alpha = int(255 * min(1.0, abs(math.sin(self._pulse))))
            surf = self.font_mid.render(f"✨ COMBO x{combo}!", True, UI_ACCENT2)
            surf.set_alpha(alpha)
            self.screen.blit(surf, (self.W//2 - surf.get_width()//2, 58))

        # Controles
        self._text("W/↑ Mover  A/D Girar  S/↓ Ré  SHIFT Dash  ESC Pausar",
                   10, self.H - 28, self.font_tiny, (120, 95, 80))

    def draw_minimap(self, cat_pos, fishes, world_x, world_z):
        S = 120; PAD = 10
        mx = self.W - S - PAD; my = self.H - S - PAD
        if self._minimap_bg is None:
            bg = pygame.Surface((S, S), pygame.SRCALPHA)
            bg.fill((85, 70, 55, 175))
            pygame.draw.rect(bg, (210, 180, 145), (0, 0, S, S), 2,
                             border_radius=6)
            self._minimap_bg = bg
        self.screen.blit(self._minimap_bg, (mx, my))

        def wm(x, z):
            nx = (x - world_x[0]) / (world_x[1] - world_x[0])
            nz = (z - world_z[0]) / (world_z[1] - world_z[0])
            return int(mx + nx*S), int(my + nz*S)

        for fish in fishes:
            if not fish.collected:
                px, pz = wm(fish.pos[0], fish.pos[2])
                c = (255, 220, 50) if fish.is_star else (80, 180, 255)
                pygame.draw.circle(self.screen, c, (px, pz), 4 if fish.is_star else 3)

        cx, cz = wm(cat_pos[0], cat_pos[2])
        pygame.draw.circle(self.screen, (255, 120, 60), (cx, cz), 5)

    # ── Telas ────────────────────────────────────────────────────────────────

    def draw_start_screen(self, phase_name="Fase 1 – O Jardim"):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((70, 50, 45, 215))
        self.screen.blit(overlay, (0, 0))

        self._text("🐱 CATWORLD 🐱", self.W//2, self.H//2 - 170,
                   self.font_big, UI_ACCENT, center=True)
        self._text(phase_name, self.W//2, self.H//2 - 105,
                   self.font_mid, UI_ACCENT2, center=True)

        self._panel(self.W//2 - 230, self.H//2 - 65, 460, 170)
        controls = [
            ("W / ↑", "Mover para frente"),
            ("S / ↓", "Ré"),
            ("A / ← e D / →", "Girar"),
            ("ESC", "Pausar"),
        ]
        for i, (key, desc) in enumerate(controls):
            y = self.H//2 - 50 + i * 36
            self._text(key, self.W//2 - 60, y, self.font_small, UI_ACCENT, right=True)
            self._text(desc, self.W//2 - 45, y, self.font_small, UI_TEXT)

        if math.sin(self._pulse) > 0:
            self._text("Pressione ESPAÇO para começar",
                       self.W//2, self.H//2 + 135, self.font_mid, UI_GREEN, center=True)
        self._text("★ Estrelas douradas valem 50 pts!",
                   self.W//2, self.H//2 + 180, self.font_small, UI_ACCENT2, center=True)

    def draw_pause(self):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((70, 50, 45, 170))
        self.screen.blit(overlay, (0, 0))
        self._text("⏸ PAUSADO", self.W//2, self.H//2 - 60,
                   self.font_big, UI_ACCENT, center=True)
        self._text("ESPAÇO ou ESC para continuar",
                   self.W//2, self.H//2 + 20, self.font_mid, UI_TEXT, center=True)

    def draw_end_screen(self, score, won, time_used, phase, total_phases):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((70, 50, 45, 200))
        self.screen.blit(overlay, (0, 0))

        if won:
            title = f"🎉 FASE {phase} CONCLUÍDA!"
            color = UI_GREEN
        else:
            title = "⏰ TEMPO ESGOTADO!"
            color = UI_RED

        self._text(title, self.W//2, self.H//2 - 130,
                   self.font_big, color, center=True)
        self._text(f"Pontuação: {score}", self.W//2, self.H//2 - 55,
                   self.font_mid, UI_TEXT, center=True)
        m, s = int(time_used)//60, int(time_used)%60
        self._text(f"Tempo: {m}:{s:02d}", self.W//2, self.H//2,
                   self.font_small, UI_ACCENT2, center=True)

        if math.sin(self._pulse) > 0:
            if won and phase < total_phases:
                self._text("ENTER → Próxima fase   |   ESPAÇO → Repetir",
                           self.W//2, self.H//2 + 70, self.font_mid, UI_ACCENT, center=True)
            else:
                self._text("ESPAÇO → Repetir fase",
                           self.W//2, self.H//2 + 70, self.font_mid, UI_ACCENT, center=True)

    def draw_win_all(self, total_score):
        overlay = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        overlay.fill((70, 50, 45, 205))
        self.screen.blit(overlay, (0, 0))
        self._text("🏆 VOCÊ COMPLETOU CATWORLD! 🏆",
                   self.W//2, self.H//2 - 150, self.font_big, UI_ACCENT, center=True)
        self._text(f"Pontuação total: {total_score}",
                   self.W//2, self.H//2 - 60, self.font_mid, UI_TEXT, center=True)
        self._text("Parabéns! Todas as 3 fases concluídas!",
                   self.W//2, self.H//2, self.font_small, UI_ACCENT2, center=True)
        if math.sin(self._pulse) > 0:
            self._text("ESPAÇO → Jogar novamente",
                       self.W//2, self.H//2 + 80, self.font_mid, UI_GREEN, center=True)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _panel(self, x, y, w, h, alpha=180):
        surf = pygame.Surface((w, h), pygame.SRCALPHA)
        surf.fill((105, 75, 60, alpha))
        pygame.draw.rect(surf, (200, 165, 135), (0, 0, w, h), 2, border_radius=8)
        self.screen.blit(surf, (x, y))

    def _text(self, text, x, y, font, color,
              center=False, right=False):
        key = (text, id(font), color)
        surf = self._text_cache.get(key)
        if surf is None:
            surf = font.render(text, True, color)
            if len(self._text_cache) > 300:   # evita crescer sem limite
                self._text_cache.clear()
            self._text_cache[key] = surf
        if center:  x -= surf.get_width() // 2
        elif right: x -= surf.get_width()
        self.screen.blit(surf, (x, y))

    def draw_fps(self, fps, quality_name):
        """Contador de FPS + qualidade atual (F3 alterna, F1 muda qualidade)."""
        self._fps_acc -= 1
        if self._fps_surf is None or self._fps_acc <= 0:
            self._fps_acc = 15   # atualiza a cada ~15 frames
            txt = f"{fps:4.0f} FPS  |  {quality_name} (F1)"
            self._fps_surf = self.font_tiny.render(txt, True, (110, 85, 70))
        self.screen.blit(self._fps_surf,
                         (self.W - self._fps_surf.get_width() - 10,
                          self.H - 26))
