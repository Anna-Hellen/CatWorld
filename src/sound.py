"""
sound.py – Som e Efeitos Sonoros Procedurais (Bônus)

Gera todos os sons programaticamente usando NumPy + pygame.mixer,
sem depender de arquivos externos.

Sons implementados:
  - Coleta de peixe: tom curto ascendente (chirp)
  - Coleta de estrela: melodia de 3 notas
  - Combo: som de fanfarra
  - Vitória: melodia completa
  - Derrota: som descendente
  - Trilha: loop de baixa frequência ambiente
"""

import numpy as np
import pygame
import math


SAMPLE_RATE = 22050
CHANNELS    = 1   # mono


def _sine(freq: float, duration: float, volume: float = 0.4,
          fade_out: bool = True, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Gera onda senoidal."""
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = np.sin(2 * math.pi * freq * t) * volume
    if fade_out and n > 0:
        env = np.linspace(1.0, 0.0, n) ** 0.5
        wave *= env
    return wave


def _square(freq: float, duration: float, volume: float = 0.2,
            sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Gera onda quadrada (mais brilhante/retro)."""
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    wave = np.sign(np.sin(2 * math.pi * freq * t)) * volume
    env = np.linspace(1.0, 0.0, n) ** 0.3
    return wave * env


def _chirp(f0: float, f1: float, duration: float, volume: float = 0.4,
           sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Gera chirp (varredura de frequência linear)."""
    n = int(sample_rate * duration)
    t = np.linspace(0, duration, n, endpoint=False)
    # Frequência instantânea: f(t) = f0 + (f1-f0)*t/duration
    phase = 2 * math.pi * (f0 * t + (f1 - f0) / (2 * duration) * t ** 2)
    wave = np.sin(phase) * volume
    env = np.linspace(1.0, 0.0, n) ** 0.4
    return wave * env


def _to_sound(wave: np.ndarray) -> pygame.mixer.Sound:
    """Converte array float64 [-1,1] em pygame.mixer.Sound."""
    wave = np.clip(wave, -1.0, 1.0)
    samples = (wave * 32767).astype(np.int16)
    # pygame.sndarray espera (N, channels)
    if CHANNELS == 2:
        samples = np.stack([samples, samples], axis=1)
    sound = pygame.sndarray.make_sound(samples)
    return sound


def _concat(*waves) -> np.ndarray:
    return np.concatenate(waves)


class SoundManager:
    """Gerencia inicialização e reprodução de sons procedurais."""

    def __init__(self):
        self.enabled = False
        self._sounds = {}
        self._music_channel = None

    def init(self) -> bool:
        """Tenta inicializar o mixer. Retorna True se bem-sucedido."""
        try:
            pygame.mixer.pre_init(SAMPLE_RATE, -16, CHANNELS, 512)
            pygame.mixer.init()
            self._generate_sounds()
            self.enabled = True
            return True
        except Exception as e:
            print(f"[sound] Áudio desabilitado: {e}")
            self.enabled = False
            return False

    def _generate_sounds(self):
        """Gera todos os efeitos sonoros proceduralmente."""

        # Coleta de peixe: chirp ascendente rápido + nota final
        fish_wave = _concat(
            _chirp(400, 900, 0.12, volume=0.35),
            _sine(900, 0.08, volume=0.3)
        )
        self._sounds['fish'] = _to_sound(fish_wave)

        # Estrela dourada: 3 notas (dó-mi-sol)
        star_wave = _concat(
            _chirp(520, 660, 0.08, 0.4),
            _sine(660, 0.06, 0.35),
            np.zeros(int(SAMPLE_RATE * 0.02)),
            _sine(780, 0.12, 0.4)
        )
        self._sounds['star'] = _to_sound(star_wave)

        # Combo: fanfarra crescente
        combo_wave = _concat(
            _square(440, 0.05, 0.25),
            _square(550, 0.05, 0.25),
            _square(660, 0.08, 0.3),
            _square(880, 0.10, 0.3),
        )
        self._sounds['combo'] = _to_sound(combo_wave)

        # Vitória: melodia alegre
        notes_win = [523, 659, 784, 1047, 784, 1047, 1175, 1047]
        durs_win  = [0.12, 0.12, 0.12, 0.20, 0.10, 0.10, 0.18, 0.28]
        win_parts = []
        for freq, dur in zip(notes_win, durs_win):
            win_parts.append(_sine(freq, dur, 0.35))
            win_parts.append(np.zeros(int(SAMPLE_RATE * 0.025)))
        self._sounds['win'] = _to_sound(_concat(*win_parts))

        # Derrota: tom descendente triste
        lose_wave = _concat(
            _chirp(440, 200, 0.5, 0.35),
            _chirp(200, 100, 0.4, 0.25),
        )
        self._sounds['lose'] = _to_sound(lose_wave)

        # Passo do gato: click suave
        step_wave = _chirp(120, 60, 0.05, 0.15)
        self._sounds['step'] = _to_sound(step_wave)

        # Trilha ambiente: drone de baixa frequência (loop)
        # Gera 3 s de drone com harmônicos
        t = np.linspace(0, 3.0, int(SAMPLE_RATE * 3.0))
        drone = (
            np.sin(2 * math.pi * 60 * t) * 0.08 +
            np.sin(2 * math.pi * 90 * t) * 0.05 +
            np.sin(2 * math.pi * 120 * t) * 0.03 +
            np.sin(2 * math.pi * 180 * t) * 0.015
        )
        # Modula levemente para dar movimento
        lfo = (np.sin(2 * math.pi * 0.3 * t) * 0.3 + 0.7)
        drone *= lfo
        self._sounds['ambient'] = _to_sound(drone)

    def play(self, name: str, loops: int = 0):
        if not self.enabled:
            return
        s = self._sounds.get(name)
        if s:
            s.play(loops=loops)

    def stop(self, name: str):
        if not self.enabled:
            return
        s = self._sounds.get(name)
        if s:
            s.stop()

    def start_ambient(self):
        """Inicia trilha ambiente em loop."""
        if not self.enabled:
            return
        s = self._sounds.get('ambient')
        if s:
            self._music_channel = s.play(loops=-1)

    def stop_ambient(self):
        if not self.enabled:
            return
        s = self._sounds.get('ambient')
        if s:
            s.stop()
