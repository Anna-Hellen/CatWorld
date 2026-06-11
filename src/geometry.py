"""
geometry.py – Primitivas gráficas e malhas 3D (RT-01)
Estética "cozy": paleta pastel quente, formas arredondadas,
decorações fofas (cogumelos, lanternas, arbustos).
"""

import numpy as np
import math
from renderer import Mesh

# ══ PALETA COZY (tons pastel quentes) ════════════════════════════════════════
# Gato – creme com manchas caramelo (estilo gatinho de pelúcia)
C_CAT_MAIN   = (247, 222, 192)   # creme claro
C_CAT_PATCH  = (232, 184, 138)   # caramelo suave
C_CAT_DARK   = (208, 158, 112)   # sombras quentes
C_CAT_BELLY  = (255, 246, 235)   # barriga quase branca
C_CAT_NOSE   = (245, 152, 162)   # rosinha
C_CAT_EYE    = ( 70,  52,  40)   # marrom doce
C_CAT_IRIS   = (130, 180, 150)   # verde menta
C_CAT_PAW    = (252, 238, 222)
C_CAT_INNER_EAR = (250, 192, 196)
C_CAT_BLUSH  = (250, 180, 170)   # bochechas rosadas!

# Peixe – azul bebê
C_FISH_TOP   = (148, 200, 235)
C_FISH_BELLY = (235, 247, 255)
C_FISH_FIN   = (120, 175, 220)
C_FISH_TAIL  = (105, 160, 210)
C_FISH_EYE   = ( 60,  60,  70)
C_FISH_SHINE = (255, 255, 255)
C_FISH_CHEEK = (255, 195, 200)   # bochecha rosa no peixe também

# Ambiente – jardim de primavera pastel
C_GRASS_A    = (168, 214, 142)   # verde clarinho
C_GRASS_B    = (155, 204, 130)
C_GRASS_C    = (180, 222, 155)   # manchas mais claras
C_PATH       = (238, 214, 178)   # caminho de areia clara
C_PATH_EDGE  = (224, 198, 160)
C_WALL_TOP   = (242, 220, 196)   # muros de pedra clara
C_WALL_SIDE  = (228, 202, 174)
C_WALL_DARK  = (205, 178, 150)
C_WALL_BASE  = (190, 162, 134)
C_CRATE_TOP  = (222, 184, 140)   # madeira clara de mel
C_CRATE_SIDE = (205, 165, 122)
C_CRATE_DARK = (180, 142, 100)
C_STONE_TOP  = (216, 208, 200)   # pedrinhas claras
C_STONE_SIDE = (196, 188, 180)
C_STONE_DARK = (172, 164, 156)
C_TRUNK      = (170, 130,  95)
C_BARK       = (150, 112,  80)
C_LEAF_A     = (150, 205, 130)   # copas fofas
C_LEAF_B     = (170, 218, 145)
C_LEAF_C     = (190, 230, 160)
C_LEAF_PINK  = (245, 195, 205)   # árvore de cerejeira!
C_LEAF_PINK2 = (250, 210, 218)
C_LEAF_PINK3 = (252, 224, 230)
C_STAR       = (255, 222,  95)
C_STAR_DARK  = (245, 198,  60)
C_FENCE      = (250, 240, 228)   # cerquinha branca
C_FENCE_2    = (240, 228, 214)
C_MUSH_CAP   = (240, 130, 120)   # cogumelo vermelho pastel
C_MUSH_DOT   = (255, 248, 240)
C_MUSH_STEM  = (250, 240, 228)
C_LANT_POST  = (140, 110,  85)   # poste da lanterna
C_LANT_GLOW  = (255, 228, 150)   # luz quentinha
C_LANT_TOP   = (110,  88,  70)
C_BUSH_A     = (160, 210, 138)
C_BUSH_B     = (178, 222, 152)
C_FLOWER_CTR = (255, 215,  90)


# ══ Builders ═════════════════════════════════════════════════════════════════

def _mk(verts, faces):
    return Mesh(np.array(verts, dtype=np.float64), faces)


def _add_box(verts, faces, ox, oy, oz, w, h, d,
             top_c, side_c, bot_c=None, front_c=None, border=False):
    """Caixa. border=False por padrão → visual mais suave/fofo."""
    if bot_c is None: bot_c = side_c
    if front_c is None: front_c = side_c
    i = len(verts)
    x, y, z = w/2, h/2, d/2
    verts += [
        [ox-x, oy-y, oz+z], [ox+x, oy-y, oz+z],
        [ox+x, oy+y, oz+z], [ox-x, oy+y, oz+z],
        [ox-x, oy-y, oz-z], [ox+x, oy-y, oz-z],
        [ox+x, oy+y, oz-z], [ox-x, oy+y, oz-z],
    ]
    faces += [
        ([i+3, i+2, i+6, i+7], top_c,   border),
        ([i+0, i+1, i+2, i+3], front_c, border),
        ([i+5, i+4, i+7, i+6], side_c,  border),
        ([i+4, i+0, i+3, i+7], side_c,  border),
        ([i+1, i+5, i+6, i+2], side_c,  border),
        ([i+4, i+5, i+1, i+0], bot_c,   False),
    ]


def _add_round_box(verts, faces, ox, oy, oz, w, h, d, top_c, side_c, bot_c=None):
    """
    'Caixa arredondada': caixa central + 4 caixas menores nos cantos chanfrados.
    Dá aparência mais fofa/orgânica que um cubo seco.
    """
    if bot_c is None: bot_c = side_c
    # Corpo principal levemente menor
    _add_box(verts, faces, ox, oy, oz, w*0.92, h, d*0.92, top_c, side_c, bot_c)
    # Painéis laterais que "completam" a largura (efeito chanfro)
    _add_box(verts, faces, ox, oy, oz, w, h*0.84, d*0.84, top_c, side_c, bot_c)


def _add_tri(verts, faces, p0, p1, p2, color, border=False):
    i = len(verts)
    verts += [list(p0), list(p1), list(p2)]
    faces.append(([i, i+1, i+2], color, border))


def _add_quad(verts, faces, p0, p1, p2, p3, color, border=False):
    i = len(verts)
    verts += [list(p0), list(p1), list(p2), list(p3)]
    faces.append(([i, i+1, i+2, i+3], color, border))


def _add_blob(verts, faces, ox, oy, oz, r, top_c, side_c, squash=0.85):
    """
    'Blob' fofinho: aproximação de esfera com 3 caixas rotacionadas em escala.
    Usado para copas de árvore, arbustos e nuvens.
    """
    _add_box(verts, faces, ox, oy, oz, r*2.0, r*2.0*squash, r*2.0, top_c, side_c)
    _add_box(verts, faces, ox, oy, oz, r*2.3, r*1.5*squash, r*1.5, top_c, side_c)
    _add_box(verts, faces, ox, oy, oz, r*1.5, r*1.5*squash, r*2.3, top_c, side_c)


# ══ Chão ═════════════════════════════════════════════════════════════════════

def floor_tile_mesh(size=4.0, color=None) -> Mesh:
    c = color or C_GRASS_A
    s = size / 2
    verts = [[-s,0,s],[s,0,s],[s,0,-s],[-s,0,-s]]
    faces = [([0,1,2,3], c, False)]
    return _mk(verts, faces)


# ══ Muros baixos de pedra clara ══════════════════════════════════════════════

def wall_segment_mesh(length, height=2.2) -> Mesh:
    """Muro baixo e fofo de pedra clara com 'pedrinhas' no topo."""
    verts, faces = [], []
    _add_box(verts, faces, 0, height/2, 0, length, height, 0.9,
             C_WALL_TOP, C_WALL_SIDE, C_WALL_BASE)
    # Pedrinhas arredondadas no topo (alternadas)
    step = 2.2
    n = max(1, int(length / step))
    for k in range(n):
        x = -length/2 + k*step + step/2
        _add_round_box(verts, faces, x, height + 0.22, 0,
                       1.1, 0.45, 1.0, C_WALL_TOP, C_WALL_DARK)
    return _mk(verts, faces)


# ══ Obstáculos ═══════════════════════════════════════════════════════════════

def crate_mesh(w=1.4, h=1.4, d=1.4) -> Mesh:
    """Caixote de madeira clara, cantos suaves."""
    verts, faces = [], []
    _add_round_box(verts, faces, 0, h/2, 0, w, h, d,
                   C_CRATE_TOP, C_CRATE_SIDE, C_CRATE_DARK)
    # Fita/tábua horizontal clara
    _add_box(verts, faces, 0, h*0.5, 0, w*1.02, h*0.16, d*1.02,
             C_CRATE_DARK, C_CRATE_DARK)
    return _mk(verts, faces)


def stone_mesh(w=1.8, h=0.9, d=1.8) -> Mesh:
    """Pedra arredondada clara."""
    verts, faces = [], []
    _add_blob(verts, faces, 0, h*0.45, 0, max(w,d)*0.5,
              C_STONE_TOP, C_STONE_SIDE, squash=h/max(w,d))
    return _mk(verts, faces)


def obstacle_mesh(w=1.2, h=1.4, d=1.2) -> Mesh:
    return stone_mesh(w,h,d) if h < 1.0 else crate_mesh(w,h,d)


# ══ Cogumelo fofo ════════════════════════════════════════════════════════════

def mushroom_mesh(scale=1.0) -> Mesh:
    verts, faces = [], []
    s = scale
    # Caule
    _add_box(verts, faces, 0, 0.22*s, 0, 0.22*s, 0.44*s, 0.22*s,
             C_MUSH_STEM, C_MUSH_STEM)
    # Chapéu (blob achatado)
    _add_blob(verts, faces, 0, 0.52*s, 0, 0.34*s,
              C_MUSH_CAP, C_MUSH_CAP, squash=0.6)
    # Bolinhas brancas
    for bx, bz in [(-0.14,0.10),(0.16,-0.06),(0.0,0.20),(-0.05,-0.18)]:
        _add_box(verts, faces, bx*s, 0.62*s, bz*s, 0.10*s, 0.06*s, 0.10*s,
                 C_MUSH_DOT, C_MUSH_DOT)
    return _mk(verts, faces)


# ══ Lanterna de jardim ═══════════════════════════════════════════════════════

def lantern_mesh() -> Mesh:
    verts, faces = [], []
    # Poste
    _add_box(verts, faces, 0, 0.8, 0, 0.12, 1.6, 0.12, C_LANT_POST, C_LANT_POST)
    # Caixa de luz (brilhante – não recebe borda)
    _add_box(verts, faces, 0, 1.75, 0, 0.42, 0.42, 0.42,
             C_LANT_GLOW, C_LANT_GLOW)
    # Telhadinho
    _add_tri(verts, faces, [-0.30,1.96, 0.30],[0.30,1.96, 0.30],[0,2.25,0], C_LANT_TOP)
    _add_tri(verts, faces, [ 0.30,1.96,-0.30],[-0.30,1.96,-0.30],[0,2.25,0], C_LANT_TOP)
    _add_tri(verts, faces, [-0.30,1.96,-0.30],[-0.30,1.96, 0.30],[0,2.25,0], C_LANT_TOP)
    _add_tri(verts, faces, [ 0.30,1.96, 0.30],[ 0.30,1.96,-0.30],[0,2.25,0], C_LANT_TOP)
    return _mk(verts, faces)


# ══ Arbusto redondinho ═══════════════════════════════════════════════════════

def bush_mesh(scale=1.0) -> Mesh:
    verts, faces = [], []
    s = scale
    _add_blob(verts, faces, 0, 0.45*s, 0, 0.5*s, C_BUSH_A, C_BUSH_B, squash=0.8)
    _add_blob(verts, faces, 0.3*s, 0.35*s, 0.15*s, 0.32*s, C_BUSH_B, C_BUSH_A, squash=0.85)
    _add_blob(verts, faces, -0.28*s, 0.38*s, -0.1*s, 0.3*s, C_BUSH_B, C_BUSH_A, squash=0.85)
    # Florzinhas no arbusto
    for fx, fy, fz in [(0.15,0.75,0.25),(-0.25,0.7,0.1),(0.3,0.6,-0.2)]:
        _add_box(verts, faces, fx*s, fy*s, fz*s, 0.10*s, 0.08*s, 0.10*s,
                 (250,200,210),(250,200,210))
    return _mk(verts, faces)


# ══ Flor ═════════════════════════════════════════════════════════════════════

def flower_mesh(color=None) -> Mesh:
    c = color or (245, 170, 185)
    verts, faces = [], []
    _add_box(verts, faces, 0, 0.22, 0, 0.05, 0.44, 0.05, (140,190,120),(125,175,105))
    # 5 pétalas redondinhas
    for i in range(5):
        a = i * 2*math.pi/5
        px, pz = math.cos(a)*0.16, math.sin(a)*0.16
        _add_box(verts, faces, px, 0.50, pz, 0.14, 0.09, 0.14, c, c)
    _add_box(verts, faces, 0, 0.52, 0, 0.13, 0.11, 0.13,
             C_FLOWER_CTR, (240,195,70))
    return _mk(verts, faces)


# ══ Árvores fofas (incluindo cerejeiras!) ════════════════════════════════════

def tree_mesh(pink=False) -> Mesh:
    """Árvore com copa de blobs fofos. pink=True → cerejeira."""
    verts, faces = [], []
    # Tronco
    _add_box(verts, faces, 0, 0.8, 0, 0.4, 1.6, 0.36, C_TRUNK, C_BARK)
    _add_box(verts, faces, 0, 0.8, 0, 0.30, 1.6, 0.46, C_TRUNK, C_BARK)

    if pink:
        c1, c2, c3 = C_LEAF_PINK, C_LEAF_PINK2, C_LEAF_PINK3
    else:
        c1, c2, c3 = C_LEAF_A, C_LEAF_B, C_LEAF_C

    # Copa: 4 blobs sobrepostos (formato de nuvem fofa)
    _add_blob(verts, faces,  0.0, 2.3,  0.0, 1.15, c2, c1, squash=0.85)
    _add_blob(verts, faces,  0.75,2.05, 0.35, 0.75, c3, c2, squash=0.85)
    _add_blob(verts, faces, -0.7, 2.1, -0.3,  0.72, c3, c2, squash=0.85)
    _add_blob(verts, faces,  0.0, 3.0,  0.0,  0.78, c3, c2, squash=0.85)
    return _mk(verts, faces)


# ══ Casinha aconchegante ═════════════════════════════════════════════════════

def house_mesh() -> Mesh:
    WALL  = (252, 240, 222)
    WALL2 = (242, 226, 205)
    ROOF  = (235, 150, 130)   # telhado salmão
    ROOF2 = (220, 132, 112)
    DOOR  = (180, 135,  95)
    WIN   = (200, 230, 248)
    WIN_F = (255, 250, 240)
    BASE  = (228, 208, 182)

    verts, faces = [], []
    _add_box(verts, faces, 0, 0.2, 0, 3.2, 0.4, 2.8, BASE, BASE)
    _add_box(verts, faces, 0, 1.8, 0, 3.0, 3.2, 2.6, WALL, WALL2, BASE)
    h0, h1, W2, D2 = 3.4, 5.0, 1.5, 1.3
    _add_tri(verts, faces, [-W2,h0, D2],[W2,h0, D2],[0,h1,0], ROOF)
    _add_tri(verts, faces, [ W2,h0,-D2],[-W2,h0,-D2],[0,h1,0], ROOF)
    _add_quad(verts, faces, [-W2,h0, D2],[-W2,h0,-D2],[0,h1,0],[0,h1,0], ROOF2)
    _add_quad(verts, faces, [ W2,h0,-D2],[ W2,h0, D2],[0,h1,0],[0,h1,0], ROOF2)
    # Chaminé com fumacinha (blob branco)
    _add_box(verts, faces, 0.5, 4.5, -0.4, 0.4, 0.9, 0.4,
             (200,178,156),(190,168,146))
    _add_blob(verts, faces, 0.5, 5.4, -0.4, 0.28, (250,250,250),(240,240,242))
    _add_blob(verts, faces, 0.72, 5.9, -0.35, 0.20, (250,250,250),(242,242,244))
    # Porta arredondada (caixa + semicaixa)
    _add_box(verts, faces, 0, 0.85, 1.32, 0.66, 1.7, 0.10, DOOR, DOOR)
    _add_box(verts, faces, 0, 1.72, 1.32, 0.50, 0.22, 0.10, DOOR, DOOR)
    _add_box(verts, faces, 0.20, 0.9, 1.38, 0.08, 0.08, 0.05,
             (240,210,140),(240,210,140))
    # Janelas redondinhas com moldura branca
    for wx in [-0.85, 0.85]:
        _add_box(verts, faces, wx, 2.1, 1.33, 0.60, 0.60, 0.08, WIN, WIN)
        _add_box(verts, faces, wx, 2.1, 1.31, 0.72, 0.72, 0.05, WIN_F, WIN_F)
        # Floreira embaixo da janela!
        _add_box(verts, faces, wx, 1.68, 1.40, 0.66, 0.16, 0.16,
                 (190,150,110),(175,138,100))
        for fi, fx in enumerate([-0.18, 0.0, 0.18]):
            fc = [(245,170,185),(255,210,120),(200,160,220)][fi]
            _add_box(verts, faces, wx+fx, 1.82, 1.42, 0.12, 0.12, 0.12, fc, fc)
    return _mk(verts, faces)


# ══ Gato fofinho ═════════════════════════════════════════════════════════════

def cat_body_mesh() -> Mesh:
    """
    Gatinho estilo pelúcia: corpo redondinho, cabeça grande (proporção chibi),
    olhos enormes, bochechas rosadas, manchas caramelo.
    """
    verts, faces = [], []

    def box(ox,oy,oz,w,h,d,tc,sc,bc=None,fc=None):
        _add_box(verts,faces,ox,oy,oz,w,h,d,tc,sc,bc,fc)
    def blob(ox,oy,oz,r,tc,sc,squash=0.85):
        _add_blob(verts,faces,ox,oy,oz,r,tc,sc,squash)
    def tri(p0,p1,p2,c):
        _add_tri(verts,faces,p0,p1,p2,c)

    # ── Corpo redondinho (blob) ───────────────────────
    blob(0, -0.05, -0.1, 0.62, C_CAT_MAIN, C_CAT_PATCH, squash=0.78)
    # Barriga clara
    box(0, -0.15, 0.42, 0.62, 0.5, 0.12, C_CAT_BELLY, C_CAT_BELLY)
    # Mancha caramelo nas costas
    box(0, 0.32, -0.15, 0.5, 0.12, 0.62, C_CAT_PATCH, C_CAT_PATCH)

    # ── Cabeça GRANDE (proporção chibi = fofura) ──────
    blob(0, 0.78, 0.22, 0.55, C_CAT_MAIN, C_CAT_PATCH, squash=0.88)
    # Mancha caramelo na orelha/olho direito (estilo malhado)
    box(0.25, 1.02, 0.18, 0.4, 0.3, 0.5, C_CAT_PATCH, C_CAT_PATCH)

    # ── Orelhas redondinhas ───────────────────────────
    # (triângulos largos e curtos = mais fofas que pontudas)
    tri([-0.45, 1.18, 0.15], [-0.13, 1.52, 0.15], [-0.05, 1.18, 0.15], C_CAT_PATCH)
    tri([-0.43, 1.20, 0.05], [-0.14, 1.48, 0.05], [-0.07, 1.20, 0.05], C_CAT_PATCH)
    tri([-0.36, 1.22, 0.17], [-0.16, 1.42, 0.17], [-0.11, 1.22, 0.17], C_CAT_INNER_EAR)
    tri([ 0.45, 1.18, 0.15], [ 0.13, 1.52, 0.15], [ 0.05, 1.18, 0.15], C_CAT_PATCH)
    tri([ 0.43, 1.20, 0.05], [ 0.14, 1.48, 0.05], [ 0.07, 1.20, 0.05], C_CAT_PATCH)
    tri([ 0.36, 1.22, 0.17], [ 0.16, 1.42, 0.17], [ 0.11, 1.22, 0.17], C_CAT_INNER_EAR)

    # ── Focinho ───────────────────────────────────────
    box(0, 0.62, 0.70, 0.36, 0.22, 0.14, C_CAT_BELLY, C_CAT_BELLY)
    # Nariz rosinha (coração invertido aproximado)
    tri([0, 0.70, 0.80], [-0.06, 0.65, 0.80], [0.06, 0.65, 0.80], C_CAT_NOSE)
    # Boquinha "w" (duas linhas)
    box(-0.05, 0.58, 0.79, 0.08, 0.02, 0.02, (170,120,110),(170,120,110))
    box( 0.05, 0.58, 0.79, 0.08, 0.02, 0.02, (170,120,110),(170,120,110))

    # ── Olhos ENORMES e brilhantes ────────────────────
    for ex in [-0.25, 0.25]:
        # Olho grande fechado feliz não... olho aberto grande
        box(ex, 0.85, 0.71, 0.20, 0.24, 0.07, C_CAT_EYE, C_CAT_EYE)
        # Íris verde menta
        box(ex, 0.83, 0.745, 0.16, 0.16, 0.04, C_CAT_IRIS, C_CAT_IRIS)
        # Pupila
        box(ex, 0.84, 0.77, 0.09, 0.14, 0.03, C_CAT_EYE, C_CAT_EYE)
        # Brilho grande (canto sup. direito)
        box(ex+0.05, 0.92, 0.79, 0.06, 0.06, 0.02, (255,255,255),(255,255,255))
        # Brilho pequeno
        box(ex-0.04, 0.80, 0.79, 0.03, 0.03, 0.02, (255,255,255),(255,255,255))

    # ── BOCHECHAS ROSADAS (essencial pra fofura) ──────
    box(-0.38, 0.66, 0.62, 0.14, 0.09, 0.06, C_CAT_BLUSH, C_CAT_BLUSH)
    box( 0.38, 0.66, 0.62, 0.14, 0.09, 0.06, C_CAT_BLUSH, C_CAT_BLUSH)

    # ── Patinhas curtas e gordinhas ───────────────────
    for px in [-0.3, 0.3]:
        # Dianteiras
        box(px, -0.52, 0.38, 0.26, 0.38, 0.26, C_CAT_MAIN, C_CAT_PATCH)
        box(px, -0.70, 0.40, 0.30, 0.16, 0.32, C_CAT_PAW, C_CAT_PAW)
        # Traseiras (mais gordinhas)
        box(px, -0.48, -0.42, 0.30, 0.42, 0.32, C_CAT_PATCH, C_CAT_MAIN)
        box(px, -0.70, -0.38, 0.32, 0.16, 0.36, C_CAT_PAW, C_CAT_PAW)

    # ── Caudinha curvada fofa ─────────────────────────
    tail = [
        (0.45, -0.05, -0.62, 0.17),
        (0.62,  0.14, -0.70, 0.16),
        (0.74,  0.38, -0.66, 0.15),
        (0.78,  0.60, -0.52, 0.14),
        (0.72,  0.78, -0.34, 0.13),
    ]
    for tx,ty,tz,sz in tail:
        box(tx, ty, tz, sz, sz, sz, C_CAT_PATCH, C_CAT_DARK)
    # Pontinha clara
    box(0.64, 0.90, -0.20, 0.15, 0.15, 0.15, C_CAT_BELLY, C_CAT_BELLY)

    return _mk(verts, faces)


# ══ Peixinho fofo ════════════════════════════════════════════════════════════

def fish_mesh() -> Mesh:
    """Peixinho gordinho com olhos grandes e bochecha rosa."""
    verts, faces = [], []
    def box(ox,oy,oz,w,h,d,tc,sc):
        _add_box(verts,faces,ox,oy,oz,w,h,d,tc,sc)
    def tri(p0,p1,p2,c):
        _add_tri(verts,faces,p0,p1,p2,c)
    def blob(ox,oy,oz,r,tc,sc,squash=0.85):
        _add_blob(verts,faces,ox,oy,oz,r,tc,sc,squash)

    # Corpo redondinho (blob gordinho)
    blob(0.02, 0, 0, 0.30, C_FISH_TOP, C_FISH_BELLY, squash=0.8)
    # Barriguinha
    box(0.02, -0.10, 0, 0.40, 0.14, 0.20, C_FISH_BELLY, C_FISH_BELLY)

    # Cauda em leque
    tri([-0.30, 0, 0], [-0.55, 0.20, 0], [-0.50, 0.00, 0], C_FISH_TAIL)
    tri([-0.30, 0, 0], [-0.55,-0.20, 0], [-0.50, 0.00, 0], C_FISH_TAIL)
    tri([-0.30, 0, 0], [-0.55, 0.20, 0], [-0.55,-0.20, 0], C_FISH_FIN)

    # Barbatana dorsal redondinha
    box(0.0, 0.28, 0, 0.18, 0.14, 0.06, C_FISH_FIN, C_FISH_FIN)
    # Barbatanas laterais
    box(0.08, -0.04, 0.20, 0.14, 0.08, 0.10, C_FISH_FIN, C_FISH_FIN)
    box(0.08, -0.04,-0.20, 0.14, 0.08, 0.10, C_FISH_FIN, C_FISH_FIN)

    # Olho GRANDE
    box(0.20, 0.06, 0.16, 0.13, 0.15, 0.05, C_FISH_EYE, C_FISH_EYE)
    box(0.23, 0.10, 0.185, 0.05, 0.05, 0.03, C_FISH_SHINE, C_FISH_SHINE)
    box(0.20, 0.06,-0.16, 0.13, 0.15, 0.05, C_FISH_EYE, C_FISH_EYE)
    box(0.23, 0.10,-0.185, 0.05, 0.05, 0.03, C_FISH_SHINE, C_FISH_SHINE)

    # Bochechas rosadas!
    box(0.16, -0.05, 0.17, 0.08, 0.05, 0.04, C_FISH_CHEEK, C_FISH_CHEEK)
    box(0.16, -0.05,-0.17, 0.08, 0.05, 0.04, C_FISH_CHEEK, C_FISH_CHEEK)

    # Boquinha
    box(0.32, -0.02, 0, 0.04, 0.03, 0.06, (90,90,100),(90,90,100))
    return _mk(verts, faces)


# ══ Estrela ══════════════════════════════════════════════════════════════════

def star_mesh() -> Mesh:
    verts, faces = [], []
    n, OR, IR, D = 5, 0.40, 0.19, 0.10
    pts = []
    for i in range(n*2):
        a = math.pi/2 + i*math.pi/n
        r = OR if i%2==0 else IR
        pts.append((r*math.cos(a), r*math.sin(a)))
    ci = len(verts); verts.append([0,0,D])
    for x,y in pts: verts.append([x,y,D])
    for i in range(len(pts)):
        j = (i+1)%len(pts)
        faces.append(([ci, ci+1+i, ci+1+j], C_STAR, False))
    cb = len(verts); verts.append([0,0,-D])
    for x,y in pts: verts.append([x,y,-D])
    for i in range(len(pts)):
        j = (i+1)%len(pts)
        faces.append(([cb, cb+1+j, cb+1+i], C_STAR_DARK, False))
    npts = len(pts)
    for i in range(npts):
        j = (i+1)%npts
        sc = C_STAR if i%2==0 else C_STAR_DARK
        faces.append(([ci+1+i, ci+1+j, cb+1+j, cb+1+i], sc, False))
    # Carinha na estrela: dois olhinhos
    verts_len = len(verts)
    _add_box(verts, faces, -0.08, 0.05, D+0.005, 0.05, 0.08, 0.01,
             (80,60,30),(80,60,30))
    _add_box(verts, faces,  0.08, 0.05, D+0.005, 0.05, 0.08, 0.01,
             (80,60,30),(80,60,30))
    return _mk(verts, faces)


# ══ Cerquinha branca ═════════════════════════════════════════════════════════

def fence_segment_mesh(length=3.0) -> Mesh:
    verts, faces = [], []
    # Estacas verticais arredondadas no topo
    n = int(length / 0.55)
    for k in range(n):
        x = -length/2 + k * (length/(n-1)) if n > 1 else 0
        _add_box(verts, faces, x, 0.42, 0, 0.13, 0.84, 0.10,
                 C_FENCE, C_FENCE_2)
        # Topo arredondado (caixinha menor)
        _add_box(verts, faces, x, 0.88, 0, 0.10, 0.10, 0.08,
                 C_FENCE, C_FENCE_2)
    # Travessa
    _add_box(verts, faces, 0, 0.5, 0.02, length, 0.10, 0.06,
             C_FENCE_2, C_FENCE_2)
    return _mk(verts, faces)
