# src/simcore/vis.py

# Despre NumPy (np):
#  - Tipul de bază este np.ndarray (tablouri numerice N×D).
#  - Operațiile sunt vectorizate și suportă broadcasting (eficient, fără bucle Python).
#  - Zgomotul gaussian se generează cu np.random.normal(size=(N, D)).
#  - Reproductibilitate: RNG-ul de simulare (seed) se setează în Simulator; aici folosim
#    un RNG separat DOAR pentru culori (nu afectează fizica).
#  - Unități: toate mărimile sunt în SI (m, s, kg); vizualul aplică DOAR o scalare grafică.

"""
VIZUALIZARE VPYTHON (OPȚIONALĂ)
===============================
Scop:
  - să vedem “norul” de particule în mișcare, fără a schimba fizica.
  - pozițiile se pot scala DOAR pentru ecran (viz_scale), ca să fie vizibile.

Important:
  - Vizualizarea este *pasivă*: nu afectează datele salvate în .dat.
  - În unități SI, deplasările Browniene pot fi nanometrice pentru Δt scurt; se folosește
    un factor de scalare grafică (viz_scale) pentru cadrare (nu modifică dinamica).
  - Dacă viz_scale = 0 -> auto-scale: fie pe baza CUTIEI (dacă există pereți), fie pe baza
    RMS-ului norului la ultimul pas.

Despre r_series:
  - r_series are forma (N, T+1, D), D ∈ {2, 3}. Pentru D=2 randăm cu z=0 (plan XY).
  - VPython cere update per-sferă; pentru performanță limităm M = min(viz_n, N).

Dependență:
  - Necesită `vpython` (pip install vpython).
"""

from __future__ import annotations
import numpy as np


# ---------- Helpers de desen: wireframe box / rect ----------

def draw_wire_box(xmin, xmax, ymin, ymax, zmin, zmax, s, vector, curve, color):
    """
    Desenează muchiile unei cutii axis-aligned (wireframe).
    Parametri:
      xmin..zmax [m]  – coordonatele cutiei în unități SI
      s               – factorul vizual de scalare (viz_scale)
      vector, curve   – ctor-uri VPython, injectate de apelant
    """
    X = [xmin * s, xmax * s]
    Y = [ymin * s, ymax * s]
    Z = [zmin * s, zmax * s]
    edges = [
        ((X[0], Y[0], Z[0]), (X[1], Y[0], Z[0])), ((X[0], Y[1], Z[0]), (X[1], Y[1], Z[0])),
        ((X[0], Y[0], Z[1]), (X[1], Y[0], Z[1])), ((X[0], Y[1], Z[1]), (X[1], Y[1], Z[1])),
        ((X[0], Y[0], Z[0]), (X[0], Y[1], Z[0])), ((X[1], Y[0], Z[0]), (X[1], Y[1], Z[0])),
        ((X[0], Y[0], Z[1]), (X[0], Y[1], Z[1])), ((X[1], Y[0], Z[1]), (X[1], Y[1], Z[1])),
        ((X[0], Y[0], Z[0]), (X[0], Y[0], Z[1])), ((X[1], Y[0], Z[0]), (X[1], Y[0], Z[1])),
        ((X[0], Y[1], Z[0]), (X[0], Y[1], Z[1])), ((X[1], Y[1], Z[0]), (X[1], Y[1], Z[1])),
    ]
    for a, b in edges:
        curve(pos=[vector(*a), vector(*b)], color=color.white, radius=0.01)


def draw_wire_rect(xmin, xmax, ymin, ymax, s, vector, curve, color):
    """
    2D: dreptunghi wireframe în planul XY.
    """
    X = [xmin * s, xmax * s]
    Y = [ymin * s, ymax * s]
    edges = [
        ((X[0], Y[0], 0.0), (X[1], Y[0], 0.0)),
        ((X[1], Y[0], 0.0), (X[1], Y[1], 0.0)),
        ((X[1], Y[1], 0.0), (X[0], Y[1], 0.0)),
        ((X[0], Y[1], 0.0), (X[0], Y[0], 0.0)),
    ]
    for a, b in edges:
        curve(pos=[vector(*a), vector(*b)], color=color.white, radius=0.01)


def _box_from_walls(walls, D):
    """
    Încearcă să reconstruiască o cutie axis-aligned din lista de pereți (n, c), plan: n·r=c.
    Returnează (mins, maxs) pe primele D axe, sau (None, None) dacă nu e box complet.
    """
    if not walls:
        return None, None
    mins = np.full(3, np.nan)
    maxs = np.full(3, np.nan)
    for (n, c) in walls:
        n = np.asarray(n, float)[:D]
        axis = int(np.argmax(np.abs(n)))
        s = float(n[axis])
        if s > 0:       # +x/+y/+z => fața "max" pe axa respectivă e la c
            maxs[axis] = c
        elif s < 0:     # -x/-y/-z => fața "min" e la -c
            mins[axis] = -c
    if np.all(np.isfinite(mins[:D])) and np.all(np.isfinite(maxs[:D])):
        return mins[:D], maxs[:D]
    return None, None


def _vec_from_point(p, scale, D, vector):
    """RO: Construiește un vector VPython dintr-un punct 2D/3D, punând z=0 în 2D."""
    if D == 2:
        x, y = (p * scale).tolist()
        return vector(x, y, 0.0)
    else:
        x, y, z = (p * scale).tolist()
        return vector(x, y, z)


# ---------- Animație 3D/2D cu VPython ----------

def animate_vpython(
    r_series: np.ndarray,
    step_dt: float,
    viz_n: int = 100,
    viz_scale: float = 0.0,
    viz_trail: bool = False,
    walls=None,
    show_planes: bool = False,     # dacă vrei și plane translucide, nu doar wireframe
    auto_close: bool = True        # închide fereastra după ultimul cadru
):
    """
    RO: Animează traiectoriile particulelor (2D/3D) în VPython.

    Parametri
    ---------
    r_series : np.ndarray, shape (N, T+1, D), D ∈ {2, 3}
        Pozițiile tuturor particulelor la momentele 0..T (în metri).
    step_dt : float
        Pasul de timp Δt [s]. (Animația rulează cu rate(60); nu sincronizăm “în timp real”.)
    viz_n : int
        Câte particule max randăm (pentru FPS bun).
    viz_scale : float
        0.0 => auto-scale (Box dacă există pereți; altfel RMS final ≈ 5 unități vizuale);
        >0  => zoom fix (doar grafic).
    viz_trail : bool
        Trasee VPython (utile la reflexii).
    walls : list[(n, c)] sau None
        Pereți plan: (n, c) cu n vector normal unit (în D dim), planul n·r = c [m].
    show_planes : bool
        Dacă True, desenează și plane translucide (pe lângă wireframe). Implicit False.
    auto_close : bool
        Dacă True, închide fereastra VPython la final (batch-friendly). Pentru demo, pune False.

    Observații
    ----------
    - Scalarea grafică nu modifică fizica; e DOAR pentru afișare.
    - Când există o cutie (toți pereții), folosim mărimea cutiei pentru cadrare.
    - Altfel, cădem pe RMS-ul norului.
    """
    try:
        from vpython import canvas, vector, sphere, color, rate, box, scene, curve
    except Exception as e:
        raise RuntimeError('VPython is not installed. Run `pip install vpython`.') from e

    N, T_plus_1, D = r_series.shape
    if D not in (2, 3):
        raise ValueError(f"animate_vpython: D must be 2 or 3, got {D}")

    M = min(viz_n, N)

    # --- AUTO-SCALE -----------------------------------------------------------
    box_based_scale = None
    mins = maxs = None
    if walls:
        try:
            mins, maxs = _box_from_walls(walls, D)
        except Exception:
            mins = maxs = None
        if mins is not None and maxs is not None:
            extents = maxs - mins
            half_extent = 0.5 * float(np.max(extents))
            if half_extent > 0:
                box_based_scale = 4.0 / half_extent  # cutia încape în ~±4 unități vizuale

    if viz_scale == 0.0:
        if box_based_scale is not None:
            viz_scale = box_based_scale
        else:
            rr = np.sqrt((r_series[:, -1, :] ** 2).sum(axis=1))  # |r_i(T)|
            rms = rr.mean() if rr.size else 1.0
            viz_scale = 5.0 / (rms + 1e-30)
    # --------------------------------------------------------------------------

    # Scena VPython
    scene = canvas(
        title='Specular / Brownian / Langevin',
        width=900, height=600,
        background=color.black
    )

    # Desenăm cutia ca wireframe (dacă o putem reconstrui din pereți)
    if mins is not None and maxs is not None:
        if D == 3:
            draw_wire_box(
                float(mins[0]), float(maxs[0]),
                float(mins[1]), float(maxs[1]),
                float(mins[2]), float(maxs[2]),
                viz_scale, vector, curve, color
            )
        else:  # D == 2
            draw_wire_rect(
                float(mins[0]), float(maxs[0]),
                float(mins[1]), float(maxs[1]),
                viz_scale, vector, curve, color
            )

    # (Opțional) și plane translucide pentru fiecare perete
    if show_planes and walls:
        for (n, c) in walls:
            n = np.asarray(n, float)
            axis = int(np.argmax(np.abs(n)))  # axa dominantă a normalului
            if D == 2:
                if axis == 0:      # x = c
                    pos = [c * viz_scale, 0.0, 0.0]; size = [0.02, 10.0, 0.02]
                else:              # y = c
                    pos = [0.0, c * viz_scale, 0.0]; size = [10.0, 0.02, 0.02]
            else:
                if axis == 0:      # x = c
                    pos = [c * viz_scale, 0.0, 0.0]; size = [0.02, 10.0, 10.0]
                elif axis == 1:    # y = c
                    pos = [0.0, c * viz_scale, 0.0]; size = [10.0, 0.02, 10.0]
                else:              # z = c
                    pos = [0.0, 0.0, c * viz_scale]; size = [10.0, 10.0, 0.02]
            box(pos=vector(*pos), size=vector(*size), color=color.white, opacity=0.2)

    # Culori pastel reproducibile (RNG separat pentru culori)
    rng = np.random.default_rng(42)
    cols = [vector(float(r), float(g), float(b)) for r, g, b in rng.uniform(0.5, 1.0, size=(M, 3))]

    # Particule
    balls = []
    for i in range(M):
        balls.append(
            sphere(
                pos=_vec_from_point(r_series[i, 0], viz_scale, D, vector),
                radius=0.05, color=cols[i],
                make_trail=viz_trail, retain=300
            )
        )

    # Animația (≈60 FPS)
    for k in range(1, T_plus_1):
        rate(60)
        for i in range(M):
            balls[i].pos = _vec_from_point(r_series[i, k], viz_scale, D, vector)

    # Închidere automată (opțională)
    if auto_close:
        try:
            scene.delete()
        except Exception:
            try:
                scene.visible = False
            except Exception:
                pass


# ---------- 2D plotting (VPython graph + gcurve) ----------

def plot_timeseries_vpython(
    t: np.ndarray,
    series: dict,
    title: str = "Analiză MSD / sum(coord)^2 vs timp",
    xlabel: str = "t [s]",
    ylabel: str = "sum coord^2 (unități SI^2)",
    legend: bool = True
):
    """
    RO: Plotează serii temporale 2D (t, y) folosind VPython (graph + gcurve).
    Această fereastră NU se închide automat (util pentru inspectare manuală).

    Parametri
    ---------
    t : np.ndarray, shape (T+1,)
        Timpul (secunde) sau index (dacă a fost cerut în analiză).
    series : dict[str, np.ndarray]
        {nume_curba -> valori_y}; toate y trebuie să aibă shape (T+1,).
        Ex.: {"sum_x2": yx, "sum_y2": yy, "sum_tot2": ytot}
    title, xlabel, ylabel : str
        Titlu și etichete pentru axe.
    legend : bool
        Legendă textuală minimală (numele curbelor în titlu).
    """
    try:
        from vpython import graph, gcurve, color
    except Exception as e:
        raise RuntimeError('VPython is not installed. Run `pip install vpython`.') from e

    palette = [color.red, color.green, color.blue, color.cyan,
               color.magenta, color.yellow, color.white, color.orange]
    names = list(series.keys())

    full_title = f"{title}  —  {' | '.join(names)}" if legend else title
    g = graph(title=full_title, xtitle=xlabel, ytitle=ylabel,
              width=900, height=600, fast=False)

    curves = [gcurve(graph=g, color=palette[i % len(palette)], label=name)
              for i, name in enumerate(names)]

    # Downsample simplu dacă T e mare (pentru performanță UI)
    T = len(t)
    step = max(1, T // 5000)
    for k in range(0, T, step):
        x = float(t[k])
        for c, name in zip(curves, names):
            c.plot(x, float(series[name][k]))
