# src/simcore/vis.py

# Despre NumPy (np):
#  - Tipul de bază este np.ndarray (tablouri numerice N×D).
#  - Operațiile sunt vectorizate și suportă broadcasting (eficient, fără bucle Python).
#  - Zgomotul gaussian se generează cu np.random.normal(size=(N, D)).
#  - Reproductibilitate: setează o singură dată seed-ul (np.random.seed(...)).
#  - Unități: toate mărimile din cod sunt în SI (m, s, kg), iar array-urile
#    păstrează aceste unități implicit prin valori; comentariile indică unitățile.

"""
VIZUALIZARE VPYTHON (OPȚIONALĂ)
===============================
Scop:
  - să vedem “norul” de particule în mișcare, fără a schimba fizica.
  - pozițiile se pot scala DOAR pentru ecran (viz_scale), ca să fie vizibile.

Important:
  - Vizualizarea este *pasivă*: nu afectează datele salvate în .dat.
  - În unități SI, deplasările Browniene pot fi extrem de mici (μm, nm) pentru
    Δt scurt; de aceea se folosește un factor de scalare grafică (viz_scale).
  - Dacă viz_scale = 0 -> auto-scale astfel încât RMS-ul distanței față de origine
    la ultimul pas să fie ~ 5 unități vizuale (doar pentru “cadrare”).

Despre NumPy:
  - r_series este un np.ndarray de formă (N, T+1, D).
  - Folosim vectorizare acolo unde e natural (dar VPython cere update per-sferă).

Dependență:
  - Necesită `vpython` instalat (pip install vpython).
"""

from __future__ import annotations
import numpy as np

def animate_vpython(
    r_series,
    step_dt,
    viz_n: int = 100,
    viz_scale: float = 0.0,
    viz_trail: bool = False
):
    """
    Animează traiectoriile particulelor folosind VPython.

    Parametri
    ---------
    r_series : np.ndarray, shape (N, T+1, D)
        Pozițiile tuturor particulelor pentru toate momentele (0..T).
        Unități: [m]. D ar trebui să fie 3 pentru o scenă 3D.
    step_dt : float
        Pasul de timp Δt [s]. (Notă: aici e informativ; VPython derulează la
        ~rate(60), nu “sincronizează” efectiv cu Δt. Dacă vrei sincronizare
        reală, poți adapta rate în funcție de Δt, dar nu este necesar vizual.)
    viz_n : int
        Câte particule maximum să afișăm (pentru performanță). Din r_series se
        vor lua M = min(viz_n, N).
    viz_scale : float
        Factorul *doar grafic* de scalare a coordonatelor pentru ecran.
        - 0.0 (implicit) => auto-scale: RMS-ul norului la ultimul pas ≈ 5 unități vizuale.
        - >0.0           => folosește valoarea dată ca zoom (nu afectează fizica).
    viz_trail : bool
        Dacă True, VPython va desena trasee (“make_trail”) pentru fiecare sferă.

    Observații
    ----------
    • Fizica nu este alterată: scalarea se aplică *după* simulare, doar la randare.
    • Performanță: desenarea M sferelor la 60 FPS poate fi costisitoare pentru M mare;
      încearcă viz_n ~ 50–300.
    • Reproducibilitate culori: folosim un RNG local pentru culori pastel (nu afectează
      RNG-ul simulării, care a fost setat în Simulator).
    """
    try:
        from vpython import canvas, vector, sphere, color, rate
    except Exception as e:
        # Mesaj prietenos dacă librăria nu e instalată.
        raise RuntimeError('VPython is not installed. Run `pip install vpython`.') from e

    # Extragem dimensiunile seriei: N particule, T+1 momente, D axe.
    N, T_plus_1, D = r_series.shape

    # Alegem câte particule vizualizăm (pentru a menține FPS bun).
    M = min(viz_n, N)

    # --- AUTO-SCALE -----------------------------------------------------------
    # Dacă viz_scale == 0.0, calculăm un factor care aduce RMS-ul distanței la
    # ~5 unități vizuale în ultimul frame (ca să încadrăm frumos scena).
    # Notă: folosim distanța față de originea globală (nu față de medie). Pentru
    # mișcări libere fără drift, media rămâne aproape 0; altfel poți schimba ușor
    # pe |r - mean| dacă vrei.
    if viz_scale == 0.0:
        rr = np.sqrt((r_series[:, -1, :] ** 2).sum(axis=1))  # |r_i(T)|
        rms = rr.mean() if rr.size else 1.0
        viz_scale = 5.0 / (rms + 1e-30)  # evităm împărțirea la 0
    # --------------------------------------------------------------------------

    # Scena VPython (fundal negru pentru contrast, fereastră 900x600)
    scene = canvas(
        title='Brownian/Langevin 3D',
        width=900, height=600,
        background=color.black
    )

    # Culori pastel random pentru diferențiere vizuală (RNG local: nu afectează simularea)
    rng = np.random.default_rng(42)
    cols = [vector(float(r), float(g), float(b)) for r, g, b in rng.uniform(0.5, 1.0, size=(M, 3))]

    # Inițializăm sferele. Radius mic (~0.05) e potrivit cu scale vizuale implicite.
    balls = []
    for i in range(M):
        # Aplicăm scalarea DOAR la randare.
        x, y, z = (r_series[i, 0] * viz_scale).tolist()
        balls.append(
            sphere(
                pos=vector(x, y, z),
                radius=0.05,
                color=cols[i],
                make_trail=viz_trail,
                retain=300  # câte puncte de trail păstrăm (compromis memorie/vizual)
            )
        )

    # Redăm în timp: rate(60) ≈ 60 FPS (nu echivalează cu 1/Δt în secunde reale).
    for k in range(1, T_plus_1):
        rate(60)
        for i in range(M):
            x, y, z = (r_series[i, k] * viz_scale).tolist()
            balls[i].pos = vector(x, y, z)
