# src/simcore/config.py

# Despre NumPy (np):
#  - Tipul de bază este np.ndarray (tablouri numerice N×D).
#  - Operațiile sunt vectorizate și suportă broadcasting (eficient, fără bucle Python).
#  - Zgomotul gaussian se generează cu np.random.normal(size=(N, D)).
#  - Reproductibilitate: setează o singură dată seed-ul (np.random.seed(...)).
#  - Unități: toate mărimile din cod sunt în SI (m, s, kg), iar array-urile
#    păstrează aceste unități implicit prin valori; comentariile indică unitățile.

from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np

@dataclass(frozen=True)
class SimConfig:
    """
    ============================================================================
    CONFIGURAȚIA SIMULĂRII — COMENTARII DE FIZICĂ (RO)
    ============================================================================

    SCOP
    ----
    Reunește toți parametrii *numerici* și *fizici* folosiți în simulare.
    Restul codului citește exclusiv din acest obiect ⇒ setările sunt centralizate,
    ușor de reprodus și de documentat.

    MODELE FIZICE  (selectate cu `scheme`)
    --------------------------------------
    1) "langevin"  (underdamped, cu inerție)
       Sistem stochastic (Euler–Maruyama):
         dv = -(γ/m) v dt + sqrt(2 γ k_B T / m^2) dW,      dr = v dt
       - Echilibru fluctuație–dissipație (zgomot ↔ frecare).
       - Timp caracteristic pentru relaxarea vitezei:
           τ_v = m / γ   [s]
         Alege Δt ≪ τ_v pentru stabilitate/precizie.

    2) "brownian" (overdamped, fără inerție explicită)
       Random walk gaussian:
         r_{n+1} = r_n + sqrt(2 D Δt) η,     D = k_B T / γ   [m^2/s]
       - În d dimensiuni, MSD:  ⟨r^2(t)⟩ = 2 d D t (→ panta per axă: 2D).

    3) "ballistic" (determinist, fără zgomot/frecare)
       Mișcare rectilinie uniformă:  r_{n+1} = r_n + v_n Δt;  v_{n+1} = v_n.

    UNITĂȚI (SI)
    -------------
      m  [kg]      – masa
      γ  [kg/s]    – frecarea vâscoasă
      T  [K]       – temperatura
      k_B[J/K]     – const. Boltzmann
      t, Δt [s]    – timp
      r  [m]       – poziție
      v  [m/s]     – viteză
    Vectorii/matricile sunt `np.ndarray`; formele: r, v → (N, dims); init_* → (dims,).

    GHID PRACTIC
    ------------
    • Alege `dt`:
        - Langevin:   Δt ≪ τ_v = m/γ
        - Brownian:   Δx_rms/axă ≈ sqrt(2 D Δt); folosește `viz_scale` pentru vizual,
                       NU mări Δt ca să “se vadă”.
    • `N` mare → statistici mai netede (cost O(N)).
    • `dims` schimbă doar forma vectorilor și factorul din MSD (2·dims·D·t).

    VIZUALIZARE (VPython)
    ---------------------
    Vizualizarea este *pasivă* (doar grafică): nu modifică dinamica.
    `viz_scale=0` face auto-scale pe RMS-ul norului la ultimul pas (~5 unități vizuale).

    Fișiere rezultate:
      - mean_position.dat   : t, x_mean[, y_mean, z_mean]
      - mean_dispersion.dat : t, std_x[, std_y, std_z], radial_rms
    """

    # -----------------------
    # NUCLEU (DISCRETIZARE)
    # -----------------------

    N: int = 10000
    # Numărul de particule. Afectează costul (O(N)) și calitatea statisticilor,
    # nu intră direct în ecuațiile fizice.

    steps: int = 2000
    # Numărul de pași de timp; timpul total simulat este ~ steps * dt.

    dt: float = 1e-3
    # Pasul de timp Δt [s].
    # Folosire:
    #   - Langevin:  v_{n+1} = v_n - (γ/m) v_n Δt + (sqrt(2 γ k_B T)/m) * sqrt(Δt) * η
    #                r_{n+1} = r_n + v_{n+1} Δt
    #   - Brownian:  r_{n+1} = r_n + sqrt(2 D Δt) * η,  D = k_B T / γ
    #   cu η ~ N(0, I) i.i.d. pe axe.
    # Stabilitate:
    #   - underdamped: Δt ≪ τ_v = m/γ;
    #   - overdamped:  Δt reglează scala incrementelor (nu există dinamică “deterministă”).

    dims: int = 3
    # Dimensionalitatea spațiului (1/2/3). În overdamped: ⟨r^2⟩ = 2 * dims * D * t.

    # --------------
    # FIZICĂ (SI)
    # --------------

    m: float = 4.19e-15
    # Masa particulei [kg].
    # Intră în:
    #   - Langevin:  termenul de frânare -(γ/m) v și zgomotul pe viteză σ_v = sqrt(2 γ k_B T)/m
    #   - τ_v = m/γ (timp de relaxare a vitezei)
    # Nu apare în schema overdamped (inerția e neglijată).

    gamma: float = 1.88e-8
    # Coeficient de frecare [kg/s].
    # Intră în:
    #   - Langevin: -(γ/m) v și σ_v
    #   - Brownian: D = k_B T / γ
    # γ ↑ ⇒ D ↓ (difuzie mai lentă) și τ_v ↓ (viteza se relaxează mai repede).

    T: float = 300.0
    # Temperatura [K]. Zgomotul termic crește cu T:
    #   - Langevin: σ_v ∝ sqrt(T)
    #   - Brownian: D ∝ T

    kB: float = 1.380649e-23
    # Constanta lui Boltzmann [J/K]. Intră în σ_v și D.

    # -------------------
    # CONDIȚII INIȚIALE
    # -------------------

    init_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Poziția inițială comună (shape = (dims,), [m]).
    # Conform cerinței “a)”: toate particulele pornesc din același punct.

    init_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Viteza inițială (shape = (dims,), [m/s]).
    # Implicit 0; ignorată în “brownian”.

    # Mod de distribuție inițială a pozițiilor
    init_dist: str = "point"
    # 'point'   → toate exact la init_pos;
    # 'jitter'  → gaussian în jurul init_pos, σ=init_jitter (i.i.d. pe axe);
    # 'uniform' → uniform în cutie (aria/volumul definit de `box`).

    init_jitter: float = 0.0
    # Deviația standard [m] pentru jitter-ul gaussian de la t=0 (valabil doar dacă init_dist='jitter').

    box: Optional[Tuple[float, ...]] = None
    # Definiția geometriei închise pentru inițializare uniformă + pereți (dacă se folosesc):
    #  - dims=2: (xmin, xmax, ymin, ymax)
    #  - dims=3: (xmin, xmax, ymin, ymax, zmin, zmax)
    # Pentru init_dist='uniform', `box` este obligatoriu.

    # ------------
    # ALGORITM
    # ------------

    scheme: str = "langevin"
    # "langevin"  → underdamped (cu inerție explicită, folosește m, v)
    # "brownian"  → overdamped (random walk gaussian; v ignorată)
    # "ballistic" → mișcare rectilinie uniformă (test reflexii/validări geometrice)

    # ----------------
    # PEREȚI / COLIZIUNI
    # ----------------

    enable_walls: bool = False
    # Dacă True, se activează tratamentul de coliziuni cu pereții definiți mai jos.

    walls: Tuple[Tuple[float, float, float, float], ...] = tuple()
    # Listă de planuri în forma (nx, ny, nz, c), cu normal unit (||n||=1) și c în [m]:
    #   planul:  n · r = c
    # Exemplu cutie 3D: 6 plane (x=xmin/xmax, y=ymin/ymax, z=zmin/zmax).
    # Reflexia este *speculară* (elastică): v' = v - 2 (v·n) n.
    # Notă: pentru dims=2, componenta lipsă a lui n se pune 0.

    test_specular: bool = False
    # Flag intern pentru presetul de validare “unghi de 30°” (fără zgomot):
    #   - setează schema 'ballistic' și un perete plan,
    #   - raportează unghiurile de incidență/reflexie la primul impact.

    seed: Optional[int] = None
    # Sămânța RNG (reproductibilitate). Afectează doar zgomotele (η) și inițializările aleatoare.

    # --------
    # OUTPUT
    # --------

    output_dir: str = "output"
    # Directorul în care sunt scrise fișierele .dat (TSV).

    # ---------------
    # VIZUALIZARE
    # ---------------

    enable_vpython: bool = False
    # Dacă True, pornește animația VPython după simulare (pasivă, doar grafică).

    viz_n: int = 100
    # Număr maxim de particule afișate (pentru performanță în VPython).

    viz_scale: float = 0.0   # 0 => auto-scale
    # Factor *strict grafic* (nu afectează datele). 0 → auto-scale pe RMS final ~ 5 unități.

    viz_trail: bool = False
    # Dacă True, desenează trasee (trails) ale particulelor în animație.
