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
    Codul din restul modulului citește exclusiv din acest obiect, astfel încât:
      - să fie clar *de unde vin* valorile,
      - să poți reproduce/partaja ușor setările.

    MODELE FIZICE  (selectate cu `scheme`)
    --------------------------------------
    1) "langevin"  (underdamped, cu inerție)
       Ecuație (pe viteză) + poziție:
         dv = -(γ/m) v dt + sqrt(2 γ k_B T / m^2) dW,      dr = v dt
       - Zgomotul termic (∝ sqrt(T)) și frânarea vâscoasă (∝ γ) sunt în
         *echilibru fluctuație–dissipație*.
       - Timp caracteristic pentru relaxarea vitezei:
           τ_v = m / γ     [s]
         În practică, alege Δt ≪ τ_v pentru stabilitate/precizie.

    2) "brownian" (overdamped, fără inerție explicită)
       Random walk gaussian cu difuzie:
         r_{n+1} = r_n + sqrt(2 D Δt) η,     D = k_B T / γ   [m^2/s]
       - În d dimensiuni, MSD teoretic:  ⟨r^2(t)⟩ = 2 d D t.
       - Pentru d = 3:                   ⟨r^2(t)⟩ = 6 D t.

    UNITĂȚI (SI)
    -------------
      m  [kg]      – masa
      γ  [kg/s]    – frecarea vâscoasă
      T  [K]       – temperatura
      k_B[J/K]     – const. Boltzmann
      t, Δt [s]    – timp
      r  [m]       – poziție
      v  [m/s]     – viteză
    Toate vectorii și matricile sunt `np.ndarray` (NumPy) în unități SI; formele:
          r, v -> (N, dims); init_pos/init_vel -> (dims,).

    GHID PRACTIC
    ------------
    • Alege `dt`:
        - Langevin:   Δt ≪ τ_v = m/γ
        - Brownian:   Δx_rms/axă ≈ sqrt(2 D Δt). Dacă vizual nu “se vede”,
                       nu mări Δt ca să falsifici fizica; folosește `viz_scale`.
    • `N` mare → statistici (medii/dispersii) mai stabile, dar cost O(N).
    • `dims` afectează doar forma vectorilor și factorul din MSD (2·dims·D·t).

    NOTĂ DESPRE IMAGINĂRI
    ---------------------
    Vizualizarea (VPython) *nu modifică* dinamica; doar scalează coordonatele pe
    ecran (`viz_scale`) ca să vezi norul (în SI, la T ~ 300 K deplasările pot fi
    extrem de mici la scări de milisecundă).

    Fișiere rezultate:
      - mean_position.dat   : t, x_mean[, y_mean, z_mean]
      - mean_dispersion.dat : t, std_x[, std_y, std_z], radial_rms
      
    """

    # -----------------------
    # NUCLEU (DISCRETIZARE)
    # -----------------------

    N: int = 10000
    # Numărul de particule din ansamblu.
    # -> Definește dimensiunea matricilor de stare r, v (forme (N, dims)).
    # -> Nu intră într-o formulă fizică; influențează doar statisticile și costul.

    steps: int = 2000
    # Numărul de pași de timp (T). Timpul total simulat ≈ steps * dt.

    dt: float = 1e-3
    # Pasul de timp Δt [s].
    # Folosire în scheme:
    #   - Langevin (Euler–Maruyama):
    #       v_{n+1} = v_n - (γ/m) v_n Δt + (sqrt(2 γ k_B T)/m) * sqrt(Δt) * η
    #       r_{n+1} = r_n + v_{n+1} Δt
    #   - Brownian (overdamped):
    #       r_{n+1} = r_n + sqrt(2 D Δt) * η,   D = k_B T / γ
    #   unde η ~ N(0, I) (vector aleator gaussian, shape = (dims,)). 
    # Stabilitate/precizie:
    #   - alege Δt ≪ τ_v = m/γ în underdamped.
    #   - în overdamped, Δt controlează scala incrementelor (nu dinamica “deterministă”).

    dims: int = 3
    # Dimensionalitatea spațiului (1 / 2 / 3).
    # Afectează factorul din MSD teoretic:  ⟨r^2⟩ = 2 * dims * D * t (overdamped).

    # --------------
    # FIZICĂ (SI)
    # --------------

    m: float = 4.19e-15 #  Particulă de 1 μm rază (a = 1e-6 m)
    # Masa particulei [kg].
    # Intră în:
    #   - termenul de frânare din Langevin:  -(γ/m) v
    #   - amplitudinea zgomotului pe viteză: σ_v = sqrt(2 γ k_B T) / m
    #   - timpul de relaxare: τ_v = m / γ
    # Nu apare în schema overdamped (acolo inerția e neglijată).

    gamma: float = 1.88e-8
    # Coeficient de frecare (amortizare) [kg/s].
    # Intră în:
    #   - Langevin: -(γ/m) v  și σ_v = sqrt(2 γ k_B T)/m
    #   - Brownian: coeficientul de difuzie  D = k_B T / γ
    # γ ↑  → D ↓ (difuzie mai lentă), v se relaxează mai repede (τ_v = m/γ ↓).

    T: float = 300.0
    # Temperatura [K]. Intră în zgomotul termic:
    #   - Langevin: σ_v ∝ sqrt(T)
    #   - Brownian: D ∝ T

    kB: float = 1.380649e-23
    # Constanta lui Boltzmann [J/K]. Standard SI.
    # Intră în σ_v și D (vezi formulele de mai sus).

    # -------------------
    # CONDIȚII INIȚIALE
    # -------------------

    init_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Poziția inițială comună tuturor particulelor (shape = (dims,), [m]).
    # Conform cerinței tale “a)”: toate din același punct.

    init_vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Viteza inițială (shape = (dims,), [m/s]).
    # Conform cerinței: inițial 0. Ignorată în schema “brownian” (overdamped).

    # ------------
    # ALGORITM
    # ------------

    scheme: str = "langevin"
    # Alege modelul numeric:
    #   "langevin"  → underdamped (folosește m, v; are inerție explicita)
    #   "brownian"  → overdamped (random walk; v nu se folosește)
    #   "ballistic" → mișcare rectilinie uniformă (fără zgomot/frânare; test)

    # Boundaries
    enable_walls: bool = False

    # listă de pereți (normal, c) cu normalul unit și c în [m]; plan: n·r = c
    walls: Tuple[Tuple[float, float, float, float], ...] = tuple()

    # --- nou: mod de test specular ---
    test_specular: bool = False
    
    seed: Optional[int] = None
    # Sămânța RNG (reproductibilitate). Afectează șirul de zgomote η.

    # --------
    # OUTPUT
    # --------

    output_dir: str = "output"
    # Directorul în care sunt scrise fișierele .dat (TSV).

    # ---------------
    # VIZUALIZARE
    # ---------------

    enable_vpython: bool = False
    # Pornește animația VPython după simulare (opțional, pentru explorare).

    viz_n: int = 100
    # Câte particule afișăm în animație (max). Doar performanță/estetică.

    viz_scale: float = 0.0   # 0 => auto-scale
    # Factor de *scalare grafică* (doar pe ecran). Nu schimbă datele salvate.
    # 0 (implicit) → auto-scale astfel încât RMS-ul norului la final ~ 5 unități.

    viz_trail: bool = False
    # Dacă True, VPython desenează “urme” (trails) — util pentru intuiție vizuală.
