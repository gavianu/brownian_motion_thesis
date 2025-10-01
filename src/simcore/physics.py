# src/simcore/physics.py

# Despre NumPy (np):
#  - Tipul de bază este np.ndarray (tablouri numerice N×D).
#  - Operațiile sunt vectorizate și suportă broadcasting (eficient, fără bucle Python).
#  - Zgomotul gaussian se generează cu np.random.normal(size=(N, D)).
#  - Reproductibilitate: setează o singură dată seed-ul (np.random.seed(...)).
#  - Unități: toate mărimile din cod sunt în SI (m, s, kg), iar array-urile
#    păstrează aceste unități implicit prin valori; comentariile indică unitățile.

import numpy as np

# =============================================================================
# LANGEVIN UNDERDAMPED  (Euler–Maruyama)
# -----------------------------------------------------------------------------
# Ecuația continuă (pe viteză) cu zgomot alb:
#   dv = -(γ/m) v dt + sqrt(2 γ k_B T / m^2) dW,        dr = v dt
#   unde dW este increment Wiener (N(0, dt)), iar ξ(t) = dW/dt ar fi “zgomotul alb”.
#
# Discretizare Euler–Maruyama pe pas Δt:
#   v_{n+1} = v_n - (γ/m) v_n Δt + σ_v sqrt(Δt) η_n,
#   r_{n+1} = r_n + v_{n+1} Δt,
#   cu  σ_v = sqrt(2 γ k_B T) / m,   η_n ~ N(0, I_D) i.i.d.
#
# Observații:
# - Folosim update “semi-implicit” pentru poziție (cu v_{n+1}); adesea e mai robust
#   numeric decât v_{n} în update-ul lui r, dar păstrează simplitatea EM.
# - Timpul caracteristic de relaxare a vitezei: τ_v = m/γ. Pentru stabilitate/precizie,
#   alege Δt ≪ τ_v (altfel apar artefacte numerice în regimul inerțial).
# - Unități (SI): m[kg], γ[kg/s], k_B[J/K], T[K], t[s], r[m], v[m/s].
# - Alternative mai precise pentru Δt mai mari: scheme OU exacte / BAOAB (neimplementate aici).
# =============================================================================
class LangevinEM:
    """Integrator Euler–Maruyama pentru modelul Langevin underdamped în D dimensiuni."""

    def __init__(self, m: float, gamma: float, kB: float, T: float):
        # Parametri fizici:
        #  m     [kg]   – masa particulei
        #  gamma [kg/s] – frecarea (amortizare vâscoasă)
        #  kB    [J/K]  – constanta lui Boltzmann
        #  T     [K]    – temperatura
        self.m, self.gamma, self.kB, self.T = m, gamma, kB, T

        # Amplitudinea zgomotului pe viteză:
        #   σ_v = sqrt(2 γ k_B T) / m     [m/s^{3/2}]
        # iar σ_v * sqrt(Δt) are unități de viteză [m/s].
        self.sigma_v = (2.0 * gamma * kB * T) ** 0.5 / m

    def step(self, r: np.ndarray, v: np.ndarray, dt: float):
        """
        Un pas de integrare EM.

        Parametri
        ---------
        r : (N, D) [m]   – pozițiile curente
        v : (N, D) [m/s] – vitezele curente
        dt: float  [s]   – pas de timp Δt

        Returnează
        ----------
        r_next : (N, D) [m]
        v_next : (N, D) [m/s]
        """
        N, D = r.shape

        # η ~ N(0,1) i.i.d. pe particule și pe axe (fără unități)
        # Incrementul Wiener discret: dW ≈ sqrt(Δt) * η
        eta = np.random.normal(size=(N, D))

        # Update viteză: frânare -(γ/m) v Δt + agitare termică σ_v sqrt(Δt) η
        v_next = v + (-(self.gamma / self.m) * v) * dt + self.sigma_v * (dt ** 0.5) * eta

        # Update poziție: r_{n+1} = r_n + v_{n+1} Δt  (semi-implicit pe r)
        r_next = r + v_next * dt

        return r_next, v_next


# =============================================================================
# BROWNIAN OVERDAMPED  (random walk gaussian)
# -----------------------------------------------------------------------------
# Eliminăm inerția explicită (regim puternic amortizat):
#   r_{n+1} = r_n + sqrt(2 D Δt) η_n,   D = k_B T / γ
#   cu η_n ~ N(0, I_D) i.i.d.
#
# Proprietăți:
# - În d dimensiuni: ⟨r^2(t)⟩ = 2 d D t (regim difuziv).
# - Incrementul pe axă are varianță 2 D Δt (unități [m^2]).
# - v nu are rol aici; păstrăm un array de viteze zero doar pentru compatibilitate interfață.
# =============================================================================
class BrownianOverdamped:
    """Mișcare browniană overdamped (difuzie) în D dimensiuni."""

    def __init__(self, gamma: float, kB: float, T: float):
        # Coeficientul de difuzie:
        #   D = k_B T / γ     [m^2/s]
        self.D = kB * T / gamma

    def step(self, r: np.ndarray, v: np.ndarray, dt: float):
        """
        Un pas de random walk gaussian (overdamped).

        Parametri
        ---------
        r : (N, D) [m]   – pozițiile curente
        v : (N, D) [m/s] – ignorat în overdamped (menținut pentru interfață)
        dt: float [s]    – pas de timp Δt

        Returnează
        ----------
        r_next : (N, D) [m]
        v_next : (N, D) [m/s] (zero-uri; placeholder pentru compatibilitate)
        """
        N, D = r.shape

        # η ~ N(0,1) i.i.d. ; saltul pe poziție are deviație standard sqrt(2 D Δt)
        eta = np.random.normal(size=(N, D))

        # r_{n+1} = r_n + sqrt(2 D Δt) η
        r_next = r + (2.0 * self.D * dt) ** 0.5 * eta

        # v nu e definită în overdamped; întoarcem zeros_like(v) pentru API uniform
        v_next = np.zeros_like(v)

        return r_next, v_next


# =============================================================================
# VERIFICĂRI TEORETICE (ghid numeric – NU schimbă codul)
# -----------------------------------------------------------------------------
# 1) OVERDAMPED (Brownian):
#    • Pe fiecare axă (1D), varianța crește liniar: Var[x(t)] = 2 D t.
#      => std_x(t)^2 ≈ 2 D t  (la timp suficient de lung, N mare pentru statistici).
#    • În d dimensiuni, MSD-ul teoretic:
#         ⟨ r^2(t) ⟩ = 2 d D t.
#      Notă: în fișierul nostru salvăm per-axă std_x, std_y, std_z și “radial_rms”.
#      ATENȚIE: “radial_rms” din cod este de fapt media distanței |r|:
#         radial_rms = E[ || r - r_mean || ]
#      care NU este √⟨r^2⟩ (root-mean-square). Pentru distribuție gaussiană
#      isotropă în 3D cu varianță σ^2 pe axă (σ^2 = 2 D t), |r| are distribuție
#      Maxwell cu:
#         E[|r|] = 2 √(2/π) σ  ≈ 1.59577 · σ
#      => în 3D, așteptăm aproximativ:
#         radial_rms(t) ≈ 2 √(2/π) · √(2 D t) = 4/√π · √(D t) ≈ 2.25676 √(D t).
#      Deci:
#         • std_x^2 ≈ 2 D t  (pe fiecare axă),
#         • E[|r|]  ≈ 2.25676 √(D t) (în 3D).
#
#    Cum verifici practic:
#      - Fă un fit liniar al lui std_x(t)^2 vs t (panta ≈ 2 D).
#      - Fă un fit pe log-log pentru radial_rms(t) vs t (panta ≈ 1/2; interceptul
#        îți dă ~ 2.25676 √D, dacă r_mean ≈ 0).
#
# 2) UNDERDAMPED (Langevin):
#    • Timp caracteristic al vitezelor: τ_v = m / γ.
#    • MSD (1D) exact pentru free Langevin:
#         ⟨Δx(t)^2⟩ = 2 D [ t - τ_v (1 - e^{-t/τ_v}) ],   cu  D = k_B T / γ.
#      - Regim scurt (t ≪ τ_v):     ⟨Δx^2⟩ ≈ (k_B T / m) t^2   (balistic)
#      - Regim lung  (t ≫ τ_v):     ⟨Δx^2⟩ ≈ 2 D t            (difuziv)
#      În d dimensiuni, înmulțești cu d.
#    • Autocorelația vitezei (VACF, 1D) în regim liber:
#         C_v(t) = ⟨ v(0) v(t) ⟩ = (k_B T / m) e^{-t/τ_v}
#      => decay exponențial cu scala τ_v.
#    • Echipartiție (steady-state):
#         ⟨ v^2 ⟩ = k_B T / m   pe fiecare axă.
#
#    Cum verifici practic (fără a schimba codul):
#      - Alege dt ≪ τ_v (altfel nu rezolvi bine regimul balistic).
#      - Poți salva temporar v (ex. în simulator) ca să verifici:
#           Var[v_x] ≈ k_B T / m   și  VACF(t) ~ e^{-t/τ_v}.
#      - Pentru poziții: după t ≫ τ_v, comportamentul pe axă tinde la std_x^2 ≈ 2 D t,
#        la fel ca în overdamped (tranziție balistic → difuziv).
#
# 3) UNITĂȚI / SCALE:
#    • D = k_B T / γ  [m^2/s],  τ_v = m/γ  [s].
#    • std_x^2 are unități [m^2], radial_rms are [m].
#    • Dacă numeric std_x^2 vs t NU e aproape liniar la timp lung în overdamped,
#      sau panta diferă mult de 2D, verifică:
#         - că dt este suficient de mic (stabilitate EM),
#         - că D este calculat corect (parametrii T, γ),
#         - că N este suficient de mare pentru statistici netede.
# =============================================================================
