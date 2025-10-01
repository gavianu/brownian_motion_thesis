# src/simcore/simulator.py

# Despre NumPy (np):
#  - Tipul de bază este np.ndarray (tablouri numerice N×D).
#  - Operațiile sunt vectorizate și suportă broadcasting (eficient, fără bucle Python).
#  - Zgomotul gaussian se generează cu np.random.normal(size=(N, D)).
#  - Reproductibilitate: setează o singură dată seed-ul (np.random.seed(...)).
#  - Unități: toate mărimile din cod sunt în SI (m, s, kg), iar array-urile
#    păstrează aceste unități implicit prin valori; comentariile indică unitățile.

import numpy as np
from .config import SimConfig
from .models import Ensemble
from .metrics import mean_position, dispersion
from .io import DataWriter
from .physics import LangevinEM, BrownianOverdamped
from .vis import animate_vpython

class Simulator:
    """
    ORCHESTRATORUL SIMULĂRII
    ------------------------
    Responsabilități (SRP):
      • inițializează ansamblul (r, v) pe baza configurației (SimConfig);
      • alege *strategia* de integrare (Langevin/Brownian) și rulează bucla de timp;
      • calculează statisticile cerute (media pozițiilor și împrăștierea) la fiecare pas;
      • scrie fișierele .dat prin DataWriter;
      • (opțional) pornește vizualizarea VPython.

    Pattern folosit:
      • Strategy — integratorul este selectat la runtime în funcție de `cfg.scheme`,
        dar Simulator nu depinde de implementarea concretă (doar de interfața `.step`).

    Conveții:
      • Toate mărimile sunt în SI: r[m], v[m/s], t[s].
      • r și v au shape (N, D); `times` are shape (T+1,).
      • Fără pereți/coliziuni: spațiul e liber (cerința a)).
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

        # RNG (opțional) pentru reproducibilitate.
        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        # Starea ansamblului: r, v (shape (N, D))
        self.ensemble = Ensemble(cfg.N, cfg.dims, cfg.init_pos, cfg.init_vel)

        # Alegem integratorul în funcție de schemă (Strategy).
        #  - LangevinEM: underdamped (are viteză explicită, inerție)
        #  - BrownianOverdamped: overdamped (random walk gaussian)
        self.integrator = (
            LangevinEM(cfg.m, cfg.gamma, cfg.kB, cfg.T)
            if cfg.scheme == 'langevin'
            else BrownianOverdamped(cfg.gamma, cfg.kB, cfg.T)
        )

        # I/O: scriere fișiere .dat (TSV)
        self.writer = DataWriter(cfg.output_dir)

    def run(self):
        """
        Rulează simularea pe un orizont de timp T = steps * dt.
        Returnează căile către fișierele scrise:
          - mean_position.dat
          - mean_dispersion.dat
        """
        N, D, T, dt = self.cfg.N, self.cfg.dims, self.cfg.steps, self.cfg.dt

        # Vectorul timpilor: t_k = k * dt, k=0..T  (shape (T+1,))
        times = np.arange(T + 1) * dt

        # Buffere pentru statistici (umplute pe măsură ce rulăm)
        #  means[k]      -> (D,)   media poziției pe axe la pasul k
        #  stds_series[k]-> (D,)   deviația standard pe axe la pasul k
        #  radial_series[k] -> ()  RMS radial față de medie (vezi metrics.py)
        means = np.zeros((T + 1, D))
        stds_series = np.zeros((T + 1, D))
        radial_series = np.zeros(T + 1)

        # Snapshot inițial (k = 0)
        r = self.ensemble.positions()
        v = self.ensemble.velocities()
        m = mean_position(r)
        means[0] = m
        d = dispersion(r, m)
        stds_series[0] = d['stds']
        radial_series[0] = d['radial_rms']

        # Dacă vrem vizualizare, memorăm traiectoriile (atenție la memorie: O(N*T*D))
        r_series = None
        if self.cfg.enable_vpython:
            r_series = np.zeros((N, T + 1, D))
            r_series[:, 0, :] = r

        # --- RULE OF THUMB (stabilitate & sanity checks) -----------------------
        # • LANGEVIN (underdamped):
        #     - Alege Δt ≪ τ_v = m/γ. Ca ordine de mărime, Δt ∈ [τ_v/1000, τ_v/50].
        #       Ex.: dacă m/γ = 2e-7 s, un Δt sigur ar fi ~ 1e-9 … 4e-9 s.
        #     - La t ≪ τ_v: regim balistic (⟨Δx^2⟩ ~ (k_B T / m) t^2).
        #       La t ≫ τ_v: regim difuziv (⟨Δx^2⟩ ~ 2 D t, cu D = k_B T/γ).
        #     - Verificări utile (dacă salvezi v pe parcurs):
        #         Var[v_x] ≈ k_B T / m   (echipartiție, pe axă)
        #         VACF(t) ≈ exp(-t/τ_v)  (autocorelația vitezei)
        #
        # • BROWNIAN (overdamped):
        #     - D = k_B T / γ  (m^2/s). Așteaptă: std_x(t)^2 ≈ 2 D t (pe fiecare axă).
        #     - În d dimensiuni: ⟨r^2(t)⟩ ≈ 2 d D t.
        #     - În 3D, media modulară E[|r|] (nu RMS!) ≈ 2.25676 √(D t).
        #     - Diagnostic numeric:
        #         – plotează std_x(t)^2 vs t → linie cu panta ≈ 2 D;
        #         – plotează radial_rms(t) în scale log-log → pantă ≈ 1/2.
        #
        # • PRACTIC:
        #     - Dacă nu “se vede” nimic în VPython, NU mări Δt ca să forțezi vizualul.
        #       Folosește 'viz_scale' (0 = auto) — vizualizarea nu schimbă fizica.
        #     - Pentru statistici mai netede folosește N mai mare (cost O(N)).
        # -----------------------------------------------------------------------
        # Bucla de timp: k = 1..T
        # Fiecare pas:
        #   1) integrator.step(...) -> (r_next, v_next)
        #   2) actualizăm Ensemble
        #   3) calculăm statisticile și (opțional) salvăm pentru vizualizare
        for k in range(1, T + 1):
            r, v = self.integrator.step(r, v, dt)
            self.ensemble.set_state(r, v)

            m = mean_position(r)
            means[k] = m
            d = dispersion(r, m)
            stds_series[k] = d['stds']
            radial_series[k] = d['radial_rms']

            if r_series is not None:
                r_series[:, k, :] = r

        # Scriere fișiere .dat (TSV)
        mp = self.writer.write_mean_positions(times, means)
        md = self.writer.write_dispersion(times, stds_series, radial_series)

        # Vizualizare (opțională): NU afectează datele scrise; doar scalează pe ecran.
        if r_series is not None:
            animate_vpython(
                r_series, dt,
                viz_n=self.cfg.viz_n,
                viz_scale=self.cfg.viz_scale,
                viz_trail=self.cfg.viz_trail
            )

        return mp, md
