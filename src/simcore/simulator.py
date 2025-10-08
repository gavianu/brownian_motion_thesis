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
from .physics import LangevinEM, BrownianOverdamped, Ballistic
from .vis import animate_vpython
from .boundaries import WallPlane

class Simulator:
    """
    ORCHESTRATORUL SIMULĂRII (SRP)
    ------------------------------
    • Inițializează ansamblul (r, v) din SimConfig.
    • Alege integratorul (Strategy): Langevin/Brownian/Ballistic.
    • Rulează bucla de timp, aplică pereți (dacă există).
    • Calculează metrici (medii, împrăștiere) și scrie fișiere .dat.
    • Opțional: animă traiectoriile în VPython.

    Conveții SI: r[m], v[m/s], t[s]; shape r, v: (N, D).
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        if cfg.seed is not None:
            np.random.seed(cfg.seed)

        # Stare ansamblu (toate particulele la init_pos, v=init_vel, by default)
        self.ensemble = Ensemble(cfg.N, cfg.dims, cfg.init_pos, cfg.init_vel)

        # Integrator (Strategy)
        if cfg.scheme == 'langevin':
            self.integrator = LangevinEM(cfg.m, cfg.gamma, cfg.kB, cfg.T)
        elif cfg.scheme == 'brownian':
            self.integrator = BrownianOverdamped(cfg.gamma, cfg.kB, cfg.T)
        elif cfg.scheme == 'ballistic':
            self.integrator = Ballistic()
        else:
            raise ValueError("Unknown scheme.")

        # Pereți (planuri n·r=c)
        self.walls = []
        if cfg.enable_walls and cfg.walls:
            for (nx, ny, nz, c) in cfg.walls:
                n = np.array([nx, ny, nz][:cfg.dims], float)
                self.walls.append(WallPlane(n, c))

        # Writer .dat (TSV)
        self.writer = DataWriter(cfg.output_dir)

        # --- INIȚIALIZARE POZIȚII după init_dist --------------------------------
        # NOTĂ: pentru testul specular (validare unghi 30°) păstrăm *exact* init_pos
        # pentru toate particulele (fără jitter/uniform). De aceea nu aplicăm nimic
        # suplimentar când cfg.test_specular=True.
        if not cfg.test_specular:
            self._apply_initial_distribution()

        # --- STAGGER (doar în modul de test specular) ---------------------------
        # Ideea: păstrăm același init_pos/init_vel pentru toate particulele,
        # dar le "lansăm" pe rând, la pași consecutivi, ca să nu se suprapună vizual.
        self._stagger_enabled = bool(cfg.test_specular)
        if self._stagger_enabled:
            # particula i devine activă în pasul k = i*GAP (GAP=2 → lansăm câte una la 2 frame-uri)
            self._stagger_gap = 2
            self._start_step = np.arange(cfg.N, dtype=int) * self._stagger_gap
        # -----------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Inițializare poziții conform init_dist:
    #  - 'point'   : nimic de făcut (toate deja la init_pos).
    #  - 'jitter'  : r0 = init_pos + N(0, σ^2 I) i.i.d. pe axe (σ=init_jitter).
    #  - 'uniform' : r0 uniform în cutia axis-aligned definită de cfg.box.
    # -------------------------------------------------------------------------
    def _apply_initial_distribution(self):
        cfg = self.cfg
        N, D = cfg.N, cfg.dims

        if cfg.init_dist == 'point':
            return  # deja setat în Ensemble

        r0 = self.ensemble.positions().copy()

        if cfg.init_dist == 'jitter':
            if cfg.init_jitter <= 0.0:
                raise ValueError("init_dist='jitter' necesită init_jitter > 0.")
            # gaussian i.i.d. pe axe; norul inițial are std = σ [m] pe fiecare axă
            r0 += np.random.normal(loc=0.0, scale=cfg.init_jitter, size=r0.shape)

        elif cfg.init_dist == 'uniform':
            if cfg.box is None:
                raise ValueError("init_dist='uniform' necesită box definit în Config.")
            b = cfg.box
            if D == 3:
                xmin, xmax, ymin, ymax, zmin, zmax = b
                x = np.random.uniform(xmin, xmax, size=N)
                y = np.random.uniform(ymin, ymax, size=N)
                z = np.random.uniform(zmin, zmax, size=N)
                r0 = np.column_stack([x, y, z])
            elif D == 2:
                xmin, xmax, ymin, ymax = b
                x = np.random.uniform(xmin, xmax, size=N)
                y = np.random.uniform(ymin, ymax, size=N)
                r0 = np.column_stack([x, y])
            else:  # D == 1
                xmin, xmax = b[:2]
                x = np.random.uniform(xmin, xmax, size=N)
                r0 = x.reshape(N, 1)

        else:
            raise ValueError(f"init_dist necunoscut: {cfg.init_dist}")

        # Commit inițializare în Ensemble (vitezele rămân cele din config)
        self.ensemble.set_state(r0, self.ensemble.velocities())

    def _apply_boundaries(self, r, v, dt):
        # aplicăm pe rând fiecare perete, cu CCD 0..1 impact/pas/perete
        for wall in self.walls:
            r, v = wall.collide_step(r, v, dt)
        return r, v

    def run(self):
        """
        Rulează simularea pe un orizont de timp T = steps * dt.
        Returnează căile către fișierele scrise:
          - mean_position.dat
          - mean_dispersion.dat
        """
        N, D, T, dt = self.cfg.N, self.cfg.dims, self.cfg.steps, self.cfg.dt
        times = np.arange(T + 1) * dt

        means = np.zeros((T + 1, D))
        stds_series = np.zeros((T + 1, D))
        radial_series = np.zeros(T + 1)

        # Snapshot inițial (după aplicarea init_dist, dacă e cazul)
        r = self.ensemble.positions()
        v = self.ensemble.velocities()
        m = mean_position(r); means[0] = m
        d = dispersion(r, m); stds_series[0] = d['stds']; radial_series[0] = d['radial_rms']

        # Traiectorii pentru VPython (atenție memorie)
        r_series = None
        if self.cfg.enable_vpython:
            r_series = np.zeros((N, T + 1, D))
            r_series[:, 0, :] = r

        # Pentru raportarea unghiurilor în testul specular (o singură dată/particulă)
        printed = np.zeros(N, dtype=bool)

        # Copii ale IC-urilor (re-activăm la stagger)
        init_pos = self.cfg.init_pos.reshape(1, D)
        init_vel = self.cfg.init_vel.reshape(1, D)

        for k in range(1, T + 1):
            # --- STAGGER ACTIVATION (doar în test_specular) ---
            if self._stagger_enabled:
                to_activate = (self._start_step == k)
                if np.any(to_activate):
                    # lansăm particulele respective exact în acest frame, din IC
                    r[to_activate, :] = init_pos
                    v[to_activate, :] = init_vel

            v_before = v.copy()

            # Integrare 1 pas
            r_new, v_new = self.integrator.step(r, v, dt)

            # Pereți (specular elastic)
            if self.walls:
                r_new, v_new = self._apply_boundaries(r_new, v_new, dt)

            # DEBUG: unghi incidență/reflexie (doar în test_specular)
            if self.cfg.test_specular and self.walls:
                n = self.walls[0].n  # presupunem un singur perete în test
                nv_before = np.einsum('nd,d->n', v_before, n)
                nv_after  = np.einsum('nd,d->n', v_new,    n)
                collided = (nv_before < 0) & (nv_after > 0) & (~printed)
                idx = np.where(collided)[0]
                for i in idx[:10]:
                    v_in = v_before[i]; v_out = v_new[i]
                    def angle_to_normal(u):
                        un = abs(np.dot(u, n)) / (np.linalg.norm(u) + 1e-30)
                        un = np.clip(un, 0.0, 1.0)
                        return np.degrees(np.arccos(un))
                    th_in  = angle_to_normal(v_in)
                    th_out = angle_to_normal(v_out)
                    print(f"[SPECULAR] id={i}  theta_in={th_in:.2f}°,  theta_out={th_out:.2f}°  (≈ egale)")
                    printed[i] = True

            # Commit pas
            r, v = r_new, v_new
            self.ensemble.set_state(r, v)

            # Metrici
            m = mean_position(r); means[k] = m
            d = dispersion(r, m); stds_series[k] = d['stds']; radial_series[k] = d['radial_rms']

            # Stocare pentru viz
            if r_series is not None:
                r_series[:, k, :] = r

        # Scriere fișiere .dat
        mp = self.writer.write_mean_positions(times, means)
        md = self.writer.write_dispersion(times, stds_series, radial_series)

        # Vizualizare (opțională) + forțare închidere (redundanță vs vis.py)
        if r_series is not None:
            walls = [(w.n, w.c) for w in self.walls]
            animate_vpython(r_series, dt, viz_n=self.cfg.viz_n,
                            viz_scale=self.cfg.viz_scale, viz_trail=self.cfg.viz_trail,
                            walls=walls)
            # --- redundanță de siguranță: forțăm închiderea ferestrei VPython ---
            try:
                from vpython import scene
                try:
                    scene.delete()
                except Exception:
                    scene.visible = False
            except Exception:
                pass

        return mp, md
