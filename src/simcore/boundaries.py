# src/simcore/boundaries.py
import numpy as np

class WallPlane:
    """
    Plan: n·r = c, cu n UNIt (||n||=1). Specular elastic reflection:
      v' = v - 2 (v·n) n
    Detecție continuă (1 coliziune max/pas):
      dacă în pasul [0, dt] semnul lui (n·r - c) se schimbă ⇒ a existat impact.
      t_hit = (c - n·r0) / (n·v), valid dacă 0 < t_hit <= dt și (n·v) ≠ 0.
    """
    def __init__(self, n: np.ndarray, c: float):
        n = np.asarray(n, float)
        n_norm = np.linalg.norm(n)
        if n_norm == 0:
            raise ValueError("Normal vector must be non-zero.")
        self.n = n / n_norm
        self.c = float(c)

    def collide_step(self, r: np.ndarray, v: np.ndarray, dt: float):
        """
        r, v: (N, D). Returnează (r_next, v_next) după 0..1 coliziuni în acest pas.
        Pentru simplitate educatională: tratăm max 1 impact/pas/particulă.
        """
        n, c = self.n, self.c
        N = r.shape[0]
        r_next = r + v * dt
        v_next = v.copy()

        # semne la început și sfârșit
        s0 = (r @ n) - c                   # (N,)
        s1 = (r_next @ n) - c              # (N,)
        crossed = (s0 * s1 <= 0)           # a trecut prin plan sau a atins

        # doar particulele cu viteză “spre” plan pot lovi
        nv = (v @ n)                       # (N,)
        candidates = crossed & (nv != 0.0)

        idx = np.where(candidates)[0]
        for i in idx:
            t_hit = (c - r[i].dot(n)) / nv[i]
            if 0.0 <= t_hit <= dt:
                # poziție la impact
                r_hit = r[i] + v[i] * t_hit
                # reflectăm viteza
                v_ref = v[i] - 2.0 * (v[i].dot(n)) * n
                # timp rămas
                dt_left = dt - t_hit
                # propagăm cu viteza reflectată
                r_end = r_hit + v_ref * dt_left
                r_next[i] = r_end
                v_next[i] = v_ref
            # altfel: fie glisează chiar pe plan (t_hit ~ 0/neg), fie n-a atins în acest pas
        return r_next, v_next
