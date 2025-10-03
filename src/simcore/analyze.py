# src/simcore/analyze.py
"""
Analiză post-procesare pentru simulări Brownian/Langevin.

Calculează pentru axele cerute:
  S_coord(t) = sum_i coord_i(t)^2 = N * ( std_coord(t)^2 + mean_coord(t)^2 )

Suplimentar:
  - S_tot(t) = sum_i ||r_i(t)||^2 = N * ( sum_{axes} std_ax(t)^2 + ||mean||^2 )
  - Fit liniar robust:
      • dacă NU dai --tmin/--tmax → folosește TOATE punctele;
      • opțional: --fit-last-frac 0.5 (ultima jumătate), --fit-last-n 100 (ultimele N puncte);
      • NU aruncă eroare dacă sunt prea puține puncte — doar sare peste fit.

Mod “fără timp”:
  - --use-index → ignoră coloana 't' și folosește indexul (0..T) ca axă orizontală;
  - opțional: --dt pentru a converti panta/step în panta/secondă (și D în [m^2/s]).
"""

from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

from .vis import plot_timeseries_vpython

VALID_COORDS = ("x", "y", "z")

# ------------------------- I/O helpers -------------------------

def _read_dat(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Nu am găsit fișierul: {os.path.abspath(path)}")
    # autodetectează delimitatorul; curăță header
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [c.strip().lstrip("\ufeff") for c in df.columns]
    return df

def _load_join(input_dir: str) -> pd.DataFrame:
    mp_path = os.path.join(input_dir, "mean_position.dat")
    md_path = os.path.join(input_dir, "mean_dispersion.dat")

    mp = _read_dat(mp_path)
    md = _read_dat(md_path)

    if "t" not in mp.columns or "t" not in md.columns:
        raise ValueError(
            "Lipsește coloana 't' într-unul dintre fișiere.\n"
            f"MP cols={list(mp.columns)}\nMD cols={list(md.columns)}"
        )

    df = pd.merge(mp, md, on="t", how="inner")
    if df.empty:
        raise ValueError(
            "Merge pe 't' a dat 0 rânduri. Probabil vectorii t NU coincid exact.\n"
            f"t_MP[0..3]={mp['t'].head(3).tolist()}  vs  t_MD[0..3]={md['t'].head(3).tolist()}"
        )
    return df

def _available_axes(df: pd.DataFrame) -> List[str]:
    axes = []
    for c in VALID_COORDS:
        if (f"{c}_mean" in df.columns) and (f"std_{c}" in df.columns):
            axes.append(c)
    return axes

# ------------------------- Core formulas -------------------------

def sum_coord2_from_df(df: pd.DataFrame, coord: str, N: int) -> pd.DataFrame:
    mean_col = f"{coord}_mean"
    std_col  = f"std_{coord}"
    if mean_col not in df.columns or std_col not in df.columns:
        raise ValueError(f"Lipsește {mean_col} sau {std_col} pentru coordonata '{coord}'.")
    mean2 = df[mean_col].to_numpy() ** 2
    var   = (df[std_col].to_numpy() ** 2)
    sum2  = N * (var + mean2)
    return pd.DataFrame({"t": df["t"].to_numpy(), "sum_coord2": sum2})

def sum_total2_from_df(df: pd.DataFrame, axes: List[str], N: int) -> pd.DataFrame:
    mean_sq = np.zeros(len(df), dtype=float)
    var_sum = np.zeros(len(df), dtype=float)
    for a in axes:
        mean_sq += (df[f"{a}_mean"].to_numpy() ** 2)
        var_sum += (df[f"std_{a}"].to_numpy() ** 2)
    sum_tot2 = N * (var_sum + mean_sq)
    return pd.DataFrame({"t": df["t"].to_numpy(), "sum_tot2": sum_tot2})

# ------------------------- Fitting -------------------------

def select_window(
    t: np.ndarray,
    y: np.ndarray,
    tmin: Optional[float],
    tmax: Optional[float],
    fit_last_frac: Optional[float],
    fit_last_n: Optional[int],
    use_index: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alege fereastra pentru fit:
      1) dacă tmin/tmax sunt date → folosește-le;
      2) altfel, dacă fit_last_frac sau fit_last_n sunt date → folosește coada;
      3) altfel → întregul domeniu.
    Dacă rezultă <2 puncte, întoarce (arrays goale) și vom sări peste fit.
    """
    if (tmin is not None) or (tmax is not None):
        mask = np.ones_like(t, dtype=bool)
        if tmin is not None: mask &= (t >= tmin)
        if tmax is not None: mask &= (t <= tmax)
        t_sel, y_sel = t[mask], y[mask]
    elif fit_last_n is not None and fit_last_n >= 2:
        t_sel, y_sel = t[-fit_last_n:], y[-fit_last_n:]
    elif fit_last_frac is not None and 0 < fit_last_frac <= 1.0:
        n = max(2, int(round(len(t) * fit_last_frac)))
        t_sel, y_sel = t[-n:], y[-n:]
    else:
        t_sel, y_sel = t, y

    # Dacă folosim index, t poate fi int; îl convertim la float pentru polyfit
    return np.asarray(t_sel, dtype=float), np.asarray(y_sel, dtype=float)

def try_linear_fit(t_sel: np.ndarray, y_sel: np.ndarray) -> Optional[Tuple[float, float]]:
    """Întoarce (m, b) sau None dacă sunt prea puține puncte."""
    if t_sel.size < 2:
        return None
    m, b = np.polyfit(t_sel, y_sel, deg=1)
    return m, b

# ------------------------- CLI -------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Analiză MSD: sum_i coord_i(t)^2 pe 1/2/3 axe din fișierele .dat")
    p.add_argument("--input", required=True, help="Folderul cu mean_position.dat și mean_dispersion.dat")
    p.add_argument("--N", type=int, required=True, help="Numărul de particule N (cel din simulare)")

    # Axele analizate
    p.add_argument("--axes", nargs="+", default=["auto"],
                   help="Axe: 'auto', 'all', sau listă (ex: x y, sau x y z)")

    # Mod timp/index
    p.add_argument("--use-index", action="store_true",
                   help="Ignoră coloana 't' și folosește indexul (0..T) ca timp (per step).")
    p.add_argument("--dt", type=float, default=None,
                   help="Dacă folosești --use-index, dă Δt [s] ca să convertim panta/step în panta/s și D în [m^2/s].")

    # Ferestre de fit (toate opționale)
    p.add_argument("--tmin", type=float, default=None, help="Timp minim pentru fit liniar (opțional)")
    p.add_argument("--tmax", type=float, default=None, help="Timp maxim pentru fit liniar (opțional)")
    p.add_argument("--fit-last-frac", type=float, default=None,
                   help="Folosește ultima fracțiune din date (ex: 0.5 pentru ultima jumătate) dacă nu dai tmin/tmax.")
    p.add_argument("--fit-last-n", type=int, default=None,
                   help="Folosește ultimele N puncte dacă nu dai tmin/tmax.")
    # Output agregat + plot
    p.add_argument("--aggregate-out", type=str, default=None,
                   help="Dacă e dat, scrie și un fișier agregat cu t/index, sum_x2[, sum_y2][, sum_z2], sum_tot2")
    p.add_argument("--plot", action="store_true", help="Plotează seriile în VPython (2D graph).")
    p.add_argument("--plot-total", action="store_true", help="Include și S_tot(t) în grafic dacă este disponibil.")
    p.add_argument("--fit-out", type=str, default=None,
               help="Dacă e setat, salvează pantele într-un DAT (axis, m, b, tmin, tmax, n, D_hat).")
    return p.parse_args()

def main():
    a = parse_args()
    df = _load_join(a.input)

    fit_rows = []  # ← colectăm pantele aici

    # Aflăm axele disponibile
    if a.axes == ["auto"]:
        axes = _available_axes(df)
    elif len(a.axes) == 1 and a.axes[0].lower() == "all":
        axes = [ax for ax in VALID_COORDS if f"{ax}_mean" in df.columns]
    else:
        axes = [ax.lower() for ax in a.axes]
    if not axes:
        raise ValueError("Nu am găsit nicio axă disponibilă în fișiere. Verifică .dat-urile.")

    # Construim S_ax(t) pentru fiecare axă
    collected: Dict[str, pd.DataFrame] = {}
    for ax in axes:
        out = sum_coord2_from_df(df, ax, a.N)
        out_path = os.path.join(a.input, f"sum_{ax}2.dat")
        out.to_csv(out_path, sep="\t", index=False, float_format="%.8g")
        print(f"[OK] scris: {out_path}")
        collected[ax] = out

        # Axa orizontală: timp sau index (0..T)
        t_full = out["t"].to_numpy()
        t_axis = np.arange(len(t_full), dtype=float) if a.use_index else t_full.astype(float)

        # Fereastra pentru fit
        t_sel, y_sel = select_window(
            t=t_axis,
            y=out["sum_coord2"].to_numpy(),
            tmin=a.tmin, tmax=a.tmax,
            fit_last_frac=a.fit_last_frac,
            fit_last_n=a.fit_last_n,
            use_index=a.use_index
        )
        fit = try_linear_fit(t_sel, y_sel)
        if fit is not None:
            m, b = fit
            if a.use_index and a.dt is not None:
                m_per_s = m / a.dt
                D_hat = m_per_s / (2.0 * a.N)
                print(f"[FIT {ax}] pe {'index' if a.use_index else 'timp'}: m/step={m:.6g}, b={b:.6g}  "
                      f"→ m/s={m_per_s:.6g},  D_hat(1D)={D_hat:.6g} m^2/s")
                fit_rows.append({
                    "axis": ax,
                    "m": m_per_s,  # m în [m^2/s]
                    "b": b,
                    "tmin_used": float(t_sel[0]), "tmax_used": float(t_sel[-1]), "n_points": int(len(t_sel)),
                    "D_hat": D_hat
                })
            elif a.use_index and a.dt is None:
                D_hat_per_step = m / (2.0 * a.N)
                print(f"[FIT {ax}] pe index: m/step={m:.6g}, b={b:.6g}  "
                      f"→ D_hat_per_step={D_hat_per_step:.6g} [m^2 per step]")
                fit_rows.append({
                    "axis": ax,
                    "m": m,  # m per-step
                    "b": b,
                    "tmin_used": float(t_sel[0]), "tmax_used": float(t_sel[-1]), "n_points": int(len(t_sel)),
                    "D_hat": D_hat_per_step
                })
            else:
                D_hat = m / (2.0 * a.N)
                print(f"[FIT {ax}] pe timp: m={m:.6g}  b={b:.6g}  → D_hat(1D)={D_hat:.6g} m^2/s")
                fit_rows.append({
                    "axis": ax,
                    "m": m,  # m în [m^2/s]
                    "b": b,
                    "tmin_used": float(t_sel[0]), "tmax_used": float(t_sel[-1]), "n_points": int(len(t_sel)),
                    "D_hat": D_hat
                })
        else:
            print(f"[FIT {ax}] insuficiente puncte pentru fit — skip.")

    # Total (d=nr axe) — îl calculăm dacă e cerut pentru agregat/plot/raport
    sum_tot_series = None
    need_total = bool(a.aggregate_out or a.plot_total or a.fit_out)
    if need_total:
        tot = sum_total2_from_df(df, axes, a.N)
        sum_tot_series = tot

        # Axa orizontală pentru total (t sau index)
        t_full = tot["t"].to_numpy()
        t_axis = np.arange(len(t_full), dtype=float) if a.use_index else t_full.astype(float)

        # Fit pe total (independent de aggregate_out)
        t_sel, y_sel = select_window(
            t=t_axis, y=tot["sum_tot2"].to_numpy(),
            tmin=a.tmin, tmax=a.tmax,
            fit_last_frac=a.fit_last_frac, fit_last_n=a.fit_last_n,
            use_index=a.use_index
        )
        fit = try_linear_fit(t_sel, y_sel)
        if fit is not None:
            m_tot, b_tot = fit
            d = len(axes)
            if a.use_index and a.dt is not None:
                m_per_s = m_tot / a.dt
                D_hat_tot = m_per_s / (2.0 * a.N * d)
                print(f"[FIT tot] m/step={m_tot:.6g}, b={b_tot:.6g} → m/s={m_per_s:.6g},  D_hat_tot={D_hat_tot:.6g} m^2/s (d={d})")
                fit_rows.append({
                    "axis": "total",
                    "m": m_per_s,  # [m^2/s]
                    "b": b_tot,
                    "tmin_used": float(t_sel[0]), "tmax_used": float(t_sel[-1]), "n_points": int(len(t_sel)),
                    "D_hat": D_hat_tot
                })
            elif a.use_index and a.dt is None:
                D_hat_tot_per_step = m_tot / (2.0 * a.N * d)
                print(f"[FIT tot] m/step={m_tot:.6g}, b={b_tot:.6g} → D_hat_tot/step={D_hat_tot_per_step:.6g} [m^2 per step] (d={d})")
                fit_rows.append({
                    "axis": "total",
                    "m": m_tot,  # per-step
                    "b": b_tot,
                    "tmin_used": float(t_sel[0]), "tmax_used": float(t_sel[-1]), "n_points": int(len(t_sel)),
                    "D_hat": D_hat_tot_per_step
                })
            else:
                D_hat_tot = m_tot / (2.0 * a.N * d)
                print(f"[FIT tot] m={m_tot:.6g}, b={b_tot:.6g} → D_hat_tot={D_hat_tot:.6g} m^2/s (d={d})")
                fit_rows.append({
                    "axis": "total",
                    "m": m_tot,  # [m^2/s]
                    "b": b_tot,
                    "tmin_used": float(t_sel[0]), "tmax_used": float(t_sel[-1]), "n_points": int(len(t_sel)),
                    "D_hat": D_hat_tot
                })
        else:
            print("[FIT tot] insuficiente puncte pentru fit — skip.")

        # Scriere agregat dacă s-a cerut
        if a.aggregate_out:
            agg = pd.DataFrame({"t" if not a.use_index else "index": t_axis})
            for ax in axes:
                agg[f"sum_{ax}2"] = collected[ax]["sum_coord2"].to_numpy()
            agg["sum_tot2"] = tot["sum_tot2"].to_numpy()
            agg.to_csv(a.aggregate_out, sep="\t", index=False, float_format="%.8g")
            print(f"[OK] scris agregat: {a.aggregate_out}")

    # Plot VPython (opțional)
    if a.plot:
        t_any = collected[axes[0]]["t"].to_numpy()
        t_axis = np.arange(len(t_any), dtype=float) if a.use_index else t_any.astype(float)
        series = {f"sum_{ax}2": collected[ax]["sum_coord2"].to_numpy() for ax in axes}
        if a.plot_total and sum_tot_series is not None:
            series["sum_tot2"] = sum_tot_series["sum_tot2"].to_numpy()
        title = f"MSD-like: sum coord^2 (N={a.N}, axes={','.join(axes)})"
        xlab  = "step index" if a.use_index else "t [s]"
        plot_timeseries_vpython(t=t_axis, series=series, title=title, xlabel=xlab,
                                ylabel="sum coord^2 [m^2]" if not a.use_index else "sum coord^2 [m^2] (per step)",
                                legend=True)

    # Scrie raportul de pante (dacă s-a cerut)
    if a.fit_out and fit_rows:
        out_df = pd.DataFrame(fit_rows)
        out_df.to_csv(a.fit_out, sep="\t", index=False, float_format="%.8g")
        print(f"[OK] pante salvate: {a.fit_out}")

if __name__ == "__main__":
    main()
