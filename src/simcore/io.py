# src/simcore/io.py

# Despre NumPy (np):
#  - Tipul de bază este np.ndarray (tablouri numerice N×D).
#  - Operațiile sunt vectorizate și suportă broadcasting (eficient, fără bucle Python).
#  - Zgomotul gaussian se generează cu np.random.normal(size=(N, D)).
#  - Reproductibilitate: setează o singură dată seed-ul (np.random.seed(...)).
#  - Unități: toate mărimile din cod sunt în SI (m, s, kg), iar array-urile
#    păstrează aceste unități implicit prin valori; comentariile indică unitățile.

import os, numpy as np, pandas as pd

class DataWriter:
    """
    Clasa responsabilă DOAR de scrierea pe disc a seriilor temporale
    (SRP: Single Responsibility). Produce fișiere .dat (TSV) ușor de
    importat în gnuplot/matplotlib/Origin/Excel.

    Convenții:
      - separare cu TAB ('\t')  -> robust la spații în cap de coloană
      - fără index pandas       -> doar coloanele utile
      - float_format='%.8g'     -> 8 cifre semnificative (bun compromis
                                   între precizie și mărimea fișierului)

    Unități (SI):
      - t [s]
      - x_mean, y_mean, z_mean [m]
      - std_x, std_y, std_z [m]
      - radial_rms [m]
    """

    def __init__(self, output_dir: str):
        # Creează directorul de output dacă nu există (idempotent).
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def write_mean_positions(self, t, means):
        """
        Scrie fișierul cu media pozițiilor pe fiecare axă.

        Parametri
        ---------
        t : array-like, shape (T+1,)
            Vectorul timpilor discreți (t = k * dt).
        means : array-like, shape (T+1, D)
            Media pozițiilor pe fiecare axă, la fiecare pas.
            D = dims (1, 2 sau 3). Coloanele sunt ordonate x,y,z.

        Output
        ------
        <output_dir>/mean_position.dat (TSV)
          Coloane: t, x_mean[, y_mean][, z_mean]
        """
        # Construim header-ul în funcție de numărul de axe (D).
        cols = ['t'] + [f'{ax}_mean' for ax in 'xyz'[:means.shape[1]]]

        # Cale completă către fișierul de ieșire.
        path = os.path.join(self.output_dir, 'mean_position.dat')

        # Îmbinăm t și means într-o singură matrice (coloane alăturate).
        df = pd.DataFrame(np.column_stack([t, means]), columns=cols)

        # Scriere TSV, fără index; 8 cifre semnificative pentru numeric.
        df.to_csv(path, sep='\t', index=False, float_format='%.8g')
        return path

    def write_dispersion(self, t, stds_series, radial_rms):
        """
        Scrie fișierul cu împrăștierea ansamblului.

        Parametri
        ---------
        t : array-like, shape (T+1,)
            Vectorul timpilor discreți (t = k * dt).
        stds_series : array-like, shape (T+1, D)
            Deviațiile standard pe fiecare axă (definiție populație, ddof=0),
            la fiecare pas de timp. D = dims (1, 2 sau 3).
        radial_rms : array-like, shape (T+1,)
            RMS-ul radial față de medie:
                radial_rms[k] = mean_i || r_i(k) - r_mean(k) ||_2
            (o singură coloană scalară, măsoară lărgirea globală a norului).

        Output
        ------
        <output_dir>/mean_dispersion.dat (TSV)
          Coloane: t, std_x[, std_y][, std_z], radial_rms
        """
        # Header adaptiv, în funcție de D.
        cols = ['t'] + [f'std_{ax}' for ax in 'xyz'[:stds_series.shape[1]]] + ['radial_rms']

        path = os.path.join(self.output_dir, 'mean_dispersion.dat')

        # Concatenăm pe coloane: t | std_x[, std_y][, std_z] | radial_rms.
        df = pd.DataFrame(np.column_stack([t, stds_series, radial_rms]), columns=cols)

        df.to_csv(path, sep='\t', index=False, float_format='%.8g')
        return path
