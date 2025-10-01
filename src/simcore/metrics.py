# src/simcore/metrics.py

# Despre NumPy (np):
#  - Tipul de bază este np.ndarray (tablouri numerice N×D).
#  - Operațiile sunt vectorizate și suportă broadcasting (eficient, fără bucle Python).
#  - Zgomotul gaussian se generează cu np.random.normal(size=(N, D)).
#  - Reproductibilitate: setează o singură dată seed-ul (np.random.seed(...)).
#  - Unități: toate mărimile din cod sunt în SI (m, s, kg), iar array-urile
#    păstrează aceste unități implicit prin valori; comentariile indică unitățile.

# NumPy: toate calculele sunt vectorizate pe axe:
#  - mean pe axis=0 => media peste particule (rezultat shape (D,));
#  - std cu ddof=0 => deviație standard de populație (nu eșantion);
#  - radial RMS => normă euclidiană pe ultimul axis, apoi mean pe particule.

import numpy as np

def mean_position(r: np.ndarray) -> np.ndarray:
    """
    Media pozițiilor pe fiecare axă.
    Parametri
    ---------
    r : np.ndarray, shape (N, D)
        Pozițiile tuturor particulelor la un anumit pas de timp.
        N = număr particule, D = dims (1/2/3). Unități: metri [m].

    Returnează
    ----------
    np.ndarray, shape (D,)
        Vectorul mediu pe axe: (x_mean[, y_mean][, z_mean]) în [m].

    Observații
    ----------
    - Calcul vectorizat: media pe axa 0 (peste particule).
    - Complexitate O(N*D).
    """
    return r.mean(axis=0)


def dispersion(r: np.ndarray, mean: np.ndarray):
    """
    Împrăștierea (dispersia) ansamblului față de media pozițiilor.

    Parametri
    ---------
    r : np.ndarray, shape (N, D)
        Pozițiile particulelor la pasul curent [m].
    mean : np.ndarray, shape (D,)
        Poziția medie a ansamblului [m] (tipic, rezultatul din mean_position).

    Returnează
    ----------
    dict cu:
      - 'stds' : np.ndarray, shape (D,)
          Deviația standard pe fiecare axă (definiție populație, ddof=0) [m].
      - 'radial_rms' : float
          RMS radial față de medie (o scală globală a lărgirii norului) [m].
          Definiție: radial_rms = mean_i || r_i - mean ||_2

    Detalii/justificări
    -------------------
    • ddof=0:
        Folosim definiția *populației* pentru deviația standard, deoarece
        considerăm întreg ansamblul de particule la acel pas (nu un eșantion).
        Formula corespunde σ = sqrt( (1/N) * Σ_i (x_i - μ)^2 ).

    • RMS radial:
        Este media normelor Euclidiene ale vectorilor de abatere față de medie:
            radial_rms = (1/N) * Σ_i sqrt( (r_i - mean) · (r_i - mean) )
        Are unități de lungime [m]; oferă o singură măsură scalară a “razei”
        tipice a norului, independentă de axele de coordonate.

    • Unități:
        std_x, std_y, std_z și radial_rms sunt în metri [m].

    • Stabilitate numerică:
        Dacă r are valori foarte mari/foarte mici, centrare pe mean ajută
        la evitarea erorilor de anulare. Implementarea de mai jos deja
        calculează pe (r - mean).

    • Complexitate:
        O(N*D), complet vectorizat.
    """
    # Deviația standard pe axă (populație: ddof=0)
    stds = r.std(axis=0, ddof=0)

    # RMS radial: media normelor Euclidiene ale abaterilor față de mean
    radial = np.sqrt(((r - mean) ** 2).sum(axis=1)).mean()

    return {'stds': stds, 'radial_rms': radial}
