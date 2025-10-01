# src/simcore/models.py

# Despre NumPy (np):
#  - Tipul de bază este np.ndarray (tablouri numerice N×D).
#  - Operațiile sunt vectorizate și suportă broadcasting (eficient, fără bucle Python).
#  - Zgomotul gaussian se generează cu np.random.normal(size=(N, D)).
#  - Reproductibilitate: setează o singură dată seed-ul (np.random.seed(...)).
#  - Unități: toate mărimile din cod sunt în SI (m, s, kg), iar array-urile
#    păstrează aceste unități implicit prin valori; comentariile indică unitățile.

import numpy as np

class Ensemble:
    """
    Ansamblu de N particule în D dimensiuni.
    Ține starea colectivă:
      - r : poziții (shape = (N, D))   [m]
      - v : viteze  (shape = (N, D))   [m/s]
    În schema overdamped (Brownian), v nu este folosită în dinamica efectivă,
    dar o păstrăm pentru compatibilitate/extensibilitate (ex. comutare rapidă
    pe schema underdamped, adăugare de forțe etc.).

    SRP (Single Responsibility): clasa are o singură responsabilitate — stocarea
    și actualizarea stărilor particulelor; nu face integrare, I/O sau metrici.
    """

    def __init__(self, N: int, dims: int, init_pos: np.ndarray, init_vel: np.ndarray):
        """
        Parametri
        ---------
        N : int
            Număr de particule.
        dims : int
            Dimensionalitatea spațiului (1/2/3).
        init_pos : np.ndarray, shape (dims,)
            Poziția inițială comună tuturor particulelor [m].
        init_vel : np.ndarray, shape (dims,)
            Viteza inițială comună tuturor particulelor [m/s].

        Notă:
        - Conform cerinței “a)”: toate particulele pornesc din același punct,
          cu viteză inițială 0 (poți furniza vectori nuli).
        - Replicăm vectorii (dims,) la matrice (N, D) cu np.repeat.
        """
        # Validăm formele vectorilor inițiali (defensiv).
        assert init_pos.shape == (dims,), "init_pos must have shape (dims,)"
        assert init_vel.shape == (dims,), "init_vel must have shape (dims,)"

        self.N, self.dims = N, dims

        # r și v sunt stocate ca (N, D). Broadcasting-ul NumPy ne ajută să
        # replicăm rapid vectorii inițiali pe N rânduri.
        self.r = np.repeat(init_pos.reshape(1, dims), N, axis=0)  # [m]
        self.v = np.repeat(init_vel.reshape(1, dims), N, axis=0)  # [m/s]

    def positions(self):
        """Returnează matricea pozițiilor curente r (shape (N, D), unități [m])."""
        return self.r

    def velocities(self):
        """Returnează matricea vitezelor curente v (shape (N, D), unități [m/s])."""
        return self.v

    def set_state(self, r, v=None):
        """
        Actualizează starea ansamblului.
        Parametri
        ---------
        r : np.ndarray, shape (N, D), [m]
            Pozițiile noi ale particulelor.
        v : np.ndarray, shape (N, D), [m/s] sau None
            Vitezele noi (dacă schema folosită are viteză explicită).
            În overdamped poți pasa None (nu schimbăm v).

        Observații:
        - Nu facem aici validări costisitoare (ex. assert pe shape) ca să rămână
          rapid în bucla de timp; integratorul este responsabil să livreze forme corecte.
        """
        self.r = r
        if v is not None:
            self.v = v
