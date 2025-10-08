# src/simcore/cli.py

# Despre NumPy (np):
#  - Tipul de bază este np.ndarray (tablouri numerice N×D).
#  - Operațiile sunt vectorizate și suportă broadcasting (eficient, fără bucle Python).
#  - Zgomotul gaussian se generează cu np.random.normal(size=(N, D)).
#  - Reproductibilitate: setează o singură dată seed-ul (np.random.seed(...)).
#  - Unități: toate mărimile din cod sunt în SI (m, s, kg), iar array-urile
#    păstrează aceste unități implicit prin valori; comentariile indică unitățile.

import argparse, numpy as np
from .config import SimConfig
from .simulator import Simulator

def parse_args():
    """
    Parsează argumentele din linia de comandă.
    Folosim argparse ca să expunem, într-un mod standard, toți parametrii
    (numerici + fizici) ai simulării, plus opțiunile de vizualizare.
    """
    p = argparse.ArgumentParser(description='3D Brownian/Langevin simulator')

    # --- OUTPUT / DISCRETIZARE ---
    p.add_argument('--output', type=str, default='output',
                   help="Directorul unde se vor scrie fișierele .dat (TSV).")
    p.add_argument('--N', type=int, default=10000,
                   help="Numărul de particule din ansamblu (cost ~ O(N)).")
    p.add_argument('--steps', type=int, default=2000,
                   help="Numărul de pași de timp (seria temporală va avea steps+1 puncte).")
    p.add_argument('--dt', type=float, default=1e-3,
                   help="Pasul de timp Δt [s]. În underdamped, alege Δt << m/γ pentru stabilitate.")
    p.add_argument('--dims', type=int, default=3, choices=[1,2,3],
                   help="Dimensionalitatea spațiului (1/2/3). Afectează doar forma vectorilor și factorul din MSD.")
    p.add_argument('--scheme', type=str, default='langevin', choices=['langevin','brownian','ballistic'],
                   help="Alege integratorul: 'langevin' (underdamped, cu viteză) sau 'brownian' (overdamped/random walk).")

    # wall single (pentru test simplu)
    p.add_argument('--enable-walls', action='store_true')
    p.add_argument('--wall-normal', type=float, nargs='+', default=None, help='Normal perete (1,2 sau 3 valori)')
    p.add_argument('--wall-c', type=float, default=0.0, help='Offset c în ecuația n·r=c')

    # preset de test 30°
    p.add_argument('--test-specular', action='store_true',
                help='Preset: câteva particule, viteză la 30° spre perete, fără zgomot.')
    
    # --- PARAMETRI FIZICI (SI) ---
    # Notă: m (kg), gamma (kg/s), T (K), kB (J/K)
    p.add_argument('--m', type=float, default=4.19e-15,
                   help="Masa particulei [kg] (utilă DOAR în schema 'langevin'). Presetat pentru o particulă de 1 μm rază (a = 1e-6 m)")
    p.add_argument('--gamma', type=float, default=1.88e-8,
                   help="Coeficient de frecare γ [kg/s]. Intră în -(γ/m)v (Langevin) și D=kB*T/γ (Brownian).")
    p.add_argument('--T', type=float, default=300.0,
                   help="Temperatura [K]. Zgomotul termic crește cu T.")
    p.add_argument('--kB', type=float, default=1.380649e-23,
                   help="Constanta lui Boltzmann [J/K] (valoare standard SI).")

    # --- CONDIȚII INIȚIALE ---
    # Pentru cerința “a)”: toate particulele pornesc din același punct cu viteză 0.
    # Acceptăm totuși valori custom (liste de lungime = dims).
    p.add_argument('--init-pos', type=float, nargs='+', default=None,
                   help="Poziția inițială (len=dims). Implicit: vector nul.")
    p.add_argument('--init-vel', type=float, nargs='+', default=None,
                   help="Viteza inițială (len=dims). Ignorată în 'brownian'. Implicit: vector nul.")
    p.add_argument('--seed', type=int, default=None,
                   help="Sămânța RNG pentru reproducibilitate (np.random.seed(seed)).")

    # --- VIZUALIZARE (doar grafic; nu afectează fizica) ---
    p.add_argument('--enable-vpython', action='store_true',
                   help="Dacă este setat, pornește animația VPython după simulare.")
    p.add_argument('--viz-n', type=int, default=100,
                   help="Număr maxim de particule afișate în animație (pentru performanță).")
    p.add_argument('--viz-scale', type=float, default=0.0, help='Factor de scalare pe ecran (0=auto-scale pe RMS).')
    p.add_argument('--viz-trail', action='store_true', help='Dacă este setat, desenează trasee (trails) pentru particulele vizualizate.')

    return p.parse_args()

def main():
    """
    Punctul de intrare când rulăm modulul ca script:
      python -m simcore.cli [ARGUMENTE]
    1) Parsează argumentele.
    2) Construiește obiectul de configurare (SimConfig).
    3) Rulează simularea cu Simulator(cfg).
    4) Afișează căile fișierelor scrise.
    """
    a = parse_args()

    # Validare și pregătire condiții inițiale (shape = (dims,))
    dims = a.dims
    init_pos = np.zeros(dims) if a.init_pos is None else np.asarray(a.init_pos, float)
    init_vel = np.zeros(dims) if a.init_vel is None else np.asarray(a.init_vel, float)

    # walls din args (opțional)
    walls = tuple()
    if a.enable_walls and a.wall_normal is not None:
        n = np.asarray(a.wall_normal, float)
        if n.size not in (1,2,3):
            raise ValueError('--wall-normal trebuie să aibă 1, 2 sau 3 componente.')
        if n.size != dims:
            n2 = np.zeros(dims, float); n2[:n.size] = n; n = n2
        # (nx, ny, nz, c)
        nx = float(n[0]); ny = float(n[1] if dims>1 else 0.0); nz = float(n[2] if dims>2 else 0.0)
        walls = ((nx, ny, nz, float(a.wall_c)),)

    # preset test specular 30° (fără zgomot)
    test_specular = False
    if a.test_specular:
        test_specular = True
        dims = max(dims, 2)
        a.scheme = 'ballistic'
        a.N = 3
        a.steps = 300
        a.dt = 1e-3
        # poziție inițială: în fața peretelui x=0
        init_pos = np.array([0.1, 0.0] + ([0.0] if dims==3 else []), float)
        # v la 30° față de plan, spre perete (-x)
        theta = np.deg2rad(30.0)
        vmag = 1.0
        vx = -vmag * np.cos(theta)
        vy =  vmag * np.sin(theta)
        init_vel = np.array([vx, vy] + ([0.0] if dims==3 else []), float)
        # perete x=0, normal (1,0[,0])
        walls = ((1.0, 0.0, 0.0, 0.0),)    

    # Verificăm că utilizatorul a dat exact dims valori pentru poziție/viteză
    if init_pos.shape != (dims,):
        raise ValueError(f'--init-pos must have length {dims}')
    if init_vel.shape != (dims,):
        raise ValueError(f'--init-vel must have length {dims}')

    # Construim obiectul de configurare. Toată logica din cod citește din SimConfig,
    # ceea ce face ușoară reproducerea experimentelor și extinderea ulterioară.
    cfg = SimConfig(
        N=a.N, steps=a.steps, dt=a.dt, dims=dims,
        m=a.m, gamma=a.gamma, T=a.T, kB=a.kB,
        init_pos=init_pos, init_vel=init_vel,
        scheme=a.scheme, 
        enable_walls=(a.enable_walls or test_specular),
        walls=walls, test_specular=test_specular,
        seed=a.seed,
        output_dir=a.output,
        enable_vpython=a.enable_vpython, viz_n=a.viz_n,
        viz_scale=a.viz_scale, viz_trail=a.viz_trail
    )

    # Rulăm simularea: orchestratorul (Simulator) apelează integratorul ales (Strategy)
    # și scrie fișierele .dat (mean_position.dat & mean_dispersion.dat).
    mp, md = Simulator(cfg).run()

    # Feedback minimal în consolă (utile pentru notebook/loguri CI)
    print('Wrote:', mp)
    print('Wrote:', md)

if __name__ == '__main__':
    # Permite rularea modulului direct:
    #   python -m simcore.cli --scheme brownian --N 5000 --steps 2000 --dt 1e-3 --dims 3 ...
    main()
