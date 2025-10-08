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
    Expunem parametri numerici/fizici ai simulării + opțiuni de inițializare și pereți.
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
                   help="Dimensionalitatea spațiului (1/2/3). Afectează forma vectorilor și factorul din MSD.")
    p.add_argument('--scheme', type=str, default='langevin',
                   choices=['langevin','brownian','ballistic'],
                   help="Integrator: 'langevin' (underdamped), 'brownian' (overdamped) sau 'ballistic' (drept).")

    # --- PEREȚI (plan n·r=c) / CUTIE ---
    p.add_argument('--enable-walls', action='store_true',
                   help="Activează coliziunile cu pereții definiți (planuri sau cutie).")
    p.add_argument('--wall-normal', type=float, nargs='+', default=None,
                   help='Vector normal al unui perete (1,2 sau 3 componente).')
    p.add_argument('--wall-c', type=float, default=0.0,
                   help='Offset c în ecuația planului n·r=c [m].')
    p.add_argument('--box', type=float, nargs='+', default=None,
                   help='Cutie axis-aligned: 3D: xmin xmax ymin ymax zmin zmax ; 2D: xmin xmax ymin ymax.')
    p.add_argument('--box-csv', type=str, default=None,
               help='Alternativă robustă: "xmin,xmax,ymin,ymax[,zmin,zmax]" (valorile despărțite prin virgulă).')
 
    # --- PRESET DE TEST SPECULAR (fără zgomot) ---
    p.add_argument('--test-specular', action='store_true',
                   help='Preset: particule la 30° spre un perete plan; pornesc pe rând (stagger), fără zgomot.')

    # --- PARAMETRI FIZICI (SI) ---
    p.add_argument('--m', type=float, default=4.19e-15,
                   help="Masa [kg] (folosită în 'langevin').")
    p.add_argument('--gamma', type=float, default=1.88e-8,
                   help="Frecare γ [kg/s]. Intră în -(γ/m)v (Langevin) și D=kB*T/γ (Brownian).")
    p.add_argument('--T', type=float, default=300.0,
                   help="Temperatura [K]. Zgomotul termic crește cu T.")
    p.add_argument('--kB', type=float, default=1.380649e-23,
                   help="Constanta lui Boltzmann [J/K].")

    # --- CONDIȚII INIȚIALE (mod + parametri) ---
    p.add_argument('--init-dist', type=str, default='point',
                   choices=['point', 'jitter', 'uniform'],
                   help="Mod inițializare poziții: "
                        "'point' = toate la --init-pos; "
                        "'jitter' = gaussian în jurul --init-pos (σ=--init-jitter); "
                        "'uniform' = uniform în --box (obligatoriu).")
    p.add_argument('--init-jitter', type=float, default=0.0,
                   help='Deviație standard [m] pentru jitter gaussian (folosită doar la init-dist=jitter).')

    p.add_argument('--init-pos', type=float, nargs='+', default=None,
                   help="Poziția inițială de referință (len=dims). Implicit: vector nul.")
    p.add_argument('--init-vel', type=float, nargs='+', default=None,
                   help="Viteza inițială (len=dims). Ignorată în 'brownian'. Implicit: vector nul.")
    p.add_argument('--seed', type=int, default=None,
                   help="Sămânța RNG pentru reproducibilitate (np.random.seed(seed)).")

    # --- VIZUALIZARE (doar grafic; nu afectează fizica) ---
    p.add_argument('--enable-vpython', action='store_true',
                   help="Dacă este setat, pornește animația VPython după simulare.")
    p.add_argument('--viz-n', type=int, default=100,
                   help="Număr maxim de particule afișate (pentru performanță).")
    p.add_argument('--viz-scale', type=float, default=0.0,
                   help='Factor de scalare pe ecran (0=auto-scale pe RMS).')
    p.add_argument('--viz-trail', action='store_true',
                   help='Dacă este setat, desenează trasee (trails).')

    return p.parse_args()

def main():
    """
    Flux:
      1) Parsează argumentele.
      2) Validează/compune pereții (planuri + cutie).
      3) Construiește SimConfig (centralizează setările).
      4) Rulează simularea (scrie .dat și, opțional, animă).
    """
    a = parse_args()

    # Dimensiune spațiu
    dims = a.dims

    # Condiții inițiale (shape = (dims,))
    init_pos = np.zeros(dims) if a.init_pos is None else np.asarray(a.init_pos, float)
    init_vel = np.zeros(dims) if a.init_vel is None else np.asarray(a.init_vel, float)

    # ------------------------
    # PEREȚI DIN ARGUMENTE
    # ------------------------
    walls = tuple()

    # Perete singular n·r=c (opțional)
    if a.enable_walls and a.wall_normal is not None:
        n = np.asarray(a.wall_normal, float)
        if n.size not in (1, 2, 3):
            raise ValueError('--wall-normal trebuie să aibă 1, 2 sau 3 componente.')
        if n.size != dims:
            # Adaptăm normalul la numărul de dimensiuni cerut
            n2 = np.zeros(dims, float); n2[:n.size] = n; n = n2
        nx = float(n[0])
        ny = float(n[1] if dims > 1 else 0.0)
        nz = float(n[2] if dims > 2 else 0.0)
        walls = walls + ((nx, ny, nz, float(a.wall_c)),)

    # Cutie axis-aligned (opțional) – adaugă 4 (2D) sau 6 (3D) pereți
    box = None 
    # 1) Încearcă --box-csv dacă e prezent (robust la copy/paste)
    if a.box_csv is not None:
        try:
            vals = [float(s) for s in a.box_csv.replace(';', ',').split(',')]
        except ValueError:
            raise ValueError('--box-csv are valori ne-numerice. Format: xmin,xmax,ymin,ymax[,zmin,zmax]')
        if dims == 3:
            if len(vals) != 6:
                raise ValueError('--box-csv în 3D cere 6 numere: xmin,xmax,ymin,ymax,zmin,zmax')
            xmin, xmax, ymin, ymax, zmin, zmax = vals
            walls = walls + (
                (+1.0, 0.0, 0.0, xmax),
                (-1.0, 0.0, 0.0, -xmin),
                (0.0, +1.0, 0.0, ymax),
                (0.0, -1.0, 0.0, -ymin),
                (0.0, 0.0, +1.0, zmax),
                (0.0, 0.0, -1.0, -zmin),
            )
            box = (xmin, xmax, ymin, ymax, zmin, zmax)
        else:
            if len(vals) != 4:
                raise ValueError('--box-csv în 2D cere 4 numere: xmin,xmax,ymin,ymax')
            xmin, xmax, ymin, ymax = vals
            walls = walls + (
                (+1.0, 0.0, 0.0, xmax),
                (-1.0, 0.0, 0.0, -xmin),
                (0.0, +1.0, 0.0, ymax),
                (0.0, -1.0, 0.0, -ymin),
            )
            box = (xmin, xmax, ymin, ymax)

    # 2) Dacă nu e --box-csv, folosește --box 
    elif a.box is not None:
        vals = list(map(float, a.box))
        if dims == 3:
            if len(vals) != 6:
                raise ValueError('--box în 3D cere 6 numere: xmin xmax ymin ymax zmin zmax')
            xmin, xmax, ymin, ymax, zmin, zmax = vals
            walls = walls + (
                (+1.0, 0.0, 0.0, xmax),
                (-1.0, 0.0, 0.0, -xmin),
                (0.0, +1.0, 0.0, ymax),
                (0.0, -1.0, 0.0, -ymin),
                (0.0, 0.0, +1.0, zmax),
                (0.0, 0.0, -1.0, -zmin),
            )
            box = (xmin, xmax, ymin, ymax, zmin, zmax)
        else:
            if len(vals) != 4:
                raise ValueError('--box în 2D cere 4 numere: xmin xmax ymin ymax')
            xmin, xmax, ymin, ymax = vals
            walls = walls + (
                (+1.0, 0.0, 0.0, xmax),
                (-1.0, 0.0, 0.0, -xmin),
                (0.0, +1.0, 0.0, ymax),
                (0.0, -1.0, 0.0, -ymin),
            )
            box = (xmin, xmax, ymin, ymax)

    # Validări pentru modul de inițializare
    if a.init_dist == 'jitter' and a.init_jitter <= 0.0:
        raise ValueError("Pentru --init-dist=jitter trebuie --init-jitter > 0.")
    if a.init_dist == 'uniform' and box is None:
        raise ValueError("Pentru --init-dist=uniform trebuie definită o cutie cu --box.")

    # PRESET TEST SPECULAR (fără zgomot; lansare pe rând → nu se suprapun vizual)
    test_specular = False
    if a.test_specular:
        test_specular = True
        dims = max(dims, 2)
        a.scheme = 'ballistic'
        a.N = 3
        a.steps = 300
        a.dt = 1e-3

        # Poziție inițială comună (păstrăm cerința)
        init_pos = np.array([0.1, 0.0] + ([0.0] if dims == 3 else []), float)

        # Viteză la 30° față de plan, spre perete (-x)
        theta = np.deg2rad(30.0); vmag = 1.0
        vx = -vmag * np.cos(theta); vy = vmag * np.sin(theta)
        init_vel = np.array([vx, vy] + ([0.0] if dims == 3 else []), float)

        # Perete x=0
        walls = walls + ((1.0, 0.0, 0.0, 0.0),)

    # Validări forme pentru init_pos/init_vel
    if init_pos.shape != (dims,):
        raise ValueError(f'--init-pos must have length {dims}')
    if init_vel.shape != (dims,):
        raise ValueError(f'--init-vel must have length {dims}')

    # ------------------------
    # CONSTRUIRE CONFIGURAȚIE
    # ------------------------
    cfg = SimConfig(
        # nucleu
        N=a.N, steps=a.steps, dt=a.dt, dims=dims,
        # fizică
        m=a.m, gamma=a.gamma, T=a.T, kB=a.kB,
        # ICs
        init_pos=init_pos, init_vel=init_vel,
        init_dist=a.init_dist, init_jitter=a.init_jitter, box=box,
        # integrator
        scheme=a.scheme,
        # pereți
        enable_walls=(a.enable_walls or test_specular or (box is not None)),
        walls=walls, test_specular=test_specular,
        # RNG / output
        seed=a.seed, output_dir=a.output,
        # viz
        enable_vpython=a.enable_vpython, viz_n=a.viz_n,
        viz_scale=a.viz_scale, viz_trail=a.viz_trail,
    )

    # ------------------------
    # RULARE + FEEDBACK
    # ------------------------
    mp, md = Simulator(cfg).run()
    print('Wrote:', mp)
    print('Wrote:', md)

if __name__ == '__main__':
    main()
