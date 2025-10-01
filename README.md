# Brownian / Langevin 3D Simulator (Many Particles, SOLID)
[...see in files for full details; math/physics comments are in Romanian within the code...]

Quick start:
1) python -m venv .venv && source .venv/bin/activate
2) pip install -e . [and optionally pip install vpython]
3) python -m simcore.cli --scheme langevin --dims 3 --N 10000 --steps 2000 --dt 1e-9 --m 4.19e-15 --gamma 1.88e-8 --T 300 --kB 1.380649e-23 --init-pos 0 0 0 --init-vel 0 0 0 --output ./outputLangevin --enable-vpython

   python -m simcore.cli --scheme brownian --dims 3 --N 10000 --steps 2000 --dt 1e-3 --gamma 1.88e-8 --T 300 --kB 1.380649e-23 --init-pos 0 0 0 --output ./outputBrownian --enable-vpython --viz-trail 

### Dependențe numerice

- **NumPy** (`import numpy as np`) — tipul `ndarray` pentru stocare numerică și operații vectorizate rapide:
  - starea sistemului: `r` (poziții), `v` (viteze) ca matrice `(N, dims)`;
  - generare zgomot gaussian: `np.random.normal(size=(N, dims))`;
  - statistici: medii/deviații standard vectorizate.

- **Pandas** — doar pentru **scriere** de fișiere `.dat` (TSV) convenabil (`DataFrame.to_csv`).
