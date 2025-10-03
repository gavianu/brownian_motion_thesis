# Brownian / Langevin 3D Simulator (Many Particles, SOLID)

Simularea mișcării browniene în 1D/2D/3D cu două scheme:

- **Langevin (underdamped)** – cu viteză, frecare și zgomot alb (Euler–Maruyama).
- **Brownian (overdamped)** – random walk gaussian, fără inerție explicită.

Arhitectură **SOLID** / separare de responsabilități:  
`physics` (integratoare) · `models` (stare) · `simulator` (orchestrare) · `metrics` (statistici) · `io` (I/O .dat) · `vis` (VPython 3D/2D) · `analyze` (post-procesare).

> Comentariile de **fizică și matematică sunt în limba română** direct în cod.

---

## 1) Instalare & mediu

> Recomandat: Python 3.10+

### macOS / Linux (bash)

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e .
pip install vpython           # pentru vizualizare 3D/2D
pip install threadpoolctl     # opțional: introspecție thread-uri BLAS
```

### Windows (PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

pip install -e .
pip install vpython
pip install threadpoolctl     # opțional
```

### (Opțional) Multi-thread BLAS via `.env`

Creează un fișier **`.env`** în rădăcina proiectului:

```
OMP_NUM_THREADS=8
OPENBLAS_NUM_THREADS=8
MKL_NUM_THREADS=8
NUMEXPR_NUM_THREADS=8
KMP_BLOCKTIME=0
```

În VS Code, setare recomandată: **Settings → Python: Env File** = `${workspaceFolder}/.env`, apoi **repornește terminalul/VS Code**.

---

## 2) Structura proiectului

```
project-root/
  src/
    simcore/
      cli.py         # rulează simularea din linia de comandă
      config.py      # parametri (SI) și docstring-uri detaliate
      models.py      # starea ansamblului: r (m), v (m/s)
      physics.py     # LangevinEM (underdamped), BrownianOverdamped (overdamped)
      metrics.py     # mean_position, dispersion (std pe axe + “radial_rms”)
      io.py          # scriere .dat (TSV)
      vis.py         # VPython: animație 3D + plot 2D time series
      analyze.py     # post-procesare (MSD-like, fit pante, plot VPython)
```

---

## 3) Rulare simulare (CLI)

> Unități **SI**: `t [s]`, `r [m]`, `v [m/s]`, `m [kg]`, `γ [kg/s]`, `T [K]`, `kB [J/K]`.  
> Cerința a): toate particulele pornesc din același punct, viteză inițială 0.

### Langevin (underdamped, 3D)

**macOS/Linux:**

```bash
python -m simcore.cli --scheme langevin --dims 3 --N 10000 --steps 2000 --dt 1e-9   --m 4.19e-15 --gamma 1.88e-8 --T 300 --kB 1.380649e-23   --init-pos 0 0 0 --init-vel 0 0 0   --output ./outputLangevin --enable-vpython
```

**Windows (PowerShell):**

```powershell
python -m simcore.cli --scheme langevin --dims 3 --N 10000 --steps 2000 --dt 1e-9 `
  --m 4.19e-15 --gamma 1.88e-8 --T 300 --kB 1.380649e-23 `
  --init-pos 0 0 0 --init-vel 0 0 0 `
  --output ./outputLangevin --enable-vpython
```

### Brownian (overdamped, 3D)

**macOS/Linux:**

```bash
python -m simcore.cli --scheme brownian --dims 3 --N 10000 --steps 2000 --dt 1e-3   --gamma 1.88e-8 --T 300 --kB 1.380649e-23   --init-pos 0 0 0   --output ./outputBrownian --enable-vpython --viz-trail
```

**Windows:**

```powershell
python -m simcore.cli --scheme brownian --dims 3 --N 10000 --steps 2000 --dt 1e-3 `
  --gamma 1.88e-8 --T 300 --kB 1.380649e-23 `
  --init-pos 0 0 0 `
  --output ./outputBrownian --enable-vpython --viz-trail
```

### Fișiere generate (TSV)

- `mean_position.dat` → `t, x_mean[, y_mean][, z_mean]`
- `mean_dispersion.dat` → `t, std_x[, std_y][, std_z], radial_rms`

> `radial_rms` = media lui `||r_i - r_mean||` (nu RMS clasic).  
> Pentru MSD folosește `std_*^2` și/sau analizorul de mai jos.

---

## 4) Analiză post-procesare (MSD-like) + plot 2D cu VPython

**Definiție:** pentru coordonata `α ∈ {x,y,z}`,

$$
S*\alpha(t) = \sum_i \alpha_i(t)^2
= N\left(\mathrm{std}*\alpha(t)^2 + \mathrm{mean}\_\alpha(t)^2\right)
$$

- **Overdamped / Langevin la timp lung:** $$ S\_\alpha(t) \approx 2ND\,t $$ ⇒ $$ panta ≈ 2ND $$.
- **Langevin la timp scurt:** $$ S\_\alpha(t) \sim N\,(k_BT/m)\,t^2 $$ (balistic), apoi liniar.

Pentru fiecare axă $$ \alpha \in \{x,y,z\} $$:

$$
S*\alpha(t) = \sum_i \alpha_i(t)^2 = N\big(\mathrm{std}*\alpha(t)^2 + \mathrm{mean}_\alpha(t)^2\big).
$$

În regim difuziv (timp lung),

$$
S_\alpha(t) \approx 2ND\,t \;\Rightarrow\; m*\alpha \approx 2ND,\ \ D \approx \frac{m*\alpha}{2N}.
$$

Pentru totalul pe $$d$$ dimensiuni (ex. $$d=3$$):

$$
S*{\text{tot}}(t)=S_x+S_y+S_z \approx 2NdD\,t \;\Rightarrow\; D \approx \frac{m*{\text{tot}}}{2Nd}.
$$

> Notă: la început (Langevin underdamped) apare regim **balistic** ($$\propto t^2$$). Fă fit-ul pe „coada” datelor (timp lung).

### Exemple

**Detectează axele disponibile și plotează (VPython 2D):**

```bash
python -m simcore.analyze --input ./outputLangevin --N 10000 --axes auto --plot --plot-total
```

**Specificați axele și fit pe o fereastră:**

```bash
python -m simcore.analyze --input ./outputBrownian --N 10000   --axes x y z --tmin 0.1 --tmax 1.0 --plot --plot-total
```

**Fără timp (pe index 0..T), cu conversie prin Δt:**

```bash
python -m simcore.analyze --input ./outputLangevin --N 10000   --axes x y z --use-index --dt 1e-6 --plot --plot-total
```

**Agregat într-un singur fișier (toate seriile + total):**

```bash
python -m simcore.analyze --input ./outputBrownian --N 10000   --axes all --aggregate-out ./outputBrownian/sum_all.dat
```

**Fit pe ultima jumătate din puncte + salvare pante (fit_report.dat):**

```bash
python -m simcore.analyze --input ./outputBrownian --N 10000 --axes x y z --fit-last-frac 0.5 --plot --plot-total --fit-out ./outputBrownian/fit_report.dat
```

**Fit pe ultimele N puncte (ex. 2000) + salvare pante:**

```bash
python -m simcore.analyze --input ./outputBrownian --N 10000 --axes x y z --fit-last-n 2000 --plot --plot-total --fit-out ./outputBrownian/fit_report.dat
```

### Fișiere generate de analiză

- `sum_x2.dat` (și/sau `sum_y2.dat`, `sum_z2.dat`) → `t, sum_coord2`
- (opțional) `sum_all.dat` → `t/index, sum_x2[, sum_y2][, sum_z2], sum_tot2`
- (opțional) `fit_report.dat` → `axis, m, b, tmin_used, tmax_used, n_points, D_hat`

---

## 5) Vizualizare (VPython)

- **3D animație nor** (în simulare): `--enable-vpython`, cu opțiuni  
  `--viz-n`, `--viz-scale` (0 = auto), `--viz-trail`.
- **2D plot** (în analiză): `--plot` deschide un grafic VPython (`graph` + `gcurve`).

> Vizualizarea **nu schimbă fizica**; e doar randare.  
> Dacă nu “se vede”, nu mări `dt`; folosește `--viz-scale` (0 = auto).

---

## 6) Reguli rapide (stabilitate & așteptări)

- **Langevin (underdamped)**: alege \(\Delta t \ll \tau_v = m/\gamma\).

  - scurt: \(\langle \Delta x^2 \rangle \sim (k_BT/m) t^2\) (balistic)
  - lung: \(\langle \Delta x^2 \rangle \sim 2 D t\), \(D=k_BT/\gamma\)

- **Brownian (overdamped)**: pe axă \(\mathrm{std}\_\alpha^2 \approx 2Dt\); în \(d\) dim: \(\langle r^2 \rangle \approx 2dDt\).

---

## 7) Dependențe numerice

- **NumPy** — `ndarray`, operații vectorizate, RNG gaussian (`np.random.normal(size=(N, dims))`).
- **Pandas** — scriere `.dat` (TSV) convenabilă (`DataFrame.to_csv`).
- **VPython** — animație 3D + grafice 2D (`graph`, `gcurve`).

---

## 8) Performanță (tips)

- Setează thread-urile BLAS via `.env` (vezi mai sus).
- Poți folosi `float32` (mai puțină memorie; verifică precizia).
- Refolosește buffere (evită alocări în bucle).
- Evită `r_series` uriaș (vizualizare) pentru N/T foarte mari.

---

## 9) Troubleshooting

- **Analiză: “prea puține puncte la fit”** — rulează fără `--tmin/--tmax`; sau `--fit-last-frac 0.5` / `--fit-last-n 100`; ori `--use-index`.
- **VPython “bilă albă”** — `pip install vpython`; încearcă `--viz-scale 0` (auto-scale).
- **`.env` ignorat în VS Code** — setează `python.envFile=${workspaceFolder}/.env` și **repornește terminalul/VS Code**.
- **Eroare la citirea .dat** — fișierele trebuie să conțină `t` și coloanele `x_mean…`, `std_x…`; delimitatorii sunt auto-detectați.

---

## 10) Licență / contribuții

MIT License

---

**Design**: Interfețele (`.step`, `.positions`, `.velocities`, `.set_state`) și formele `(N, dims)` rămân constante → poți adăuga ușor PBC, coliziuni, forțe externe etc., respectând substituibilitatea (LSP).
