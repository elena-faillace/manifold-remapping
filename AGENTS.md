# AGENTS.md — Manifold Remapping Project

## Project Purpose

Clean, final-figures repository for analysing neural manifold representations in mice navigating a circular arena.  
The dataset comprises **23 mice** (WT vs 5xFAD genotypes × young vs old ages) with calcium-imaging recordings across multiple sessions and fields of view (FOVs).

The core scientific question: **How does the low-dimensional ring manifold of head-direction tuning curves relate across sessions, conditions, genotypes, and ages?**

---

## Repo Layout

```
manifold-remapping/
├── AGENTS.md                 # ← you are here
├── .env.example              # template — copy to .env and fill in your paths
├── .gitignore
├── pyproject.toml            # uv-managed, editable install
├── README.md
├── src/
│   └── remapping/
│       ├── __init__.py       # re-exports key classes/functions
│       ├── config.py         # loads .env → DATA_ROOT, FIGURES_ROOT
│       ├── dataset.py        # MiceDataset, Animals enum
│       ├── processing.py     # smooth_tuning_curves_circularly, get_tuning_curves, average_by_phi_bin
│       ├── metrics.py        # safe_corrcoef, r_squared, rmse, nrmse, rsa_spearman
│       ├── alignments.py     # procrustes, canoncorr (CCA)
│       ├── decoders.py       # WienerFilterRegression
│       └── plotting.py       # BehaviorColors, PlotStyle, figure-saving helpers
└── notebooks/
    ├── 0.data_overview/
    │   ├── 00_load_data.ipynb
    │   ├── 01_experiments_data_statistics.ipynb
    │   └── 02_neural_data_statistics.ipynb
    ├── 1.embeddings/
    ├── 2.alignment/
    └── 3.figures/
```

---

## Setup Instructions

```bash
# 1. Clone & enter
cd manifold-remapping

# 2. Create env file
cp .env.example .env
#    → edit .env to set DATA_ROOT and FIGURES_ROOT to your local paths

# 3. Create venv + install
uv venv
source .venv/bin/activate
uv pip install -e .

# 4. Verify
python -c "from remapping.dataset import MiceDataset; print('OK')"
```

---

## Configuration — `.env`

The `.env` file (git-ignored) holds two paths:

```
DATA_ROOT=/Users/you/path/to/datasets/elena_anns
FIGURES_ROOT=/Users/you/path/to/figures
```

These are loaded by `remapping.config` and used everywhere instead of hardcoded paths.  
**Never hardcode absolute paths in source code or notebooks.**

---

## Package Modules — What Each Contains

### `config.py`
- Loads `.env` via `python-dotenv`
- Exposes `DATA_ROOT: Path` and `FIGURES_ROOT: Path`
- Both raise `FileNotFoundError` at import time if the `.env` is missing or paths don't exist

### `dataset.py`
- **`Animals`** enum: all 23 mouse IDs (`M62`..`M141`)
- **Module-level constants** (importable directly):
  - `GROUP_ORDER` — canonical display order for genotype × age groups
  - `SESSION_ORDER` — canonical display order for session types
  - `EXPERIMENT_ORDER` — canonical display order for `(session, run)` pairs
  - `SESSION_TYPE_MAP` — maps each of the 8 session types to one of 3 recording-day categories
  - `EXPERIMENT_TYPE_ORDER` — display order for the 3 recording-day categories
  - `EXPERIMENT_TYPE_COLORS` — one colour per recording-day category
  - `SESSION_COLORS` — one colour per session type
  - `COLORS_EXPERIMENTS`, `COLORS_GROUPS`, `COLORS_SUBJECTS` — colour dictionaries
  - `_SUBJECTS_CONFIG` — genotype/age metadata per `Animals` member
  - `_FILENAME_RE` — regex for parsing Parquet filenames
- **`MiceDataset`** class — the central data interface:
  - Subject metadata: `get_all_subjects()`, `get_subjects_by_genotype(g)`, `get_subjects_by_age(a)`, `get_subjects_by_group(g, a)`, `get_subject_info(subject)`, `get_genotypes()`, `get_ages()`
  - Colours: `get_experiment_color(session, run)`, `get_subject_color(subject, fov)`, `get_colors_genotype_age(genotype, age)`, `get_group_palette()`
  - Recording discovery (filesystem scan, **cached** after first call per subject): `get_available_recordings(subject)`, `get_available_fovs(subject)`, `get_available_sessions(subject, fov)`, `get_available_runs(subject, fov, session)`
  - Data loading: `load_data(subject, fov, session, run)` → raw spikes DataFrame; `load_spikes_binned(...)` → binned arrays; `load_all_data_from_spikes_binned_smoothed(...)` → fully processed firing rates + tuning curves
  - Lightweight queries: `get_n_neurons(...)` → int (reads Parquet schema only); `get_duration(...)` → float seconds (reads only time column)
  - Cell indexing: `from_local_to_global_cell_index(subject, fov, session, local_indexes)`
  - Tuning curves: `get_tuning_curves(firing_rates, phi, n_points)` → `(ring_neural, phi_bins)` — vectorised with `np.bincount`

### `processing.py`
- `smooth_tuning_curves_circularly(tuning_curves, kernel_size)` — circular moving-average
- `get_tuning_curves(firing_rates, phi, n_points)` — standalone version (also on MiceDataset)
- `average_by_phi_bin(embedding, phi, n_bins)` — bin a T×K trajectory into phi bins

### `metrics.py`
- `safe_corrcoef(arr1, arr2)` — Pearson r handling constant arrays
- `r_squared(y_true, y_pred)` — coefficient of determination
- `rmse(y_true, y_pred)` — root mean squared error
- `nrmse(y_true, y_pred, normalization)` — normalized RMSE (range or std)
- `rsa_spearman(mat1, mat2)` — Spearman r between pairwise distance matrices

### `alignments.py`
- `canoncorr(X, Y, fullReturn=False)` — CCA (MATLAB port)
- `procrustes(X, Y, scaling=True, reflection='best')` — Procrustes alignment

### `decoders.py`
- `WienerFilterRegression(regularization=None)` — simple linear decoder (None / "l2" / "LARS")

### `plotting.py`
- `BehaviorColors` class — colour constants for behavioural variables
- `PlotStyle` class — A4-optimised rcParams (min 7 pt text); exposes `FULL_WIDTH`, `HALF_WIDTH`, `THIRD_WIDTH` figure widths in inches
- `get_figures_path(*subdirs)` → `FIGURES_ROOT / subdirs`, creating dirs as needed

---

## Data Description

### Terminology

| Term | Meaning |
|------|--------|
| **Session** | A recording day, named after the runs it contains (e.g. `fam1fam1revfam1` = 3 runs) |
| **Run** | A single continuous recording within a session (`fam1`, `fam1rev`, `fam1r2`, `nov`, `fam2`) |
| **FOV** | Field of view — an imaging location that can be tracked across days |
| **Virtual FOV** | A FOV with suffix `s2`/`s3` for sessions recorded on a second/third visit to the same physical FOV |

### Subject Groups

| Group | Count | IDs |
|-------|-------|-----|
| WT old | 6 | M62, M66, M70, M116, M117, M133 |
| WT young | 6 | M120, M127, M129, M130, M134, M135 |
| 5xFAD old | 6 | M111, M118, M125, M139, M140, M141 |
| 5xFAD young | 5 | M77, M79, M121, M128, M132 |

### File System Layout (under `DATA_ROOT`)

```
DATA_ROOT/
└── {genotype}_{age}_{mXX}/
    ├── {mXX}_fov{N}_{session}-{run}_spikes.parquet
    └── {mXX}_fov{N}_global_index_ref.parquet
```

### Session Types

| Session | Runs | Description |
|---------|------|-------------|
| `fam1fam2` | fam1, fam2 | Familiar → different familiar |
| `fam1fam2fam1` | fam1, fam2, fam1r2 | Familiar → different familiar → return |
| `fam1nov` | fam1, nov | Familiar → novel |
| `fam1novfam1` | fam1, nov, fam1r2 | Familiar → novel → return |
| `fam1fam1rev` | fam1, fam1rev | Familiar → reversed direction |
| `fam1fam1revfam1` | fam1, fam1rev, fam1r2 | Familiar → reversed → return |
| `fam1fam2s2/s3` | fam1, fam2 | Repeat sessions (virtual FOV "Xs2") |

### Data Processing Pipeline

```
raw spikes parquet
  → optional movement filter (movement_status == "moving")
  → bin 3→1 (sum spikes, mean phi/time)
  → Gaussian smooth (σ=3 bins ≈ 0.29s at ~10.3 Hz)
  → square root
  → tuning curves (360 angular bins, then circular smooth with kernel=20)
```

**Sampling rate**: 30.9 Hz → after 3×binning ≈ 10.3 Hz

---

## Notebook Conventions

### Imports — No sys.path hacks
```python
# The package is installed editably, so just:
from remapping.dataset import MiceDataset, Animals
from remapping.processing import smooth_tuning_curves_circularly
from remapping.plotting import BehaviorColors, PlotStyle, get_figures_path
from remapping.metrics import safe_corrcoef, r_squared
```

### Standard First Cell
```python
from remapping.dataset import MiceDataset, Animals
from remapping.plotting import BehaviorColors, PlotStyle, get_figures_path

mice = MiceDataset()
colors = BehaviorColors()
style = PlotStyle()
```

### Imports — No sys.path hacks
```python
# The package is installed editably, so just:
from remapping.dataset import MiceDataset, Animals
from remapping.processing import smooth_tuning_curves_circularly
from remapping.plotting import BehaviorColors, PlotStyle, get_figures_path
from remapping.metrics import safe_corrcoef, r_squared
```

### Parameters Cell
Every notebook has a clearly marked **Parameters** cell near the top defining all tuneable values.

### Figure Saving
Figures are saved as PDF in `FIGURES_ROOT` with a folder structure that **mirrors the notebook path**.  
For a notebook at `notebooks/X/Y.ipynb`, figures go to `FIGURES_ROOT/X/Y/`.

```python
# In the setup cell:
fig_path = get_figures_path("0.data_overview", "01_experiments_data_statistics.ipynb")

# After each plot (before plt.show()):
fig.savefig(fig_path / "descriptive_name.pdf")
plt.show()
```

`savefig.bbox` and `savefig.dpi` are set globally by `PlotStyle` (`tight`, 300 dpi).

### No inline helper re-definitions
All reusable functions live in `src/remapping/`. Notebooks only contain plotting/analysis logic specific to that notebook.

---

## Plotting Conventions

### Figure Sizes (A4 paper, 20 mm margins)

| Constant | Width | Use |
|----------|-------|-----|
| `PlotStyle.FULL_WIDTH` | 6.7″ | Single-column figure |
| `PlotStyle.HALF_WIDTH` | 3.3″ | Two panels side-by-side |
| `PlotStyle.THIRD_WIDTH` | 2.2″ | Three panels |

```python
fig, ax = plt.subplots(figsize=(PlotStyle.FULL_WIDTH, 3.5))
```

### Font Sizes (all ≥ 7 pt)

| Element | Size | rcParam |
|---------|------|---------|
| Figure title | 11 pt | `figure.titlesize` |
| Axes title | 10 pt | `axes.titlesize` |
| Axis labels | 9 pt | `axes.labelsize` |
| Legend title | 9 pt | `legend.title_fontsize` |
| Tick labels | 8 pt | `xtick.labelsize`, `ytick.labelsize` |
| Legend text | 8 pt | `legend.fontsize` |
| Base font | 8 pt | `font.size` |

### Experiment-Type Grouping (3 Recording Days)

All 8 session types collapse into **3 recording-day categories** via `SESSION_TYPE_MAP`:

| Category | Sessions | Colour |
|----------|----------|--------|
| fam1 → fam2 | `fam1fam2`, `fam1fam2fam1`, `fam1fam2s2`, `fam1fam2s3` | `#ff8800` (orange) |
| fam1 → novel | `fam1nov`, `fam1novfam1` | `#99ca3c` (green) |
| fam1 → reversed | `fam1fam1rev`, `fam1fam1revfam1` | `#d727fc` (purple) |

- **fam1** and **fam2** are distinct familiar environments (not the same environment twice)
- **novel** is a never-before-seen environment
- **reversed** is fam1 with reversed running direction
- s2/s3 revisit sessions are grouped with fam1 → fam2 (same experiment type, just a later visit)

Use `EXPERIMENT_TYPE_ORDER` for display order and `EXPERIMENT_TYPE_COLORS` for colouring.

### Standard Colour Palettes

| Element | Source | Example |
|---------|--------|---------|
| Genotype × age group | `mice.get_colors_genotype_age(g, a)` | WT young → pink, 5xFAD old → teal |
| Recording-day category | `EXPERIMENT_TYPE_COLORS[label]` | fam1 → fam2 → orange |
| Experiment (session, run) | `mice.get_experiment_color(session, run)` | fam1fam2/fam1 → orange |
| Session type | `SESSION_COLORS[session]` | fam1fam1rev → purple |
| Subject × FOV | `mice.get_subject_color(subject, fov)` | Per-mouse unique colour |

### Group and Session Ordering

Groups are always ordered: `["WT_young", "WT_old", "5xFAD_young", "5xFAD_old"]`

Sessions/runs follow `mice.order_experiments` canonical order:
1. fam1fam2: fam1 → fam2
2. fam1fam2fam1: fam1 → fam2 → fam1r2
3. fam1fam2s2/s3: fam1 → fam2
4. fam1nov: fam1 → nov
5. fam1novfam1: fam1 → nov → fam1r2
6. fam1fam1rev: fam1 → fam1rev
7. fam1fam1revfam1: fam1 → fam1rev → fam1r2

### Standard Plot Template

```python
from remapping.dataset import MiceDataset, GROUP_ORDER
from remapping.plotting import PlotStyle

mice = MiceDataset()
style = PlotStyle()
palette = mice.get_group_palette()

fig, ax = plt.subplots(figsize=(PlotStyle.FULL_WIDTH, 3.5))
# ... plot ...
ax.set_xlabel("Label")
ax.set_ylabel("Label")
ax.set_title("Title")
plt.tight_layout()
```

---

## Coding Standards

- **Minimalist**: use the minimum code necessary. No over-engineering.
- **No generic error handling**: code should break explicitly when problems occur. No try/except blocks unless there's a specific recovery action.
- **Step-by-step**: break complex tasks into small, testable steps.
- **Type hints on public functions**: use standard Python type hints.
- **Docstrings**: Google-style, brief.
- **Dependencies**: only those in `pyproject.toml`. Stick to numpy, pandas, scipy, scikit-learn, matplotlib, seaborn.

---

## Summary of notebooks

### `00_load_data.ipynb` — Data Loading Tutorial
Shows the full API of `MiceDataset`:
1. List genotypes, ages, subjects per group
2. Pick a subject → discover FOVs, sessions, runs
3. Load raw data with `load_data()`
4. Load processed data with `load_all_data_from_spikes_binned_smoothed()`
5. Show effect of parameters: `only_moving`, `bins_compress`, `bins_smoothing`, `bins_phi`
6. Plot example tuning curves (a few neurons) and firing rate traces
7. Show tuning curve smoothing with `smooth_tuning_curves_circularly(kernel_size=20)`

### `01_experiments_data_statistics.ipynb` — Experiment Overview
1. Table: subjects × groups with counts
2. Table: total FOVs, sessions, runs per subject (auto-discovered from filesystem)
3. Boxplots: number of neurons per recording — left by group, right by experiment day (3 categories)
4. Bar chart: session-type counts colour-coded with `SESSION_COLORS`
5. Heatmap: subjects × session-types showing FOV availability, with group colour bar (left) and experiment-type colour bar (bottom), both with grey labels
6. Histogram: run duration distribution in minutes (bins=15, KDE overlay)
7. Bar chart: total recording time per group in minutes

All figures saved to `FIGURES_ROOT/0.data_overview/01_experiments_data_statistics/`.

### `02_neural_data_statistics.ipynb` — Neural Data Overview
1. Per-recording spike statistics table: n_neurons, duration, mean/std spike rate (Hz), fraction active, peak amplitude
2. Example raster plot — raw spike events for 20 neurons over 60 s, colour-coded by amplitude
3. Angular-position tuning curve selectivity (peak/mean) — horizontal KDE distributions by group

All figures saved to `FIGURES_ROOT/0.data_overview/02_neural_data_statistics/`.
