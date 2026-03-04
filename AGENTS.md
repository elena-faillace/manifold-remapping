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
├── RESULTS.md                # cumulative analysis results & scientific insights
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
    │   ├── 02_neural_data_statistics.ipynb
    │   └── 03_firing_rate_decoding.ipynb
    ├── 1.embeddings/
    │   ├── 00_pca_manifold.ipynb
    │   ├── 01_pca_manifolds_common_space.ipynb
    │   ├── 02_tc_pca_vs_plssvd.ipynb
    │   └── 03_fam2_correlates.ipynb
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
  - Data loading: `load_data(subject, fov, session, run)` → raw spikes DataFrame; `load_spikes(...)` → binned spike arrays; `load_firing_rates(...)` → Gaussian-smoothed firing rates + tuning curves
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

### Distribution Plots — No Bar Charts

**Never use bar plots.** Prefer violin plots or horizontal KDE distributions.  
**Always overlay the actual data points** (e.g. `sns.stripplot`, `ax.scatter` with jitter) so individual observations remain visible.  
**Always annotate violins with summary statistics**: show Q1/Q3 as thin black lines, median as a thicker black line, and mean as a white diamond with black edge. Use a helper function (`annotate_violins`) rather than `inner="quart"`:

```python
def annotate_violins(ax, data, x, y, order, width=0.15):
    """Overlay Q1/Q3 lines, median line, and mean diamond on violin plots."""
    for i, cat in enumerate(order):
        vals = data.loc[data[x] == cat, y].dropna()
        if len(vals) == 0:
            continue
        q1, med, q3 = vals.quantile([0.25, 0.5, 0.75])
        mean_val = vals.mean()
        ax.hlines([q1, q3], i - width, i + width, color="k", linewidth=0.8, zorder=3)
        ax.hlines(med, i - width, i + width, color="k", linewidth=1.5, zorder=3)
        ax.scatter(i, mean_val, marker="D", color="white", edgecolors="k",
                   s=14, linewidths=0.8, zorder=4)

# Good — violin with strip overlay + summary stats
sns.violinplot(data=df, x="group", y="value", hue="group", palette=palette,
               inner=None, alpha=0.3, legend=False)
sns.stripplot(data=df, x="group", y="value", hue="group", palette=palette,
              size=3, alpha=0.7, jitter=True, legend=False)
annotate_violins(ax, df, "group", "value", GROUP_ORDER)

# Good — horizontal KDE with individual points
for group in GROUP_ORDER:
    subset = df.loc[df["Group"] == group, "value"]
    sns.kdeplot(subset, ax=ax, color=palette[group], fill=True, alpha=0.25, label=group)
```

Exceptions: count-based summaries (e.g. session-type counts) where raw data points don't exist may still use bars.

### Terminology

- Use **"angular position"** (not "head direction") when referring to the `phi` variable.
- fam1 and fam2 are **distinct** familiar environments (not the same environment twice).

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
- **Warning-free notebooks**: after creating or editing a notebook, run all cells and fix any warnings or errors before considering the work done. Common pitfalls:
  - `plt.tight_layout()` is incompatible with 3-D axes and `gridspec` — use `fig.set_layout_engine("constrained")` or manual `fig.subplots_adjust()` instead.
  - Seaborn `palette` without `hue` is deprecated (v0.14+) — always pass `hue=` matching the `x` variable with `legend=False`.
  - `ax.set_xticklabels()` without `ax.set_xticks()` first triggers a `UserWarning` — always call `set_xticks()` before `set_xticklabels()`.

---

## Summary of notebooks

### `00_load_data.ipynb` — Data Loading Tutorial
Shows the full API of `MiceDataset`:
1. List genotypes, ages, subjects per group
2. Pick a subject → discover FOVs, sessions, runs
3. Load raw data with `load_data()`
4. Load processed data with `load_firing_rates()`
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
2. Spike rate distributions — violin + strip overlay of mean rate and rate std by genotype × age group
3. Example raster plot — raw spike events for 20 neurons over 60 s, heatmap coloured by amplitude
4. Angular-position tuning curve selectivity (peak/mean) — overlaid KDE distributions by group
5. Firing rate traces — example neurons showing Gaussian-smoothed + √-transformed activity
6. Smoothing kernel sweep — median pairwise correlation vs σ, one recording per group
7. Pairwise correlation distributions — KDE at σ = 0 / pipeline / wide for one example recording

All figures saved to `FIGURES_ROOT/0.data_overview/02_neural_data_statistics/`.

### `03_firing_rate_decoding.ipynb` — Firing-Rate Decoding: Space vs Time
Linear regression (RidgeCV, auto-tuned α) to decode angular position and elapsed time from firing rates. 5-fold shuffled CV, 321 recordings (≥ 30 neurons).

1. Data collection: per-recording RidgeCV decoding of [sin φ, cos φ] (space) and normalised time, run types: fam1, fam2, nov, fam1rev (fam1r2 grouped with fam1)
2. Spatial decoding by run type (violin + strip): R² and mean angular error (°)
3. Temporal decoding by run type (violin + strip): R²
4. Space vs time scatter (coloured by run type)
5. Spatial and temporal decoding by genotype × age group (2×2 faceted violins)
6. Decoding R² vs number of neurons (scatter)
7. Example decoding traces: true vs predicted angular position and time (1 per run type, median-R² recordings)
8. Summary table: median [Q1, Q3] and mean ± SD per run type

All figures saved to `FIGURES_ROOT/0.data_overview/03_firing_rate_decoding/`.

---

### `1.embeddings/00_pca_manifold.ipynb` — PCA Manifold of Tuning Curves
PCA on the (360 × N) tuning-curve matrix for each recording independently.

1. Example recording: PCA on tuning curves → ring in PC1–PC2–PC3, scree plot
2. Firing-rate trajectories projected onto tuning-curve PCs (time-resolved 3-D)
3. Group-level variance-explained profiles (321 recordings, all with ≥ 30 neurons):
   - Cumulative variance in first 3 PCs by group (violin + strip)
   - Mean scree curves by group (first 10 PCs)
   - Variance in first 3 PCs by experiment type (3 recording-day categories)
4. Gallery: PCA rings coloured by angular position for multiple subjects per group

All figures saved to `FIGURES_ROOT/1.embeddings/00_pca_manifold/`.

### `1.embeddings/01_pca_manifolds_common_space.ipynb` — Common-Space PCA
For each animal-FOV, matches cells across sessions using the global cell index, fits PCA on concatenated tuning curves, and projects each run back into the shared PC space.

1. Collect tuning curves with matched cells across sessions (42 FOVs total)
2. PCA in common space — median cumulative variance (3 PCs): 44.1%
3. Gallery: ring manifolds per animal-FOV, colour-coded by session/run
4. Pairwise ring similarity (Spearman ρ of pairwise distance matrices):
   - 1079 total pairs (283 within-session, 796 across-session)
   - Within- vs across-session similarity by group
   - Within-session similarity split by group and pair type
5. Variance explained in common PC space by group
6. Similarity vs number of matched cells (quality control)

All figures saved to `FIGURES_ROOT/1.embeddings/01_pca_manifolds_common_space/`.

### `1.embeddings/02_tc_pca_vs_plssvd.ipynb` — TC-PCA Projection vs PLSSVD Regression Embedding
Compares two 3-D embedding methods across 321 recordings (≥ 30 neurons):

- **TC-PCA**: PCA on smoothed tuning curves, firing rates projected onto those PCs
- **PLSSVD**: firing rates regressed against Y = [sin φ, cos φ, sin 2φ, cos 2φ]

Sections:
1. Data loading: both embeddings + φ-binned rings for all 321 recordings
2. Visual comparison: 3-D ring plots per group (TC-PCA vs PLSSVD side by side)
3. Quantitative similarity: RSA, CCA dim-1, R² (from PLSSVD → TC-PCA linear map)
4. Chance level: circular time-shift null (23 mice × 100 shuffles)
5. Metric distributions by genotype × age group (violin + strip)
6. Metrics by experiment type (3 recording-day categories)
7. Metrics vs n_neurons (scatter)
8. Gallery: 4 examples per group
9. Summary table: median [Q1, Q3] per group

All figures saved to `FIGURES_ROOT/1.embeddings/02_tc_pca_vs_plssvd/`.

### `1.embeddings/03_fam2_correlates.ipynb` — Firing-Rate PCA: Temporal vs Spatial Encoding by Environment
Tests whether fam2 firing-rate PCs correlate more with time (drift) than fam1 / nov / fam1rev PCs.
PCA on firing rates directly (not tuning curves), 321 recordings (≥ 30 neurons), run types: fam1, fam2, nov, fam1rev (fam1r2 grouped with fam1).

Sections:
1. Data collection: per-recording PCA on firing rates, time correlation (|r| with normalised time), angular-position encoding (linear |r| and sin/cos R²), variance explained
2. Time correlation by run type (violin + strip, combined max and per-PC breakdown)
3. Angular-position encoding by run type (sin/cos R² and linear |r|, combined + per-PC)
4. Time vs space scatter (max time corr vs max phi R², coloured by run type)
5. Variance explained by run type (cumulative PC1–3 and per-PC boxplots)
6. Breakdown by genotype × age group (2×2 faceted violins for time corr and phi R²)
7. Example traces: PC1 vs normalised time colour-coded by angular position (1 per run type)
8. Summary table: median [Q1, Q3] per run type

All figures saved to `FIGURES_ROOT/1.embeddings/03_fam2_correlates/`.

---

## Results Registry

Persistent quantitative findings are tracked in **`RESULTS.md`** at the project root.
This file accumulates key numbers, insights, and conclusions from every notebook so that:
- Context builds up across sessions without re-reading every notebook
- Scientific findings are centralized and cross-referenceable
- Downstream analyses can cite upstream numbers directly

**Convention**: after completing a notebook or discovering a significant result, append it to `RESULTS.md` following the format documented there.
