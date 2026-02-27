# Manifold Remapping

Neural manifold analysis for mice circular-arena head-direction data.  
23 mice (WT vs 5xFAD × young vs old) with calcium-imaging recordings across multiple sessions and FOVs.

## Setup

```bash
# 1. Clone & enter the repo
cd manifold-remapping

# 2. Create your local config (git-ignored)
cp .env.example .env
# → edit .env to set DATA_ROOT and FIGURES_ROOT to your OneDrive paths

# 3. Create venv + install
uv venv
source .venv/bin/activate
uv pip install -e .

# 4. Verify
python -c "from remapping.dataset import MiceDataset; print('OK')"
```

## Using Notebooks

Open any notebook in VS Code and select the `.venv` Python as kernel.  
All dependencies (including `ipykernel`) are installed by `uv pip install -e .`.

Start every notebook with:

```python
from remapping.dataset import MiceDataset, Animals
from remapping.plotting import BehaviorColors, PlotStyle, get_figures_path

mice = MiceDataset()
colors = BehaviorColors()
style = PlotStyle()
```

## Quick API Reference

```python
from remapping.dataset import MiceDataset, Animals, GROUP_ORDER, SESSION_ORDER

mice = MiceDataset()

# Subject metadata
mice.get_all_subjects()                          # all 23 Animals
mice.get_subjects_by_group("WT", "old")          # [Animals.M62, ...]
mice.get_subject_info(Animals.M62)               # {'genotype': 'WT', 'age': 'old'}
mice.get_group_palette()                         # {'WT_young': '#f6a6c1', ...}

# Discover recordings (auto-scanned from filesystem)
mice.get_available_fovs(Animals.M62)             # [1, 2, '1s2']
mice.get_available_sessions(Animals.M62, 1)      # ['fam1fam2', 'fam1nov', ...]
mice.get_available_runs(Animals.M62, 1, 'fam1fam2')  # ['fam1', 'fam2']

# Lightweight queries (no full file load)
mice.get_n_neurons(Animals.M62, 1, 'fam1fam2', 'fam1')  # int, reads schema only
mice.get_duration(Animals.M62, 1, 'fam1fam2', 'fam1')   # float seconds

# Load raw data
df = mice.load_data(Animals.M62, 1, 'fam1fam2', 'fam1')

# Load fully processed data (bin → smooth → sqrt → tuning curves)
fr, phi, time, (cells, registered), tc, phi_bins = mice.load_all_data_from_spikes_binned_smoothed(
    Animals.M62, 1, 'fam1fam2', 'fam1',
    only_moving=False, bins_compress=3, bins_smoothing=3, bins_phi=360,
)
```

## Data Format

All data is stored as **Parquet** files (Snappy compression) under `DATA_ROOT`:

```
DATA_ROOT/
└── {genotype}_{age}_{mXX}/
    ├── {mXX}_fov{N}_{session}-{run}_spikes.parquet
    └── {mXX}_fov{N}_global_index_ref.parquet
```

### Virtual FOVs (s2/s3)

Some mice have sessions recorded on a second day for the same physical FOV.  
These are exposed as virtual FOVs (e.g. `"1s2"`), combining s2-specific sessions with shared sessions from the base FOV.

## Package Modules

| Module | Contents |
|--------|----------|
| `dataset` | `MiceDataset`, `Animals`, `GROUP_ORDER`, `SESSION_ORDER`, `EXPERIMENT_ORDER`, colour dicts |
| `processing` | `smooth_tuning_curves_circularly`, `get_tuning_curves`, `average_by_phi_bin` |
| `metrics` | `safe_corrcoef`, `r_squared`, `rmse`, `nrmse`, `rsa_spearman` |
| `alignments` | `procrustes`, `canoncorr` (CCA) |
| `decoders` | `WienerFilterRegression` |
| `plotting` | `BehaviorColors`, `PlotStyle`, `get_figures_path` |

See [AGENTS.md](AGENTS.md) for full documentation.
