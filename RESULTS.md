# RESULTS.md — Cumulative Analysis Findings

> **Purpose**: persistent registry of quantitative results and scientific insights.
> Updated after each notebook is completed, so that context accumulates across sessions
> without re-reading every notebook. Downstream analyses can cite upstream numbers directly.

---

## How to use this file

- **After completing a notebook**: append a new entry under the relevant section.
- **Format**: each entry has a short title, the source notebook, key numbers, and an interpretation line.
- **Numbers**: always include sample sizes, point estimates, and spread (IQR or CI).
- **Status tags**: `[confirmed]` = robust finding, `[preliminary]` = needs more validation, `[superseded]` = replaced by a later analysis.

---

## 0 — Data Overview

Source: `notebooks/0.data_overview/`

### Dataset size
- **23 mice**: 4 groups (WT young=6, WT old=6, 5xFAD young=5, 5xFAD old=6)
- **321 recordings** pass the ≥ 30 neuron threshold (from all sessions/FOVs)
- 42 unique animal-FOV combinations tracked across sessions

### Recording statistics
- Typical neuron count: ranges from ~30 to ~400+ per recording
- Sampling rate: 30.9 Hz raw → ~10.3 Hz after 3× temporal binning

### 0.3 — Firing-rate decoding: fam1 encodes space best, fam2 is close [confirmed]
Source: `03_firing_rate_decoding.ipynb`

RidgeCV (auto-tuned α, 5-fold shuffled CV) to decode angular position [sin φ, cos φ] and normalised time from firing rates. 321 recordings (≥ 30 neurons).

**Spatial decoding R² by run type, median [Q1, Q3]**:

| Run type | n | Space R² | Ang. error (°) |
|----------|---|----------|----------------|
| fam1 | 204 | 0.751 [0.605, 0.826] | 19.5° [14.9, 28.8] |
| fam2 | 42 | 0.709 [0.614, 0.795] | 22.2° [16.7, 28.1] |
| nov | 41 | 0.586 [0.450, 0.643] | 30.5° [26.4, 39.1] |
| fam1rev | 34 | 0.612 [0.462, 0.689] | 28.4° [23.7, 38.9] |

**Temporal decoding R²**:

| Run type | Time R² |
|----------|---------|
| fam1 | 0.738 [0.653, 0.823] |
| fam2 | 0.737 [0.633, 0.831] |
| nov | 0.699 [0.522, 0.761] |
| fam1rev | 0.676 [0.623, 0.767] |

**Interpretation**: Spatial decoding is best for fam1, slightly lower for fam2, and notably worse for novel and reversed environments. fam2 is close to fam1 — the drop is modest (~0.04 R²). The largest gap is between familiar (fam1/fam2) and unfamiliar/changed environments (nov/fam1rev). Temporal decoding is high and similar across all run types (~0.7), consistent with slow drift being a general property of firing rates in all conditions. Both metrics scale with neuron count.

---

## 1 — Embeddings

Source: `notebooks/1.embeddings/`

### 1.0 — PCA on tuning curves captures a ring manifold [confirmed]
Source: `00_pca_manifold.ipynb`

- The first 3 PCs of the (360 × N) tuning-curve matrix form a clear **ring** in PC space for every recording.
- The ring is colour-ordered by angular position (φ), confirming that the dominant population structure is the angular-position signal.
- 321 recordings analysed (all with ≥ 30 neurons).
- **No obvious group differences** in variance-explained profiles across genotype × age groups.

### 1.1 — Common-space PCA: rings are stable across sessions [confirmed]
Source: `01_pca_manifolds_common_space.ipynb`

- 42 FOVs with matched cells across sessions.
- Median cumulative variance in first 3 common PCs: **44.1%**.
- 1079 pairwise comparisons (283 within-session, 796 across-session).
- Ring manifolds in the common PC space preserve angular-position structure across recording days.
- Within-session pairs are more similar than across-session pairs (as expected from the shared cell population and closer temporal proximity).

### 1.2 — TC-PCA and PLSSVD recover the same ring manifold [confirmed]
Source: `02_tc_pca_vs_plssvd.ipynb`

Two independent embedding methods — one unsupervised (TC-PCA: PCA on tuning curves, project firing rates) and one supervised (PLSSVD: maximise covariance with angular-position regressors) — produce nearly identical 3-D ring embeddings.

**Quantitative agreement (321 recordings, median [Q1, Q3])**:

| Metric | All | WT young (n=105) | WT old (n=32) | 5xFAD young (n=71) | 5xFAD old (n=113) |
|--------|-----|-------------------|----------------|----------------------|--------------------|
| RSA (Spearman ρ) | 0.963 [0.925, 0.981] | 0.967 [0.931, 0.981] | 0.949 [0.902, 0.971] | 0.972 [0.942, 0.984] | 0.949 [0.905, 0.975] |
| CCA dim-1 | 0.974 [0.959, 0.984] | 0.978 [0.963, 0.987] | 0.968 [0.950, 0.976] | 0.980 [0.970, 0.987] | 0.967 [0.956, 0.977] |
| R² | 0.885 [0.812, 0.925] | 0.889 [0.829, 0.927] | 0.858 [0.816, 0.902] | 0.906 [0.855, 0.933] | 0.858 [0.771, 0.915] |

**Chance level (circular φ-shift null, 23 mice × 100 shuffles)**:

| Metric | Null mean | Null 95th pctile | Observed median |
|--------|-----------|------------------|-----------------|
| RSA | 0.678 | 0.932 | 0.963 |
| CCA dim-1 | 0.680 | 0.918 | 0.974 |
| R² | 0.374 | 0.665 | 0.885 |

**Interpretation**: The angular-position signal so dominates CA1 population activity that any reasonable dimensionality-reduction method — supervised or unsupervised — recovers the same ring. This validates using either method interchangeably for downstream alignment analyses.

**Subtle group pattern**: young mice (both WT and 5xFAD) show slightly tighter PCA–PLSSVD agreement than old mice, suggesting the angular-position signal may be somewhat noisier in aged animals. This is a trend, not a dramatic effect.

### 1.3 — Fam2 firing-rate PCs do NOT show elevated time correlation [preliminary]
Source: `03_fam2_correlates.ipynb`

PCA on firing rates directly (not tuning curves), 321 recordings (≥ 30 neurons), grouped by run type (fam1 n=204, fam2 n=42, nov n=41, fam1rev n=34; fam1r2 grouped with fam1).

**Time correlation — Max |r(PC, normalised time)| across PC1–3, median [Q1, Q3]**:

| Run type | n | Max |r(PC, time)| |
|----------|---|-------------------|
| fam1 | 204 | 0.160 [0.087, 0.266] |
| fam2 | 42 | 0.171 [0.109, 0.287] |
| nov | 41 | 0.165 [0.093, 0.293] |
| fam1rev | 34 | 0.159 [0.107, 0.221] |

**Angular-position encoding — Max R²(PC ~ sin φ + cos φ), median [Q1, Q3]**:

| Run type | Max R²(PC ~ φ) | Max |r(PC, φ)| |
|----------|-----------------|-----------------|
| fam1 | 0.350 [0.031, 0.516] | 0.370 [0.122, 0.486] |
| fam2 | 0.156 [0.042, 0.440] | 0.260 [0.138, 0.432] |
| nov | 0.109 [0.017, 0.253] | 0.211 [0.084, 0.378] |
| fam1rev | 0.262 [0.014, 0.357] | 0.321 [0.076, 0.410] |

**Variance explained (cumulative PC1–3)**:

| Run type | Cum. var PC1–3 |
|----------|----------------|
| fam1 | 0.170 [0.136, 0.201] |
| fam2 | 0.167 [0.140, 0.201] |
| nov | 0.137 [0.101, 0.164] |
| fam1rev | 0.141 [0.121, 0.153] |

**Interpretation**: Contrary to the initial hypothesis, fam2 PCs do **not** show substantially higher time correlation than other environments — all run types have similar median time |r| (~0.16). However, fam2 and especially nov environments show **lower angular-position encoding** (lower sin/cos R² and linear |r|) than fam1, consistent with the idea that fam2 manifolds are less spatially structured.  The dissimilarity of fam2 manifolds across animals likely stems from weaker or less consistent spatial tuning rather than temporal drift. Nov environments show the least spatial encoding overall, which may also explain why novel manifolds are harder to align.

**Group breakdown**: The pattern is broadly consistent across all 4 genotype × age groups. No genotype- or age-specific interaction is apparent from the faceted distributions.

---

## 2 — Alignment

Source: `notebooks/2.alignment/`

*(No results yet)*

---

## 3 — Figures

Source: `notebooks/3.figures/`

*(No results yet)*

---

## Open Questions

1. Does the ring manifold geometry change systematically between familiar vs novel vs reversed environments?
2. Is the slight age-related decrease in PCA–PLSSVD agreement driven by reduced angular-position tuning, or by increased noise?
3. How do Procrustes-aligned ring manifolds relate across sessions within the same FOV?
