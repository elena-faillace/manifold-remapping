"""MiceDataset — central data interface for mice circular-arena recordings."""

from enum import Enum
import re

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from .config import DATA_ROOT


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Animals(Enum):
    """Mouse subject identifiers."""
    # WT old
    M62 = "m62"
    M66 = "m66"
    M70 = "m70"
    M116 = "m116"
    M117 = "m117"
    M133 = "m133"
    # WT young
    M120 = "m120"
    M127 = "m127"
    M129 = "m129"
    M130 = "m130"
    M134 = "m134"
    M135 = "m135"
    # 5xFAD old
    M111 = "m111"
    M118 = "m118"
    M125 = "m125"
    M139 = "m139"
    M140 = "m140"
    M141 = "m141"
    # 5xFAD young
    M77 = "m77"
    M79 = "m79"
    M121 = "m121"
    M128 = "m128"
    M132 = "m132"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GROUP_ORDER: list[str] = ["WT_young", "WT_old", "5xFAD_young", "5xFAD_old"]
"""Canonical display order for genotype x age groups."""

SESSION_ORDER: list[str] = [
    "fam1fam2",
    "fam1fam2fam1",
    "fam1fam2s2",
    "fam1fam2s3",
    "fam1nov",
    "fam1novfam1",
    "fam1fam1rev",
    "fam1fam1revfam1",
]
"""Canonical display order for session types (recording days)."""

EXPERIMENT_ORDER: list[tuple[str, str]] = [
    ("fam1fam2", "fam1"), ("fam1fam2", "fam2"),
    ("fam1fam2fam1", "fam1"), ("fam1fam2fam1", "fam2"), ("fam1fam2fam1", "fam1r2"),
    ("fam1fam2s2", "fam1"), ("fam1fam2s2", "fam2"),
    ("fam1fam2s3", "fam1"), ("fam1fam2s3", "fam2"),
    ("fam1nov", "fam1"), ("fam1nov", "nov"),
    ("fam1novfam1", "fam1"), ("fam1novfam1", "nov"), ("fam1novfam1", "fam1r2"),
    ("fam1fam1rev", "fam1"), ("fam1fam1rev", "fam1rev"),
    ("fam1fam1revfam1", "fam1"), ("fam1fam1revfam1", "fam1rev"), ("fam1fam1revfam1", "fam1r2"),
]
"""Canonical display order for (session, run) pairs."""

SESSION_TYPE_MAP: dict[str, str] = {
    "fam1fam2": "fam1 → fam2",
    "fam1fam2fam1": "fam1 → fam2",
    "fam1fam2s2": "fam1 → fam2",
    "fam1fam2s3": "fam1 → fam2",
    "fam1nov": "fam1 → novel",
    "fam1novfam1": "fam1 → novel",
    "fam1fam1rev": "fam1 → reversed",
    "fam1fam1revfam1": "fam1 → reversed",
}
"""Map each session type to one of the 3 recording-day categories.
fam1 and fam2 are distinct familiar environments; novel is a new environment;
reversed is fam1 with reversed running direction.
s2/s3 revisit sessions are grouped with their base experiment type."""

EXPERIMENT_TYPE_ORDER: list[str] = [
    "fam1 → fam2", "fam1 → novel", "fam1 → reversed",
]
"""Display order for the 3 recording-day categories."""

EXPERIMENT_TYPE_COLORS: dict[str, str] = {
    "fam1 → fam2": "#ff8800",      # orange  — from COLORS_EXPERIMENTS fam1fam2/fam1
    "fam1 → novel": "#99ca3c",     # green   — from COLORS_EXPERIMENTS fam1nov/fam1
    "fam1 → reversed": "#d727fc",  # purple  — from COLORS_EXPERIMENTS fam1fam1rev/fam1
}
"""One colour per recording-day category (picked from COLORS_EXPERIMENTS first-run colours)."""

COLORS_EXPERIMENTS: dict[tuple[str, str], str] = {
    ("fam1fam2", "fam1"): "#e85d04",
    ("fam1fam2fam1", "fam1"): "#e85d04",
    ("fam1fam2", "fam2"): "#ff8800",
    ("fam1fam2fam1", "fam2"): "#ff8800",
    ("fam1fam2fam1", "fam1r2"): "#ffba08",
    ("fam1fam2s2", "fam1"): "#0d41e1",
    ("fam1fam2s2", "fam2"): "#0a85ed",
    ("fam1fam2s3", "fam1"): "#0d41e1",
    ("fam1fam2s3", "fam2"): "#0a85ed",
    ("fam1nov", "fam1"): "#208b3a",
    ("fam1novfam1", "fam1"): "#208b3a",
    ("fam1nov", "nov"): "#99ca3c",
    ("fam1novfam1", "nov"): "#99ca3c",
    ("fam1novfam1", "fam1r2"): "#cbdb47",
    ("fam1fam1rev", "fam1"): "#7f25fb",
    ("fam1fam1revfam1", "fam1"): "#7f25fb",
    ("fam1fam1rev", "fam1rev"): "#d727fc",
    ("fam1fam1revfam1", "fam1rev"): "#d727fc",
    ("fam1fam1revfam1", "fam1r2"): "#fd23de",
}

# One representative color per session type (= color of its first run)
SESSION_COLORS: dict[str, str] = {
    s: next(COLORS_EXPERIMENTS[(sess, run)] for sess, run in EXPERIMENT_ORDER if sess == s)
    for s in SESSION_ORDER
}
"""Map each session name to its first-run colour for bar charts."""

COLORS_GROUPS: dict[tuple[str, str], str] = {
    ("WT", "young"): "#f6a6c1",
    ("WT", "old"): "#c2185b",
    ("5xFAD", "young"): "#80cbc4",
    ("5xFAD", "old"): "#00796b",
}

COLORS_SUBJECTS: dict[str, str] = {
    # WT old
    "m62_fov1": "#f7a3ba", "m62_fov2": "#f992ad",
    "m66_fov1": "#fbbcee", "m66_fov2": "#f6a5e4",
    "m70_fov1": "#fab4c8", "m70_fov2": "#f59bb3",
    "m116_fov1": "#f78ecf", "m116_fov2": "#e776c2",
    "m117_fov1": "#b88ae6", "m117_fov2": "#a178db",
    # WT young
    "m120_fov1": "#a062dd", "m120_fov2": "#a480f2", "m120_fov2s2": "#cfb9f7",
    "m127_fov1": "#d4b0f9", "m127_fov2": "#7b6f8d",
    "m129_fov1": "#c580ed", "m129_fov2": "#d199f1",
    "m130_fov1": "#cc9ab5", "m130_fov2": "#b37495",
    "m133_fov1": "#e6b3cc",
    "m134_fov1": "#f08080", "m134_fov2": "#fab9a4",
    "m135_fov1": "#e8749a", "m135_fov2": "#f497b6",
    # 5xFAD young
    "m77_fov1": "#80c7b7", "m77_fov1s2": "#95d1c3", "m77_fov2": "#66b8a8",
    "m79_fov1": "#9bd4c8", "m79_fov2": "#7fc8bb",
    "m121_fov1": "#6fb3a8", "m121_fov2": "#5aa99c",
    "m128_fov1": "#70c1b3", "m128_fov2": "#58b6a7",
    "m132_fov1": "#9ad5ca", "m132_fov2": "#80c9bc",
    # 5xFAD old
    "m111_fov1": "#368f8b", "m111_fov1s2": "#429a96", "m111_fov2": "#2b7c74",
    "m118_fov1": "#2a7f7a", "m118_fov2": "#256d69", "m118_fov3": "#1f5d59",
    "m125_fov1": "#4ba09c",
    "m139_fov1": "#38948f", "m139_fov2": "#2d8580",
    "m140_fov1": "#3a9692", "m140_fov2": "#2e8783", "m140_fov3": "#227874",
    "m141_fov1": "#46a29f", "m141_fov2": "#3b9491", "m141_fov3": "#328985",
}

_SUBJECTS_CONFIG: dict[Animals, dict[str, str]] = {
    # WT old
    Animals.M62: {"genotype": "WT", "age": "old"},
    Animals.M66: {"genotype": "WT", "age": "old"},
    Animals.M70: {"genotype": "WT", "age": "old"},
    Animals.M116: {"genotype": "WT", "age": "old"},
    Animals.M117: {"genotype": "WT", "age": "old"},
    Animals.M133: {"genotype": "WT", "age": "old"},
    # WT young
    Animals.M120: {"genotype": "WT", "age": "young"},
    Animals.M127: {"genotype": "WT", "age": "young"},
    Animals.M129: {"genotype": "WT", "age": "young"},
    Animals.M130: {"genotype": "WT", "age": "young"},
    Animals.M134: {"genotype": "WT", "age": "young"},
    Animals.M135: {"genotype": "WT", "age": "young"},
    # 5xFAD old
    Animals.M111: {"genotype": "5xFAD", "age": "old"},
    Animals.M118: {"genotype": "5xFAD", "age": "old"},
    Animals.M125: {"genotype": "5xFAD", "age": "old"},
    Animals.M139: {"genotype": "5xFAD", "age": "old"},
    Animals.M140: {"genotype": "5xFAD", "age": "old"},
    Animals.M141: {"genotype": "5xFAD", "age": "old"},
    # 5xFAD young
    Animals.M77: {"genotype": "5xFAD", "age": "young"},
    Animals.M79: {"genotype": "5xFAD", "age": "young"},
    Animals.M121: {"genotype": "5xFAD", "age": "young"},
    Animals.M128: {"genotype": "5xFAD", "age": "young"},
    Animals.M132: {"genotype": "5xFAD", "age": "young"},
}

# Regex to parse filenames like  m62_fov2_fam1fam2-fam1_spikes.parquet
_FILENAME_RE = re.compile(
    r"^(?P<mouse>m\d+)_fov(?P<fov>\d+)_(?P<session>[^-]+)-(?P<run>[^_]+)_spikes\.parquet$"
)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MiceDataset:
    """Dataset class for mice circular-arena calcium-imaging data.

    All paths are resolved from ``DATA_ROOT`` (set via ``.env``).
    Colour maps and ordering constants are module-level for easy reuse.
    """

    # Expose constants as class attributes for convenience
    GROUP_ORDER = GROUP_ORDER
    SESSION_ORDER = SESSION_ORDER
    EXPERIMENT_ORDER = EXPERIMENT_ORDER

    def __init__(self):
        self.data_path = DATA_ROOT
        self.sampling_rate = 30.9  # Hz, before binning

        self.subjects = _SUBJECTS_CONFIG
        self._recordings_cache: dict[Animals, dict] = {}

    # ------------------------------------------------------------------
    # Subject metadata
    # ------------------------------------------------------------------

    def get_all_subjects(self) -> list[Animals]:
        """All 23 mouse subjects."""
        return list(self.subjects.keys())

    def get_subjects_by_genotype(self, genotype: str) -> list[Animals]:
        return [s for s, c in self.subjects.items() if c["genotype"] == genotype]

    def get_subjects_by_age(self, age: str) -> list[Animals]:
        return [s for s, c in self.subjects.items() if c["age"] == age]

    def get_subjects_by_group(self, genotype: str, age: str) -> list[Animals]:
        return [
            s for s, c in self.subjects.items()
            if c["genotype"] == genotype and c["age"] == age
        ]

    def get_subject_info(self, subject: Animals) -> dict[str, str]:
        """Return ``{'genotype': ..., 'age': ...}``."""
        return self.subjects[subject]

    def get_genotypes(self) -> list[str]:
        return sorted({c["genotype"] for c in self.subjects.values()})

    def get_ages(self) -> list[str]:
        return sorted({c["age"] for c in self.subjects.values()})

    # ------------------------------------------------------------------
    # Colours (thin wrappers over module constants)
    # ------------------------------------------------------------------

    @staticmethod
    def get_experiment_color(session: str, run: str) -> str:
        return COLORS_EXPERIMENTS.get((session, run), "#888888")

    @staticmethod
    def get_subject_color(subject: Animals, fov) -> str:
        return COLORS_SUBJECTS.get(f"{subject.value}_fov{fov}", "#888888")

    @staticmethod
    def get_colors_genotype_age(genotype: str, age: str) -> str:
        return COLORS_GROUPS.get((genotype, age), "#888888")

    @staticmethod
    def get_group_palette() -> dict[str, str]:
        """Return ``{group_label: hex}`` for all four groups in display order."""
        return {
            f"{g}_{a}": COLORS_GROUPS[(g, a)]
            for g, a in [("WT", "young"), ("WT", "old"), ("5xFAD", "young"), ("5xFAD", "old")]
        }

    # ------------------------------------------------------------------
    # Recording discovery (cached)
    # ------------------------------------------------------------------

    def _subject_folder(self, subject: Animals) -> str:
        info = self.subjects[subject]
        return f"{info['genotype']}_{info['age']}_{subject.value}"

    def get_available_recordings(self, subject: Animals) -> dict:
        """Discover recordings by scanning the filesystem (cached).

        Returns:
            dict mapping FOV (int or str like '1s2') -> {session: [runs]}.
        """
        if subject in self._recordings_cache:
            return self._recordings_cache[subject]

        mouse_dir = self.data_path / self._subject_folder(subject)
        recordings: dict = {}

        if not mouse_dir.exists():
            print(f"Directory not found: {mouse_dir}")
            self._recordings_cache[subject] = recordings
            return recordings

        for path in mouse_dir.glob(f"{subject.value}_fov*_*_spikes.parquet"):
            m = _FILENAME_RE.match(path.name)
            if m is None:
                continue
            base_fov = int(m.group("fov"))
            session = m.group("session")
            run = m.group("run")

            # Sessions with s2/s3 suffix -> virtual FOV
            if "s2" in session or "s3" in session:
                suffix = "s2" if "s2" in session else "s3"
                fov_key = f"{base_fov}{suffix}"
            else:
                fov_key = base_fov

            recordings.setdefault(fov_key, {}).setdefault(session, set()).add(run)

        # Copy non-conflicting base-FOV sessions into virtual FOVs
        for fov in list(recordings.keys()):
            if not isinstance(fov, int):
                continue
            for suffix in ("s2", "s3"):
                vfov = f"{fov}{suffix}"
                if vfov not in recordings:
                    continue
                covered = {sess.replace(suffix, "") for sess in recordings[vfov] if suffix in sess}
                for sess, runs in recordings[fov].items():
                    if suffix not in sess and sess not in covered:
                        recordings[vfov].setdefault(sess, set()).update(runs)

        # Sets -> sorted lists
        for fov in recordings:
            for sess in recordings[fov]:
                recordings[fov][sess] = sorted(recordings[fov][sess])

        self._recordings_cache[subject] = recordings
        return recordings

    def _sort_experiments(self, pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
        """Sort (session, run) pairs by EXPERIMENT_ORDER."""
        order = {sr: i for i, sr in enumerate(EXPERIMENT_ORDER)}
        return sorted(pairs, key=lambda sr: order.get(sr, len(EXPERIMENT_ORDER)))

    def get_available_fovs(self, subject: Animals) -> list:
        """FOVs for *subject* (ints first, then virtual str FOVs)."""
        recs = self.get_available_recordings(subject)
        int_fovs = sorted(f for f in recs if isinstance(f, int))
        str_fovs = sorted(f for f in recs if isinstance(f, str))
        return int_fovs + str_fovs

    def get_available_sessions(self, subject: Animals, fov) -> list[str]:
        """Sessions for a given FOV, in canonical order."""
        recs = self.get_available_recordings(subject)
        sessions = recs.get(fov, {})
        pairs = [(s, r) for s, runs in sessions.items() for r in runs]
        ordered = self._sort_experiments(pairs)
        seen: set[str] = set()
        return [s for s, _ in ordered if s not in seen and not seen.add(s)]

    def get_available_runs(self, subject: Animals, fov, session: str) -> list[str]:
        """Runs for a given FOV + session, in canonical order."""
        recs = self.get_available_recordings(subject)
        runs = recs.get(fov, {}).get(session, [])
        ordered = self._sort_experiments([(session, r) for r in runs])
        return [r for _, r in ordered]

    # ------------------------------------------------------------------
    # Tuning curves (static, vectorised)
    # ------------------------------------------------------------------

    @staticmethod
    def get_tuning_curves(
        firing_rates: np.ndarray,
        phi: np.ndarray,
        n_points: int = 360,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bin firing rates by angular position.

        Args:
            firing_rates: (T, N) array.
            phi: (T,) angles in degrees. NaN values are excluded.
            n_points: number of angular bins.

        Returns:
            ring_neural: (n_points, N) mean firing rate per bin.
            phi_bins: (n_points,) bin centres in degrees.
        """
        # Exclude time points where phi is NaN
        valid = np.isfinite(phi)
        if not valid.all():
            phi = phi[valid]
            firing_rates = firing_rates[valid]

        dphi = 360.0 / n_points
        bin_idx = np.clip(np.floor((phi % 360) / dphi).astype(int), 0, n_points - 1)

        # Vectorised bincount per neuron
        n_neurons = firing_rates.shape[1]
        ring_neural = np.zeros((n_points, n_neurons))
        counts = np.bincount(bin_idx, minlength=n_points).astype(float)

        for j in range(n_neurons):
            ring_neural[:, j] = np.bincount(bin_idx, weights=firing_rates[:, j], minlength=n_points)

        mask = counts > 0
        ring_neural[mask] /= counts[mask, np.newaxis]
        ring_neural[~mask] = np.nan

        # Circular interpolation for empty bins
        for b in np.where(~mask)[0]:
            prev_b = (b - 1) % n_points
            next_b = (b + 1) % n_points
            ring_neural[b] = (ring_neural[prev_b] + ring_neural[next_b]) / 2

        phi_bins = np.arange(n_points) * dphi
        return ring_neural, phi_bins

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_fov_for_file(fov) -> int:
        """Strip virtual suffix to get the physical FOV number for filenames."""
        if isinstance(fov, str) and (fov.endswith("s2") or fov.endswith("s3")):
            return int(fov[:-2])
        return int(fov)

    def _build_filepath(self, subject: Animals, fov, session: str, run: str):
        """Build the full Parquet path for a recording."""
        folder = self._subject_folder(subject)
        file_fov = self._resolve_fov_for_file(fov)
        filename = f"{subject.value}_fov{file_fov}_{session}-{run}_spikes.parquet"
        return self.data_path / folder / filename

    def load_data(self, subject: Animals, fov, session: str, run: str) -> pd.DataFrame:
        """Load a spikes Parquet file for one recording."""
        return pd.read_parquet(self._build_filepath(subject, fov, session, run))

    def get_n_neurons(self, subject: Animals, fov, session: str, run: str) -> int:
        """Return the neuron count without loading the full file (reads Parquet schema only)."""
        import pyarrow.parquet as pq

        schema = pq.read_schema(self._build_filepath(subject, fov, session, run))
        return sum(1 for name in schema.names if name.isdigit())

    def get_duration(self, subject: Animals, fov, session: str, run: str) -> float:
        """Return the recording duration in seconds (reads only the time column)."""
        filepath = self._build_filepath(subject, fov, session, run)
        time = pd.read_parquet(filepath, columns=["glob_time"])["glob_time"]
        return float(time.iloc[-1] - time.iloc[0])

    # ------------------------------------------------------------------
    # Processing helpers (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_arrays(
        data: pd.DataFrame,
        only_moving: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Extract spike, phi, time arrays from a raw DataFrame."""
        cells_cols = [c for c in data.columns if c.isdigit()]

        if only_moving:
            mask = data["movement_status"] == "moving"
            spikes = data.loc[mask, cells_cols].values
            phi = data.loc[mask, "phi"].values
            time = data.loc[mask, "glob_time"].values
        else:
            spikes = data[cells_cols].values
            phi = data["phi"].values
            time = data["glob_time"].values

        if np.isnan(spikes).any():
            raise ValueError("Spikes contain NaN values")

        return spikes, phi, time, cells_cols

    @staticmethod
    def _temporal_bin(
        spikes: np.ndarray,
        phi: np.ndarray,
        time: np.ndarray,
        bins_compress: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sum-bin spikes and mean-bin phi/time by *bins_compress*."""
        if bins_compress <= 1:
            return spikes, phi, time
        n = spikes.shape[0] // bins_compress
        s = spikes[: n * bins_compress].reshape(n, bins_compress, -1).sum(axis=1)
        p = phi[: n * bins_compress].reshape(n, bins_compress).mean(axis=1)
        t = time[: n * bins_compress].reshape(n, bins_compress).mean(axis=1)
        return s, p, t

    def _resolve_global_cells(
        self, subject: Animals, fov, session: str, cells_cols: list[str],
    ) -> tuple[list[str], bool]:
        """Map local -> global cell indices, with fallback."""
        global_cells = self.from_local_to_global_cell_index(subject, fov, session, cells_cols)
        if global_cells is None:
            return cells_cols, False
        return global_cells, True

    # ------------------------------------------------------------------
    # Public processing pipelines
    # ------------------------------------------------------------------

    def load_spikes_binned(
        self,
        subject: Animals,
        fov,
        session: str,
        run: str,
        only_moving: bool = False,
        bins_compress: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[list[str], bool]]:
        """Load spikes, optionally filter, and temporally bin.

        Returns:
            spikes_binned (T', N), phi_binned (T',), time_binned (T',),
            (cell_ids, registered_across_days).
        """
        data = self.load_data(subject, fov, session, run)
        spikes, phi, time, cells_cols = self._extract_arrays(data, only_moving)
        spikes_b, phi_b, time_b = self._temporal_bin(spikes, phi, time, bins_compress)
        cell_ids, registered = self._resolve_global_cells(subject, fov, session, cells_cols)
        return spikes_b, phi_b, time_b, (cell_ids, registered)

    def load_all_data_from_spikes_binned_smoothed(
        self,
        subject: Animals,
        fov,
        session: str,
        run: str,
        only_moving: bool = False,
        bins_compress: int = 3,
        bins_smoothing: int = 3,
        bins_phi: int = 360,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[list[str], bool], np.ndarray, np.ndarray]:
        """Full pipeline: load -> filter -> bin -> smooth -> sqrt -> tuning curves.

        Returns:
            firing_rates (T', N), phi_binned (T',), time_binned (T',),
            (cell_ids, registered), tuning_curves (bins_phi, N), phi_bins (bins_phi,).
        """
        data = self.load_data(subject, fov, session, run)
        spikes, phi, time, cells_cols = self._extract_arrays(data, only_moving)
        spikes_b, phi_b, time_b = self._temporal_bin(spikes, phi, time, bins_compress)

        # Smooth + sqrt transform
        firing_rates = np.sqrt(gaussian_filter1d(spikes_b.astype(float), sigma=bins_smoothing, axis=0))

        # Tuning curves
        tuning_curves, phi_bins = self.get_tuning_curves(firing_rates, phi_b, n_points=bins_phi)

        # Global cell indices
        cell_ids, registered = self._resolve_global_cells(subject, fov, session, cells_cols)

        return firing_rates, phi_b, time_b, (cell_ids, registered), tuning_curves, phi_bins

    # ------------------------------------------------------------------
    # Cell index mapping
    # ------------------------------------------------------------------

    def from_local_to_global_cell_index(
        self,
        subject: Animals,
        fov,
        session: str,
        local_indexes: list[str],
    ) -> list[str] | None:
        """Map daily cell indices to global (cross-session) indices.

        Returns None if the reference file is missing or session not found.
        """
        folder = self._subject_folder(subject)
        ref_path = self.data_path / folder / f"{subject.value}_fov{fov}_global_index_ref.parquet"

        if not ref_path.exists():
            return None

        df_ref = pd.read_parquet(ref_path)

        if session not in df_ref.columns:
            return None

        global_indexes: list[str] = []
        for li in local_indexes:
            matches = np.where(df_ref[session] == int(li))[0]
            if len(matches) == 0:
                raise IndexError(
                    f"Local cell {li} not found in global index for "
                    f"{subject.value}_fov{fov}_{session}"
                )
            global_indexes.append(str(df_ref.loc[matches[0], "global_index"]))
        return global_indexes
