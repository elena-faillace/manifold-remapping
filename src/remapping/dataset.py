"""MiceDataset — central data interface for mice circular-arena recordings."""

from enum import Enum
import os
import glob
import functools

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


class MiceDataType(Enum):
    """Data modalities stored on disk."""
    SPIKES = "spikes"
    TRACES = "traces"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MiceDataset:
    """Dataset class for mice circular-arena calcium-imaging data.

    All paths are resolved from ``DATA_ROOT`` (set via ``.env``).
    """

    def __init__(self):
        self.data_path = DATA_ROOT
        self.sampling_rate = 30.9  # Hz

        self.subjects = self._get_subjects_config()

        # Cache for get_available_recordings — filled lazily
        self._recordings_cache: dict[Animals, dict] = {}

        # -- colour maps -------------------------------------------------------
        self.colors_experiments = {
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

        self.colors_subjects = {
            "m62_fov1": "#f7a3ba", "m62_fov2": "#f992ad",
            "m66_fov1": "#fbbcee", "m66_fov2": "#f6a5e4",
            "m70_fov1": "#fab4c8", "m70_fov2": "#f59bb3",
            "m116_fov1": "#f78ecf", "m116_fov2": "#e776c2",
            "m117_fov1": "#b88ae6", "m117_fov2": "#a178db",
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

        self.colors_groups = {
            ("WT", "young"): "#f6a6c1",
            ("WT", "old"): "#c2185b",
            ("5xFAD", "young"): "#80cbc4",
            ("5xFAD", "old"): "#00796b",
        }

        self.order_experiments = [
            ("fam1fam2", "fam1"), ("fam1fam2", "fam2"),
            ("fam1fam2fam1", "fam1"), ("fam1fam2fam1", "fam2"), ("fam1fam2fam1", "fam1r2"),
            ("fam1fam2s2", "fam1"), ("fam1fam2s2", "fam2"),
            ("fam1fam2s3", "fam1"), ("fam1fam2s3", "fam2"),
            ("fam1nov", "fam1"), ("fam1nov", "nov"),
            ("fam1novfam1", "fam1"), ("fam1novfam1", "nov"), ("fam1novfam1", "fam1r2"),
            ("fam1fam1rev", "fam1"), ("fam1fam1rev", "fam1rev"),
            ("fam1fam1revfam1", "fam1"), ("fam1fam1revfam1", "fam1rev"), ("fam1fam1revfam1", "fam1r2"),
        ]

    # ------------------------------------------------------------------
    # Subject metadata
    # ------------------------------------------------------------------

    @staticmethod
    def _get_subjects_config() -> dict:
        return {
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

    def get_all_subjects(self) -> list[Animals]:
        """All 23 mouse subjects."""
        return list(self.subjects.keys())

    def get_subjects_by_genotype(self, genotype: str) -> list[Animals]:
        """Filter by genotype ('WT' or '5xFAD')."""
        return [s for s, c in self.subjects.items() if c["genotype"] == genotype]

    def get_subjects_by_age(self, age: str) -> list[Animals]:
        """Filter by age ('young' or 'old')."""
        return [s for s, c in self.subjects.items() if c["age"] == age]

    def get_subjects_by_group(self, genotype: str, age: str) -> list[Animals]:
        """Filter by genotype **and** age."""
        return [
            s for s, c in self.subjects.items()
            if c["genotype"] == genotype and c["age"] == age
        ]

    def get_subject_info(self, subject: Animals) -> dict:
        """Return ``{'genotype': ..., 'age': ...}`` for *subject*."""
        return self.subjects[subject]

    def get_genotypes(self) -> list[str]:
        """Unique genotypes (sorted)."""
        return sorted({c["genotype"] for c in self.subjects.values()})

    def get_ages(self) -> list[str]:
        """Unique age groups (sorted)."""
        return sorted({c["age"] for c in self.subjects.values()})

    # ------------------------------------------------------------------
    # Recording discovery (cached)
    # ------------------------------------------------------------------

    def get_available_recordings(self, subject: Animals) -> dict:
        """Discover available recordings by scanning the filesystem.

        Results are cached after the first call per subject.

        Returns:
            dict mapping FOV (int or str like '1s2') → {session: [runs]}.
        """
        if subject in self._recordings_cache:
            return self._recordings_cache[subject]

        info = self.subjects[subject]
        folder = f"{info['genotype']}_{info['age']}_{subject.value}"
        mouse_dir = self.data_path / folder

        recordings: dict = {}

        if not mouse_dir.exists():
            print(f"Directory not found: {mouse_dir}")
            self._recordings_cache[subject] = recordings
            return recordings

        pattern = f"{subject.value}_fov*_fam*_*.parquet"
        files = glob.glob(str(mouse_dir / pattern))

        for file in files:
            filename = os.path.basename(file)
            parts = filename.replace(f"{subject.value}_", "").split("_")
            if len(parts) < 3:
                continue
            fov_part = parts[0]  # e.g. "fov2"
            session_run_part = parts[1]  # e.g. "fam1fam1rev-fam1"

            base_fov = int(fov_part[3])
            session, run = session_run_part.split("-", 1)

            if "s2" in session or "s3" in session:
                suffix = "s2" if "s2" in session else "s3"
                virtual_fov = f"{base_fov}{suffix}"
                recordings.setdefault(virtual_fov, {}).setdefault(session, set()).add(run)
            else:
                recordings.setdefault(base_fov, {}).setdefault(session, set()).add(run)

        # Populate virtual FOVs with non-conflicting regular sessions
        for fov in list(recordings.keys()):
            if not isinstance(fov, int):
                continue
            for suffix in ("s2", "s3"):
                virtual_fov = f"{fov}{suffix}"
                if virtual_fov not in recordings:
                    continue
                sx_bases = set()
                for sess in recordings[virtual_fov]:
                    if suffix in sess:
                        sx_bases.add(sess.replace(suffix, ""))
                for sess, runs in recordings[fov].items():
                    if suffix not in sess and sess not in sx_bases:
                        recordings[virtual_fov].setdefault(sess, set()).update(runs)

        # Convert sets → sorted lists
        for fov in recordings:
            for sess in recordings[fov]:
                recordings[fov][sess] = sorted(recordings[fov][sess])

        self._recordings_cache[subject] = recordings
        return recordings

    def _sort_session_runs(self, session_run_pairs):
        """Sort (session, run) pairs by canonical experiment order."""
        def _key(sr):
            try:
                return self.order_experiments.index(sr)
            except ValueError:
                return len(self.order_experiments)
        return sorted(session_run_pairs, key=_key)

    def get_available_fovs(self, subject: Animals) -> list:
        """All FOVs for *subject* (ints first, then virtual str FOVs)."""
        recs = self.get_available_recordings(subject)
        int_fovs = sorted(f for f in recs if isinstance(f, int))
        str_fovs = sorted(f for f in recs if isinstance(f, str))
        return int_fovs + str_fovs

    def get_available_sessions(self, subject: Animals, fov) -> list[str]:
        """Sessions for a given FOV, in canonical order."""
        recs = self.get_available_recordings(subject)
        sessions = recs.get(fov, {})
        pairs = [(s, r) for s, runs in sessions.items() for r in runs]
        ordered = self._sort_session_runs(pairs)
        seen: set[str] = set()
        out: list[str] = []
        for s, _ in ordered:
            if s not in seen:
                out.append(s)
                seen.add(s)
        return out

    def get_available_runs(self, subject: Animals, fov, session: str) -> list[str]:
        """Runs for a given FOV + session, in canonical order."""
        recs = self.get_available_recordings(subject)
        runs = recs.get(fov, {}).get(session, [])
        pairs = [(session, r) for r in runs]
        ordered = self._sort_session_runs(pairs)
        return [r for _, r in ordered]

    # ------------------------------------------------------------------
    # Colours
    # ------------------------------------------------------------------

    def get_experiment_color(self, session: str, run: str) -> str:
        return self.colors_experiments.get((session, run), "#888888")

    def get_subject_color(self, subject: Animals, fov) -> str:
        return self.colors_subjects.get(f"{subject.value}_fov{fov}", "#888888")

    def get_colors_genotype_age(self, genotype: str, age: str) -> str:
        return self.colors_groups.get((genotype, age), "#888888")

    # ------------------------------------------------------------------
    # Tuning curves
    # ------------------------------------------------------------------

    @staticmethod
    def get_tuning_curves(
        firing_rates: np.ndarray,
        phi: np.ndarray,
        n_points: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bin firing rates by head-direction angle to get tuning curves.

        Args:
            firing_rates: (T, N) array.
            phi: (T,) array of angles in degrees.
            n_points: number of angular bins.

        Returns:
            ring_neural: (n_points, N) mean firing rate per bin.
            phi_bins: (n_points,) bin centres in degrees.
        """
        phi_mod = phi % 360
        dphi = 360 / n_points
        bin_idx = np.floor(phi_mod / dphi).astype(int)
        bin_idx = np.clip(bin_idx, 0, n_points - 1)

        ring_neural = np.zeros((n_points, firing_rates.shape[1]))
        counts = np.zeros(n_points, dtype=int)
        for i in range(len(phi_mod)):
            ring_neural[bin_idx[i]] += firing_rates[i]
            counts[bin_idx[i]] += 1
        for b in range(n_points):
            if counts[b] > 0:
                ring_neural[b] /= counts[b]

        phi_bins = np.arange(n_points) * dphi

        # Fill any remaining NaN bins by circular interpolation
        if np.isnan(ring_neural).any():
            for b in range(ring_neural.shape[0]):
                if np.isnan(ring_neural[b]).any():
                    prev_b = (b - 1) % n_points
                    next_b = (b + 1) % n_points
                    ring_neural[b] = (ring_neural[prev_b] + ring_neural[next_b]) / 2

        return ring_neural, phi_bins

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _subject_folder(self, subject: Animals) -> str:
        info = self.subjects[subject]
        return f"{info['genotype']}_{info['age']}_{subject.value}"

    def load_data(
        self,
        subject: Animals,
        fov,
        session: str,
        run: str,
        data_type: MiceDataType = MiceDataType.SPIKES,
    ) -> pd.DataFrame:
        """Load a raw Parquet file for one recording.

        Args:
            subject: e.g. ``Animals.M62``
            fov: int or str (e.g. 2 or '2s2')
            session: e.g. ``'fam1fam1rev'``
            run: e.g. ``'fam1'``
            data_type: ``MiceDataType.SPIKES`` (default) or ``TRACES``

        Returns:
            pandas DataFrame.
        """
        # Handle virtual s2/s3 FOVs
        if isinstance(fov, str) and (fov.endswith("s2") or fov.endswith("s3")):
            file_fov = int(fov[:-2])
            file_session = session
        else:
            file_fov = fov
            file_session = session

        folder = self._subject_folder(subject)
        filename = f"{subject.value}_fov{file_fov}_{file_session}-{run}_{data_type.value}.parquet"
        filepath = self.data_path / folder / filename
        return pd.read_parquet(filepath)

    def load_spikes_binned(
        self,
        subject: Animals,
        fov,
        session: str,
        run: str,
        only_moving: bool = False,
        bins_compress: int = 3,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple]:
        """Load spikes, bin temporally, and map to global cell indices.

        Args:
            subject: mouse enum.
            fov: field of view.
            session: session name.
            run: run name.
            only_moving: keep only ``movement_status == 'moving'`` frames.
            bins_compress: temporal binning factor (default 3 → ~10.3 Hz).

        Returns:
            spikes_binned: (T', N) summed spike counts.
            phi_binned: (T',) mean angle per bin.
            time_binned: (T',) mean time per bin.
            (global_cell_ids, registered_across_days): cell index info.
        """
        data = self.load_data(subject, fov, session, run, MiceDataType.SPIKES)
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

        if pd.isnull(spikes).any():
            raise ValueError(
                f"Spikes contain NaNs: {subject.value} fov{fov} {session}-{run}"
            )

        n_bins = spikes.shape[0] // bins_compress
        spikes_binned = np.sum(
            spikes[: n_bins * bins_compress].reshape(-1, bins_compress, spikes.shape[1]),
            axis=1,
        )
        phi_binned = np.mean(
            phi[: n_bins * bins_compress].reshape(-1, bins_compress), axis=1
        )
        time_binned = np.mean(
            time[: n_bins * bins_compress].reshape(-1, bins_compress), axis=1
        )

        final_cells = self.from_local_to_global_cell_index(subject, fov, session, cells_cols)
        if final_cells is None:
            print(f"Could not map global indices for {subject.value} fov{fov} {session}-{run}")
            print("\tReturning local indexes instead.")
            final_cells = cells_cols
            registered_across_days = False
        else:
            registered_across_days = True

        return spikes_binned, phi_binned, time_binned, (final_cells, registered_across_days)

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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, np.ndarray, np.ndarray]:
        """Full pipeline: bin → smooth → sqrt → tuning curves.

        Args:
            subject: mouse enum.
            fov: field of view.
            session: session name.
            run: run name.
            only_moving: keep only moving frames.
            bins_compress: temporal binning factor (default 3).
            bins_smoothing: Gaussian sigma in bins (default 3 ≈ 0.29 s).
            bins_phi: angular bins for tuning curves (default 360).

        Returns:
            firing_rates: (T', N) sqrt-smoothed rates.
            phi_binned: (T',) angles.
            time_binned: (T',) timestamps.
            (global_cell_ids, registered_across_days): cell info.
            tuning_curves: (bins_phi, N).
            phi_bins: (bins_phi,) bin centres in degrees.
        """
        data = self.load_data(subject, fov, session, run, MiceDataType.SPIKES)
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

        if pd.isnull(spikes).any():
            raise ValueError(
                f"Spikes contain NaNs: {subject.value} fov{fov} {session}-{run}"
            )

        # Temporal binning
        n_bins = spikes.shape[0] // bins_compress
        spikes_binned = np.sum(
            spikes[: n_bins * bins_compress].reshape(-1, bins_compress, spikes.shape[1]),
            axis=1,
        )
        phi_binned = np.mean(
            phi[: n_bins * bins_compress].reshape(-1, bins_compress), axis=1
        )
        time_binned = np.mean(
            time[: n_bins * bins_compress].reshape(-1, bins_compress), axis=1
        )

        # Gaussian smooth + sqrt
        spikes_smoothed = gaussian_filter1d(spikes_binned, sigma=bins_smoothing, axis=0)
        firing_rates = np.sqrt(spikes_smoothed)

        # Tuning curves
        tuning_curves, phi_bins = self.get_tuning_curves(firing_rates, phi_binned, n_points=bins_phi)

        # Global cell indices
        final_cells = self.from_local_to_global_cell_index(subject, fov, session, cells_cols)
        if final_cells is None:
            print(f"Could not map global indices for {subject.value} fov{fov} {session}-{run}")
            print("\tReturning local indexes instead.")
            final_cells = cells_cols
            registered_across_days = False
        else:
            registered_across_days = True

        return (
            firing_rates,
            phi_binned,
            time_binned,
            (final_cells, registered_across_days),
            tuning_curves,
            phi_bins,
        )

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

        Returns None if the reference file is missing.
        """
        folder = self._subject_folder(subject)
        ref_path = self.data_path / folder / f"{subject.value}_fov{fov}_global_index_ref.parquet"

        if not ref_path.exists():
            print(f"Global index file not found: {ref_path}")
            return None

        df_ref = pd.read_parquet(ref_path)

        if session not in df_ref.columns:
            print(f"Session '{session}' not in global index file for {subject.value} fov{fov}")
            return None

        global_indexes: list[str] = []
        for li in local_indexes:
            matches = np.where(df_ref[session] == int(li))[0]
            if len(matches) == 0:
                raise IndexError(
                    f"Local cell {li} not found in global index for {subject.value}_fov{fov}_{session}"
                )
            global_indexes.append(str(df_ref.loc[matches[0], "global_index"]))
        return global_indexes
