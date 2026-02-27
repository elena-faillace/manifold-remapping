"""One-time script: convert all CSV data files to Parquet (Snappy compression).

Usage:
    python scripts/convert_csv_to_parquet.py

Reads from the OLD data root (elena_anns CSV folder) and writes to the NEW
data root (manifold-remapping/data parquet folder) preserving the same
directory structure.
"""

from pathlib import Path
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────
SRC_ROOT = Path(
    "/Users/elenafaillace/Library/CloudStorage/"
    "OneDrive-ImperialCollegeLondon/remapping_collab/datasets/elena_anns"
)
DST_ROOT = Path(
    "/Users/elenafaillace/Library/CloudStorage/"
    "OneDrive-ImperialCollegeLondon/manifold-remapping/data"
)


def convert_file(src: Path, dst: Path) -> tuple[int, int]:
    """Read a CSV, write as Parquet. Returns (csv_bytes, parquet_bytes)."""
    df = pd.read_csv(src, low_memory=False)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, engine="pyarrow", compression="snappy", index=False)
    return src.stat().st_size, dst.stat().st_size


def main() -> None:
    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"Source not found: {SRC_ROOT}")

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    total_csv = 0
    total_pq = 0
    file_count = 0

    for mouse_dir in sorted(SRC_ROOT.iterdir()):
        if not mouse_dir.is_dir():
            continue
        print(f"\n{'='*60}")
        print(f"  {mouse_dir.name}")
        print(f"{'='*60}")

        for csv_file in sorted(mouse_dir.glob("*.csv")):
            pq_name = csv_file.stem + ".parquet"
            dst_file = DST_ROOT / mouse_dir.name / pq_name

            csv_bytes, pq_bytes = convert_file(csv_file, dst_file)
            ratio = pq_bytes / csv_bytes * 100

            total_csv += csv_bytes
            total_pq += pq_bytes
            file_count += 1

            csv_mb = csv_bytes / 1e6
            pq_mb = pq_bytes / 1e6
            print(f"  {csv_file.name:60s}  {csv_mb:7.1f} MB -> {pq_mb:7.1f} MB  ({ratio:4.1f}%)")

    print(f"\n{'='*60}")
    print(f"  TOTAL: {file_count} files")
    print(f"  CSV:     {total_csv / 1e9:.2f} GB")
    print(f"  Parquet: {total_pq / 1e9:.2f} GB")
    print(f"  Savings: {(total_csv - total_pq) / 1e9:.2f} GB ({(1 - total_pq / total_csv) * 100:.1f}%)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

