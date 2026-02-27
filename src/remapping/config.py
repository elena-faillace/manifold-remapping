"""Load .env and expose DATA_ROOT / FIGURES_ROOT as Path objects."""

from pathlib import Path
from dotenv import load_dotenv
import os

# Load .env from repo root (walks parents automatically)
load_dotenv()

_data = os.getenv("DATA_ROOT")
_figs = os.getenv("FIGURES_ROOT")

if _data is None:
    raise FileNotFoundError(
        "DATA_ROOT not set. Copy .env.example → .env and fill in your paths."
    )

DATA_ROOT: Path = Path(_data)
if not DATA_ROOT.exists():
    raise FileNotFoundError(f"DATA_ROOT does not exist: {DATA_ROOT}")

if _figs is None:
    raise FileNotFoundError(
        "FIGURES_ROOT not set. Copy .env.example → .env and fill in your paths."
    )

FIGURES_ROOT: Path = Path(_figs)
FIGURES_ROOT.mkdir(parents=True, exist_ok=True)
