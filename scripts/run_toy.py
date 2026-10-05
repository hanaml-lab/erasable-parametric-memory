"""Run the CPU toy end to end (python scripts/run_toy.py [--config configs/toy.yaml] [--seed 1]); same as python -m slate.demo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from slate.demo import main  # noqa: E402

if __name__ == "__main__":
    main()
