from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import plot_training_curves


def main():
    output_root = Path("outputs")
    for exp_dir in sorted(output_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        logs_dir = exp_dir / "logs"
        figures_dir = exp_dir / "figures"
        if not logs_dir.exists():
            continue
        csv_files = sorted(logs_dir.glob("*.csv"))
        for csv_path in csv_files:
            experiment_name = csv_path.stem
            figure_path = plot_training_curves(
                csv_path=str(csv_path),
                figure_dir=str(figures_dir),
                experiment_name=experiment_name,
            )
            print(f"Generated: {figure_path}")


if __name__ == "__main__":
    main()
