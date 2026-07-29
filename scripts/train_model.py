"""CLI wrapper: python -m scripts.train_model [--csv PATH] [--out PATH]"""
import argparse
from pathlib import Path

from src.config import CONFIG
from src.model.train import train_from_csv

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "generated" / "training_data.csv"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out", type=Path, default=CONFIG.model_path)
    args = parser.parse_args()

    if not args.csv.exists():
        raise SystemExit(
            f"{args.csv} not found. Run `python -m scripts.generate_training_data` first."
        )
    train_from_csv(args.csv, args.out)

if __name__ == "__main__":
    main()
