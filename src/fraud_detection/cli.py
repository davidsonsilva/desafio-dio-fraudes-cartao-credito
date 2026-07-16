import argparse

import pandas as pd

from .config import settings
from .data import load_dataset, synthetic_dataset
from .reporting import generate_pdf
from .training import train_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de detecção de fraudes")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", default="data/raw/creditcard.csv")
    train.add_argument("--demo", action="store_true")
    train.add_argument("--rows", type=int, default=5000)
    commands.add_parser("report")
    args = parser.parse_args()
    if args.command == "train":
        frame = synthetic_dataset(args.rows) if args.demo else load_dataset(args.data)
        if settings.feedback_path.exists() and not args.demo:
            confirmed = pd.read_csv(settings.feedback_path)
            frame = pd.concat([frame, confirmed], ignore_index=True)
            print(f"Autoaprendizado: {len(confirmed)} exemplos confirmados incorporados.")
        artifact = train_all(frame)
        print(f"Modelo: {artifact['model_name']} | features: {artifact['feature_count']} | artefato: {settings.artifact_path}")
    else:
        print(generate_pdf())


if __name__ == "__main__":
    main()
