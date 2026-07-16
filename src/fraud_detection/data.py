from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {"Time", "Amount", "Class", *(f"V{i}" for i in range(1, 29))}


def load_dataset(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    if frame.isna().any().any():
        raise ValueError("O dataset contém valores ausentes.")
    return frame


def synthetic_dataset(rows: int = 5000, random_state: int = 42) -> pd.DataFrame:
    """Gera dados compatíveis para smoke tests; não representa transações reais."""
    rng = np.random.default_rng(random_state)
    frauds = max(20, int(rows * 0.01))
    y = np.zeros(rows, dtype=int)
    y[rng.choice(rows, frauds, replace=False)] = 1
    data = {"Time": rng.uniform(0, 172800, rows)}
    for index in range(1, 29):
        shift = y * (0.7 if index <= 8 else 0.15)
        data[f"V{index}"] = rng.normal(shift, 1, rows)
    data["Amount"] = rng.lognormal(3.2 + y * 0.7, 1.0, rows)
    data["Class"] = y
    return pd.DataFrame(data)
