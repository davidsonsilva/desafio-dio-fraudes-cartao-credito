from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    random_state: int = 42
    test_size: float = 0.25
    minimum_recall: float = 0.80
    artifact_path: Path = ROOT / "models" / "fraud_detector.joblib"
    metrics_path: Path = ROOT / "reports" / "metrics.json"
    feedback_path: Path = ROOT / "feedback" / "confirmed.csv"


settings = Settings()
