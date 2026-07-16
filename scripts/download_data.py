import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
target = ROOT / "data" / "raw"
target.mkdir(parents=True, exist_ok=True)
if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")) and not (Path.home() / ".kaggle" / "kaggle.json").exists():
    raise SystemExit("Configure KAGGLE_USERNAME/KAGGLE_KEY ou ~/.kaggle/kaggle.json.")
subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", "mlg-ulb/creditcardfraud", "-p", str(target), "--unzip"], check=True)
print(target / "creditcard.csv")
