import csv

from .config import settings


def append_feedback(transaction: dict, label: int) -> None:
    path = settings.feedback_path
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {**transaction, "Class": int(label)}
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
