import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import settings


def generate_pdf(metrics_path: Path = settings.metrics_path, output: Path | None = None) -> Path:
    output = output or settings.metrics_path.with_name("fraud_detection_report.pdf")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph("Relatório de Detecção de Fraudes", styles["Title"]), Spacer(1, 0.4 * cm)]
    dataset = payload["dataset"]
    story.append(Paragraph(f"Dataset: {dataset['rows']:,} transações, {dataset['frauds']:,} fraudes ({dataset['fraud_rate']:.4%}). Modelo selecionado: {payload['model_name']}. Features: {payload['feature_count']}.", styles["BodyText"]))
    story.append(Spacer(1, 0.5 * cm))
    rows = [["Modelo", "PR-AUC", "ROC-AUC", "Precisão", "Recall", "F1", "Limiar"]]
    for item in sorted(payload["metrics"], key=lambda value: value["pr_auc"], reverse=True):
        rows.append([item["name"], *(f"{item[key]:.4f}" for key in ("pr_auc", "roc_auc", "precision", "recall", "f1", "threshold"))])
    table = Table(rows, repeatRows=1, colWidths=[4.5 * cm] + [1.7 * cm] * 6)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#152238")), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), 0.3, colors.grey), ("FONTSIZE", (0, 0), (-1, -1), 7), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2f7")])]))
    story.extend([table, Spacer(1, 0.5 * cm), Paragraph("A seleção prioriza PR-AUC. O limiar é calibrado na validação para maximizar F1 respeitando o recall mínimo quando possível. Resultados sintéticos servem apenas para validação técnica.", styles["BodyText"])])
    SimpleDocTemplate(str(output), pagesize=A4, rightMargin=1.2 * cm, leftMargin=1.2 * cm).build(story)
    return output
