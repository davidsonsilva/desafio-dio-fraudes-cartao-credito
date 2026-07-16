"""Gera a documentação técnica versionável do projeto em PDF."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "pdf" / "relatorio-tecnico-fraudes-cartao.pdf"


def header_footer(canvas, document):
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(1.6 * cm, 1.45 * cm, A4[0] - 1.6 * cm, 1.45 * cm)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(1.6 * cm, 1.05 * cm, "Deteccao de Fraudes em Cartoes de Credito")
    canvas.drawRightString(A4[0] - 1.6 * cm, 1.05 * cm, f"Pagina {document.page}")
    canvas.restoreState()


def bullets(items, styles):
    return [Paragraph(f"- {item}", styles["BodyText"]) for item in items]


def build_pdf(output: Path = OUTPUT) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            "CoverTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=25,
            leading=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0F172A"),
            spaceAfter=18,
        )
    )
    styles.add(
        ParagraphStyle(
            "Section",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#0F4C81"),
            spaceBefore=12,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            "Subsection",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#1E3A5F"),
            spaceBefore=8,
            spaceAfter=5,
        )
    )
    styles["BodyText"].fontName = "Helvetica"
    styles["BodyText"].fontSize = 9.5
    styles["BodyText"].leading = 14
    styles["BodyText"].spaceAfter = 5

    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.7 * cm,
        bottomMargin=1.8 * cm,
        title="Relatorio Tecnico - Deteccao de Fraudes em Cartoes de Credito",
        author="Davidson Silva",
    )
    story = [
        Spacer(1, 2.2 * cm),
        Paragraph("Deteccao de Anomalias e Fraudes em Transacoes de Cartao de Credito", styles["CoverTitle"]),
        Paragraph(
            "Relatorio tecnico do sistema completo de machine learning, deep learning, API REST e dashboard interativo",
            ParagraphStyle("CoverSubtitle", parent=styles["BodyText"], alignment=TA_CENTER, fontSize=13, leading=18, textColor=colors.HexColor("#475569")),
        ),
        Spacer(1, 1.3 * cm),
        Table(
            [
                ["Dataset", "284.807 transacoes / 492 fraudes"],
                ["Modelos", "10 abordagens supervisionadas e de anomalia"],
                ["Features", "Mais de 100 variaveis apos engenharia"],
                ["Interfaces", "CLI, FastAPI, Streamlit e PDF"],
                ["Repositorio", "github.com/davidsonsilva/desafio-dio-fraudes-cartao-credito"],
            ],
            colWidths=[3.3 * cm, 11.5 * cm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#0F4C81")),
                    ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
                    ("BACKGROUND", (1, 0), (1, -1), colors.HexColor("#F1F5F9")),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        ),
        Spacer(1, 2 * cm),
        Paragraph("Davidson Silva - 2026", ParagraphStyle("Author", parent=styles["BodyText"], alignment=TA_CENTER, textColor=colors.HexColor("#64748B"))),
        PageBreak(),
        Paragraph("1. Contexto e objetivo", styles["Section"]),
        Paragraph(
            "O projeto utiliza o dataset publico Credit Card Fraud Detection, com 284.807 transacoes e somente 492 fraudes (aproximadamente 0,173%). O objetivo e construir um fluxo reproduzivel para detectar transacoes suspeitas, comparar diferentes familias de modelos e disponibilizar a inferencia por aplicacao web e API.",
            styles["BodyText"],
        ),
        Paragraph(
            "Como a classe positiva e extremamente rara, acuracia nao e usada como criterio principal. A avaliacao prioriza PR-AUC, recall, precisao, F1 e matriz de confusao.",
            styles["BodyText"],
        ),
        Paragraph("2. Arquitetura da solucao", styles["Section"]),
        *bullets(
            [
                "Entrada pelo CSV real do Kaggle ou por dados sinteticos compativeis para smoke tests.",
                "Validacao do contrato com Time, Amount, Class e V1 ate V28.",
                "Divisao estratificada entre treino e teste antes das transformacoes aprendidas.",
                "Engenharia deterministica de mais de 100 features e escalonamento robusto.",
                "Treinamento e comparacao de dez modelos com estrategias proprias de balanceamento.",
                "Selecao do modelo por PR-AUC e calibracao dinamica do threshold.",
                "Persistencia de um artefato unico consumido pela CLI, FastAPI e Streamlit.",
                "Registro auditavel de feedback confirmado para o proximo retreinamento.",
            ],
            styles,
        ),
        Paragraph("3. Engenharia de features", styles["Section"]),
        Paragraph(
            "As variaveis originais sao enriquecidas com representacao ciclica do horario, indicadores temporais, transformacoes de Amount, interacoes entre variaveis PCA, modulos, quadrados e estatisticas agregadas. RobustScaler reduz a influencia de outliers sem aprender informacao do conjunto de teste.",
            styles["BodyText"],
        ),
        Paragraph("4. Modelos avaliados", styles["Section"]),
    ]
    models = [
        ["Modelo", "Tipo", "Tratamento do desbalanceamento"],
        ["Regressao Logistica", "Supervisionado", "SMOTE e peso balanceado"],
        ["Random Forest", "Supervisionado", "SMOTE e peso balanceado"],
        ["Extra Trees", "Supervisionado", "SMOTE e peso balanceado"],
        ["HistGradientBoosting", "Supervisionado", "SMOTE"],
        ["XGBoost", "Supervisionado", "scale_pos_weight"],
        ["TensorFlow / Keras", "Deep learning", "class weights e early stopping"],
        ["Isolation Forest", "Anomalia", "Somente exemplos normais"],
        ["Local Outlier Factor", "Novidade", "novelty=True; somente normais"],
        ["One-Class SVM", "Novidade", "Somente exemplos normais"],
        ["PCA Reconstruction", "Anomalia", "Erro de reconstrucao"],
    ]
    story.extend(
        [
            Table(
                models,
                repeatRows=1,
                colWidths=[4.4 * cm, 3.2 * cm, 8.1 * cm],
                style=TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F4C81")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F1F5F9")]),
                        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("FONTSIZE", (0, 0), (-1, -1), 8),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                ),
            ),
            Paragraph("5. Avaliacao e threshold", styles["Section"]),
            Paragraph(
                "Cada estimador gera um score continuo. A curva precision-recall e usada para localizar o threshold que maximiza F1 entre os pontos que atendem ao recall minimo configurado. Se a restricao nao puder ser atingida, o melhor F1 disponivel e utilizado. O vencedor e selecionado por PR-AUC, com F1 como desempate.",
                styles["BodyText"],
            ),
            Paragraph("6. Interfaces e artefatos", styles["Section"]),
            *bullets(
                [
                    "FastAPI: health check, informacoes do modelo, previsao individual, lote e feedback.",
                    "Streamlit: indicadores, upload de CSV, classificacao, ranking de risco e exportacao.",
                    "CLI: treinamento com dataset real ou sintetico e geracao do relatorio dinamico.",
                    "Joblib: modelo, transformacoes, threshold, colunas, versao e metricas.",
                    "ReportLab: relatorios PDF tecnicos e de resultados.",
                ],
                styles,
            ),
            Paragraph("7. Processo de execucao", styles["Section"]),
            KeepTogether(
                [
                    Paragraph("1. Criar ambiente Python 3.11 a 3.13 e instalar o pacote com pip install -e \".[dev]\".", styles["BodyText"]),
                    Paragraph("2. Baixar creditcard.csv pelo script Kaggle ou usar o modo --demo.", styles["BodyText"]),
                    Paragraph("3. Executar python -m fraud_detection.cli train.", styles["BodyText"]),
                    Paragraph("4. Iniciar o dashboard com streamlit run app/streamlit_app.py.", styles["BodyText"]),
                    Paragraph("5. Iniciar a API com uvicorn fraud_detection.api:app --reload.", styles["BodyText"]),
                    Paragraph("6. Gerar o relatorio de metricas com python -m fraud_detection.cli report.", styles["BodyText"]),
                ]
            ),
            Paragraph("8. Qualidade, seguranca e limitacoes", styles["Section"]),
            *bullets(
                [
                    "Testes cobrem contrato dos dados, features, threshold, API e catalogo de modelos.",
                    "Dataset, credenciais, modelos e feedback gerados nao sao versionados.",
                    "Joblib deve ser carregado somente de uma fonte confiavel.",
                    "Resultados sinteticos validam integracao, nao desempenho real.",
                    "Producao requer validacao temporal, monitoramento de drift e revisao humana.",
                    "As variaveis PCA anonimizadas limitam explicacoes semanticas detalhadas.",
                ],
                styles,
            ),
            PageBreak(),
            Paragraph("9. Referencias oficiais", styles["Section"]),
        ]
    )
    references = [
        ("Dataset Kaggle", "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"),
        ("scikit-learn", "https://scikit-learn.org/stable/"),
        ("Novelty and Outlier Detection", "https://scikit-learn.org/stable/modules/outlier_detection.html"),
        ("XGBoost", "https://xgboost.readthedocs.io/en/stable/"),
        ("TensorFlow", "https://www.tensorflow.org/?hl=pt-br"),
        ("TensorFlow - Imbalanced Data", "https://www.tensorflow.org/tutorials/structured_data/imbalanced_data"),
        ("Streamlit", "https://docs.streamlit.io/"),
        ("FastAPI", "https://fastapi.tiangolo.com/"),
        ("imbalanced-learn SMOTE", "https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html"),
    ]
    for label, url in references:
        story.append(Paragraph(f"<b>{label}</b><br/><link href='{url}' color='#0F4C81'>{url}</link>", styles["BodyText"]))
    story.extend(
        [
            Spacer(1, 0.5 * cm),
            Paragraph("10. Repositorio", styles["Section"]),
            Paragraph(
                "Codigo-fonte, README, testes e historico de desenvolvimento:<br/><link href='https://github.com/davidsonsilva/desafio-dio-fraudes-cartao-credito' color='#0F4C81'>https://github.com/davidsonsilva/desafio-dio-fraudes-cartao-credito</link>",
                styles["BodyText"],
            ),
        ]
    )
    document.build(story, onFirstPage=header_footer, onLaterPages=header_footer)
    return output


if __name__ == "__main__":
    print(build_pdf())
