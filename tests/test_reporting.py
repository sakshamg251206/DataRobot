import pandas as pd

from autods.core.reporting import ReportContext, build_html_report, build_pdf_report


def test_html_report_escapes_user_content():
    df = pd.DataFrame({"<script>alert(1)</script>": [1, 2, 3], "b": [4.0, 5.0, None]})
    report = build_html_report(
        ReportContext(df=df, dataset_name="<b>x</b>", processing_log=["a<b"])
    )
    assert "<script>alert(1)</script>" not in report
    assert "&lt;b&gt;x&lt;/b&gt;" in report
    assert report.count("cdn.plot.ly") == 1


def test_pdf_report_handles_unicode_and_models(classification_df):
    summary = pd.DataFrame([{"Model": "Random Forest", "F1": 0.91, "Accuracy": 0.9}])
    ctx = ReportContext(
        df=classification_df.assign(city="Zürich → 東京"),
        dataset_name="données",
        raw_shape=(400, 6),
        processing_log=["Dropped `x` → done"],
        model_summary=summary,
        model_target="label",
        model_task="Classification",
    )
    pdf = build_pdf_report(ctx)
    assert pdf.startswith(b"%PDF")
