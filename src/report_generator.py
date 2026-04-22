import streamlit as st
import pandas as pd
import base64
import plotly.express as px

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False


# ── PDF Builder ────────────────────────────────────────────────────────────────
def build_pdf_report(
    df:            pd.DataFrame,
    raw_shape:     tuple,
    cleaned_shape: tuple,
) -> bytes | None:
    """
    Build a static PDF report.
    Returns raw bytes on success, None on failure.
    """
    if not FPDF_AVAILABLE:
        st.error("Install `fpdf2` to generate PDF reports: `pip install fpdf2`")
        return None

    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Auto Data Science Report", ln=1, align="C")

    # Section 1: Dataset dimensions
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "1. Dataset Summary", ln=1)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Original Dimensions: {raw_shape[0]:,} rows, {raw_shape[1]} columns", ln=1)
    pdf.cell(0, 10, f"Cleaned Dimensions:  {cleaned_shape[0]:,} rows, {cleaned_shape[1]} columns", ln=1)

    rows_removed = raw_shape[0] - cleaned_shape[0]
    cols_removed = raw_shape[1] - cleaned_shape[1]
    pdf.cell(0, 10, f"Rows removed: {rows_removed:,} | Columns removed: {cols_removed}", ln=1)

    pdf.ln(5)

    # Section 2: Missing values summary
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "2. Missing Values (Cleaned Data)", ln=1)
    pdf.set_font("Helvetica", "", 10)
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if missing_cols.empty:
        pdf.cell(0, 8, "No missing values remain after cleaning.", ln=1)
    else:
        for col, cnt in missing_cols.items():
            # FIX: encode non-ASCII col names to avoid FPDF latin-1 errors
            safe_col = str(col).encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 8, f"  {safe_col}: {cnt} missing", ln=1)

    pdf.ln(5)

    # Section 3: Descriptive statistics
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "3. Descriptive Statistics (Top 5 Numeric Columns)", ln=1)
    pdf.set_font("Helvetica", "", 10)
    num_df = df.select_dtypes(include=["number"])
    if num_df.empty:
        pdf.cell(0, 8, "No numeric columns found.", ln=1)
    else:
        desc = num_df.describe().round(2)
        for col in list(desc.columns)[:5]:
            safe_col = str(col).encode("latin-1", errors="replace").decode("latin-1")
            mean_v   = desc.loc["mean", col]
            max_v    = desc.loc["max",  col]
            min_v    = desc.loc["min",  col]
            std_v    = desc.loc["std",  col]
            pdf.cell(0, 8, f"{safe_col} - Mean: {mean_v} | Std: {std_v} | Min: {min_v} | Max: {max_v}", ln=1)

    pdf.ln(5)

    # Section 4: Categorical overview
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "4. Categorical Columns Overview", ln=1)
    pdf.set_font("Helvetica", "", 10)
    cat_df = df.select_dtypes(include=["object", "category"])
    if cat_df.empty:
        pdf.cell(0, 8, "No categorical columns.", ln=1)
    else:
        for col in list(cat_df.columns)[:5]:
            safe_col = str(col).encode("latin-1", errors="replace").decode("latin-1")
            n_unique  = cat_df[col].nunique()
            top_val   = str(cat_df[col].mode()[0]) if not cat_df[col].mode().empty else "N/A"
            safe_top  = top_val.encode("latin-1", errors="replace").decode("latin-1")
            pdf.cell(0, 8, f"{safe_col}: {n_unique} unique values | Top: {safe_top}", ln=1)

    pdf.ln(10)

    # Footer notice
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "5. Notice", ln=1)
    pdf.set_font("Helvetica", "", 12)
    pdf.multi_cell(
        0, 8,
        "This PDF captures key statistics from your cleaned dataset. "
        "For interactive charts and full visual analysis, generate the HTML report instead."
    )

    # FIX: use output() to bytes directly instead of writing a temp file
    try:
        pdf_bytes = pdf.output()   # fpdf2 returns bytes
        return bytes(pdf_bytes)
    except TypeError:
        # older fpdf fallback: output() returns string
        import io
        buf = io.BytesIO()
        pdf.output(buf)
        return buf.getvalue()


# ── HTML Builder ───────────────────────────────────────────────────────────────
def build_html_report(df: pd.DataFrame, raw_shape: tuple) -> str:
    """
    Build a self-contained interactive HTML report using Plotly charts.
    All charts are embedded inline so no server is required to view.
    """
    # ── Head ──────────────────────────────────────────────────────────────────
    html_out = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Auto Data Science — Interactive Report</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: 'Segoe UI', Helvetica, Arial, sans-serif;
      margin: 0; padding: 24px;
      background: #f1f5f9; color: #1e293b;
    }}
    .container {{
      max-width: 1200px; margin: 0 auto;
      background: #fff; padding: 40px;
      border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }}
    h1 {{ color: #0ea5e9; font-size: 2rem; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; margin-top: 0; }}
    h2 {{ color: #334155; font-size: 1.3rem; margin-top: 2rem; }}
    .metric-grid {{
      display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px; margin: 16px 0;
    }}
    .metric-card {{
      background: #f8fafc; border: 1px solid #e2e8f0;
      border-radius: 8px; padding: 16px; text-align: center;
    }}
    .metric-card .value {{ font-size: 1.6rem; font-weight: 700; color: #0ea5e9; }}
    .metric-card .label {{ font-size: 0.8rem; color: #64748b; margin-top: 4px; }}
    pre {{
      background: #1e293b; color: #f8fafc;
      padding: 16px; border-radius: 8px;
      font-size: 13px; overflow-x: auto; white-space: pre-wrap;
    }}
    .chart-wrap {{ margin: 24px 0; }}
    footer {{ margin-top: 40px; font-size: 0.8rem; color: #94a3b8; text-align: center; }}
  </style>
</head>
<body>
<div class="container">
  <h1>📊 Auto Data Science — Interactive Report</h1>

  <h2>Section 1 — Dataset Overview</h2>
  <div class="metric-grid">
    <div class="metric-card"><div class="value">{raw_shape[0]:,}</div><div class="label">Original Rows</div></div>
    <div class="metric-card"><div class="value">{raw_shape[1]}</div><div class="label">Original Columns</div></div>
    <div class="metric-card"><div class="value">{df.shape[0]:,}</div><div class="label">Cleaned Rows</div></div>
    <div class="metric-card"><div class="value">{df.shape[1]}</div><div class="label">Cleaned Columns</div></div>
    <div class="metric-card"><div class="value">{int(df.isnull().sum().sum())}</div><div class="label">Missing Values Remaining</div></div>
    <div class="metric-card"><div class="value">{df.select_dtypes(include="number").shape[1]}</div><div class="label">Numeric Columns</div></div>
    <div class="metric-card"><div class="value">{df.select_dtypes(include="object").shape[1]}</div><div class="label">Categorical Columns</div></div>
  </div>

  <h2>Section 2 — Missing Values by Column</h2>
  <pre>{df.isnull().sum().to_string()}</pre>
"""

    # Section 3: Correlation heatmap
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if len(num_cols) >= 2:
        html_out += "\n  <h2>Section 3 — Correlation Matrix</h2>\n  <div class='chart-wrap'>"
        # FIX: truncate to 20 cols max for readability
        plot_cols = num_cols[:20]
        corr_fig  = px.imshow(
            df[plot_cols].corr(),
            text_auto=".2f", aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Interactive Correlation Matrix"
        )
        html_out += corr_fig.to_html(full_html=False, include_plotlyjs=False)
        html_out += "  </div>"
    else:
        html_out += "\n  <h2>Section 3 — Correlation Matrix</h2>\n  <p>Fewer than 2 numeric columns — correlation matrix not available.</p>"

    # Section 4: Distributions (top 4 numeric cols)
    html_out += "\n  <h2>Section 4 — Feature Distributions</h2>"
    for col in num_cols[:4]:
        dist_fig = px.histogram(
            df, x=col, marginal="box",
            title=f"Distribution: {col}"
        )
        html_out += f"\n  <div class='chart-wrap'>{dist_fig.to_html(full_html=False, include_plotlyjs=False)}</div>"

    # Section 5: Top categorical distributions
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if cat_cols:
        html_out += "\n  <h2>Section 5 — Categorical Distributions</h2>"
        for col in cat_cols[:3]:
            counts = df[col].value_counts().head(20).reset_index()
            counts.columns = [col, "Count"]
            bar_fig = px.bar(counts, x=col, y="Count", title=f"Top values: {col}")
            html_out += f"\n  <div class='chart-wrap'>{bar_fig.to_html(full_html=False, include_plotlyjs=False)}</div>"

    # Footer
    html_out += """
  <footer>Generated by Auto Data Science Pro &nbsp;|&nbsp; Open in any browser for full interactivity.</footer>
</div>
</body>
</html>"""

    return html_out


# ── Streamlit section ──────────────────────────────────────────────────────────
def render_report_section():
    st.header("📄 Automated Report Generator")
    st.write(
        "Generate a downloadable report of your dataset with key statistics "
        "and interactive visualizations."
    )
    st.divider()

    # FIX: check both raw and cleaned data are available
    if st.session_state.get("cleaned_data") is None or st.session_state.get("raw_data") is None:
        st.warning(
            "Please upload and clean a dataset first (Steps 1 & 4 or Step 3) "
            "before generating a report."
        )
        return

    cleaned_df = st.session_state.cleaned_data
    raw_shape  = st.session_state.raw_data.shape

    col1, col2 = st.columns(2)

    with col1:
        st.write("### Interactive HTML Report ⭐")
        st.info(
            "Complete interactive charts in a self-contained HTML file. "
            "Open in any browser — no server required."
        )
        if st.button("Build Interactive HTML Report", type="primary"):
            with st.spinner("Compiling HTML report..."):
                try:
                    html_blob = build_html_report(cleaned_df, raw_shape)
                    b64       = base64.b64encode(html_blob.encode("utf-8")).decode("utf-8")
                    href = (
                        f'<a href="data:text/html;base64,{b64}" '
                        f'download="data_science_report.html" target="_blank">'
                        f'<button style="background:#0ea5e9;color:#fff;border:none;'
                        f'padding:10px 22px;border-radius:6px;font-weight:bold;cursor:pointer;'
                        f'font-size:15px;">📥 Download Interactive HTML</button></a>'
                    )
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("HTML report ready! Click the button above to download.")
                except Exception as e:
                    st.error(f"HTML report generation failed: {e}")

    with col2:
        st.write("### PDF Report")
        st.info(
            "Compact static PDF with summary statistics. "
            "Best for printing or sharing without a browser."
        )
        if st.button("Build PDF Report"):
            with st.spinner("Generating PDF..."):
                pdf_bytes = build_pdf_report(
                    cleaned_df, raw_shape, cleaned_df.shape
                )
                if pdf_bytes is not None:
                    st.download_button(
                        label="📥 Download PDF",
                        data=pdf_bytes,
                        file_name="data_science_report.pdf",
                        mime="application/pdf",   # FIX: correct MIME type
                    )
                    st.success("PDF ready!")