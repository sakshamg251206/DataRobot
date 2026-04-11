import streamlit as st
import pandas as pd
from fpdf import FPDF
import base64
import plotly.express as px

# Kept Existing Logic for PDF Standard format constraints
def build_pdf_report(df, raw_shape, cleaned_shape):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 24)
    pdf.cell(0, 20, "Auto Data Science Report", ln=1, align='C')
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, "1. Dataset Summary", ln=1)
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f"Original Dimensions: {raw_shape[0]} rows, {raw_shape[1]} columns", ln=1)
    pdf.cell(0, 10, f"Cleaned Dimensions: {cleaned_shape[0]} rows, {cleaned_shape[1]} columns", ln=1)
    
    pdf.ln(5)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, "2. Descriptive Statistics (Top 5 Numeric)", ln=1)
    pdf.set_font('Helvetica', '', 10)
    num_df = df.select_dtypes(include=['number']).head(5)
    desc = num_df.describe().round(2)
    for col in list(desc.columns)[:3]:
        stat_line = f"{col} - Mean: {desc.loc['mean', col]}, Max: {desc.loc['max', col]}, Min: {desc.loc['min', col]}"
        pdf.cell(0, 8, stat_line, ln=1)

    pdf.ln(10)
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, "3. Notice", ln=1)
    pdf.set_font('Helvetica', '', 12)
    pdf.multi_cell(0, 8, "This basic text-PDF layout captures statistical transformations over your dataset. For rich visual mapping, generate an Interactive HTML Report from the dashboard natively.")
    
    pdf_output = "report.pdf"
    pdf.output(pdf_output)
    with open(pdf_output, "rb") as pdf_file:
        return pdf_file.read()

# NEW HTML Generator System
def build_html_report(df, raw_shape):
    html_out = f"""
    <html>
    <head>
        <title>DataRobot Pro - Interactive Report</title>
        <style>
            body {{ font-family: 'Helvetica', 'Arial', sans-serif; padding: 40px; background-color: #f7f9fc; color: #1e293b; }}
            .container {{ max-width: 1200px; margin: auto; background: white; padding: 40px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }}
            h1 {{ color: #0ea5e9; font-size: 36px; border-bottom: 2px solid #e2e8f0; padding-bottom: 10px; }}
            h2 {{ color: #334155; margin-top: 30px; font-size: 24px; }}
            .metric-box {{ background: #f1f5f9; padding: 20px; border-radius: 8px; margin-bottom: 20px; font-weight: bold; font-size: 18px; color: #0f172a; }}
            pre {{ background: #1e293b; color: #f8fafc; padding: 15px; border-radius: 6px; font-size: 14px; overflow-x: auto; }}
        </style>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>DataRobot Advanced Interactive Report</h1>
            
            <h2>Section 1: Data Topology Overview</h2>
            <div class="metric-box">
                Initial Dataset Dimensions: {raw_shape[0]} rows x {raw_shape[1]} columns <br><br>
                Cleaned Final Frame Matrix Shape: {df.shape[0]} rows x {df.shape[1]} columns
            </div>
            
            <h2>Section 2: Missing Values Validation</h2>
            <pre>{df.isnull().sum().to_string()}</pre>
            
            <h2>Section 3: Mathematical Correlation Arrays</h2>
            """
            
    num_cols = df.select_dtypes(include=['number']).columns
    if len(num_cols) >= 2:
        corr_fig = px.imshow(df[num_cols].corr(), text_auto=".2f", aspect="auto", color_continuous_scale="RdBu_r", title="Interactive Correlation Matrix (Drag/Hover/Select)")
        html_out += corr_fig.to_html(full_html=False, include_plotlyjs=False)
    
    html_out += f"""
            <h2>Section 4: Primary Distribution Isolators</h2>
            """
            
    for col in list(num_cols)[:4]: 
        dist_fig = px.histogram(df, x=col, marginal="box", title=f"Distribution Mapping for Numerical Feature [{col}]")
        html_out += dist_fig.to_html(full_html=False, include_plotlyjs=False)

    html_out += """
        </div>
    </body>
    </html>
    """
    return html_out

def render_report_section():
    st.header("📄 Automated Report Generator")
    st.write("Seamlessly construct a massive downloadable report highlighting the absolute most pivotal structures inside your dataset utilizing dynamic Plotly embeds.")
    st.divider()
    
    if st.session_state.cleaned_data is not None and st.session_state.raw_data is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Interactive HTML Report ⭐")
            st.info("The Premium Choice: Complete interactive visuals and dynamic layout mapping isolated natively inside a standard browser view file.")
            if st.button("Build Interactive HTML"):
                with st.spinner("Compiling Mega-HTML Interactive Payload..."):
                    html_blob = build_html_report(st.session_state.cleaned_data, st.session_state.raw_data.shape)
                    b64 = base64.b64encode(html_blob.encode()).decode()
                    href = f'<a href="data:text/html;base64,{b64}" download="auto_data_science_interactive_report.html" target="_blank" style="text-decoration:none;"><button style="background-color:#0ea5e9; color:white; border:none; padding:10px 20px; border-radius:4px; font-weight:bold; cursor:pointer;" >📥 Download Interactive HTML</button></a>'
                    st.markdown(href, unsafe_allow_html=True)

        with col2:
            st.write("### Statutory PDF Report")
            st.info("Plain-Text PDF: Basic bounding box mapping ideal for text scraping and printing static numbers securely via explicit string formatting constraints.")
            if st.button("Build Classic PDF Layout"):
                with st.spinner("Generating Layout..."):
                    pdf_bytes = build_pdf_report(st.session_state.cleaned_data, st.session_state.raw_data.shape, st.session_state.cleaned_data.shape)
                    st.download_button(label="📥 Download Secure PDF", data=pdf_bytes, file_name="classic_report.pdf", mime='application/octet-stream')

    else:
        st.warning("Please preprocess a dataset through the Auto routine first before establishing Report metrics.")
