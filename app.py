import streamlit as st
import pandas as pd
import os
from src.data_loading import load_data
from src.eda import show_data_info, show_missing_values, show_summary_statistics
from src.cleaning import auto_clean_data
from src.feature_engineering import encode_categorical, scale_numerical, auto_feature_engineering
from src.visualization import plot_histogram, plot_correlation_heatmap, plot_pairplot
from src.modeling import prepare_modeling
from src.time_series import run_time_series_analysis
from src.report_generator import render_report_section
from src.ai_assistant import render_ai_assistant
from src.smart_mode import smart_auto_pipeline
from src.advanced_analysis import render_advanced_analysis

st.set_page_config(
    page_title="Auto Data Science Pro",
    layout="wide",
    page_icon="📊"
)

# ── Section definitions ────────────────────────────────────────────────────────
SECTIONS = {
    "1":  ("1. Upload Dataset",                  "upload"),
    "2":  ("2. Exploratory Data Analysis",        "eda"),
    "3":  ("3. Smart Auto Mode (One Click) 🚀",   "smart"),
    "4":  ("4. Advanced Cleaning (Manual)",        "cleaning"),
    "5":  ("5. Feature Tracking (Manual)",         "features"),
    "6":  ("6. Visualization",                     "viz"),
    "7":  ("7. 📊 Advanced Analysis",              "advanced"),
    "8":  ("8. Time Series Analysis",              "timeseries"),
    "9":  ("9. Machine Learning",                  "ml"),
    "10": ("10. AI Assistant Chat 🤖",             "ai"),
    "11": ("11. 📄 Report Generator",              "report"),
}
SECTION_LABELS = [v[0] for v in SECTIONS.values()]
SECTION_KEYS   = {v[0]: v[1] for v in SECTIONS.values()}


# ── Session state ──────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "raw_data":      None,
        "cleaned_data":  None,
        "encoded_data":  None,
        "pipeline_logs": [],
        "model_results": None,
        "ts_results":    None,
        "report_data":   {},
        "test_data":     None,
        "best_model":    None,
        "best_model_name": None,
        "feature_cols":  None,
        "task_type":     None,
        "encoders":      None,
        "scaler":        None,
        "chat_messages": [],
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_pipeline():
    """Clear all pipeline state (keeps API key env var intact)."""
    keys_to_clear = [
        "raw_data", "cleaned_data", "encoded_data",
        "pipeline_logs", "model_results", "ts_results", "report_data",
        "test_data", "best_model", "best_model_name",
        "feature_cols", "task_type", "encoders", "scaler", "chat_messages",
    ]
    for k in keys_to_clear:
        if k in ("pipeline_logs", "chat_messages"):
            st.session_state[k] = []
        elif k in ("report_data",):
            st.session_state[k] = {}
        else:
            st.session_state[k] = None


# ── Sidebar status indicator ───────────────────────────────────────────────────
def sidebar_pipeline_status():
    st.sidebar.markdown("**Pipeline Status**")
    checks = [
        ("Raw data loaded",     st.session_state.raw_data     is not None),
        ("Data cleaned",        st.session_state.cleaned_data is not None),
        ("Features engineered", st.session_state.encoded_data is not None),
        ("Model trained",       st.session_state.model_results is not None),
    ]
    for label, ok in checks:
        icon = "🟢" if ok else "⚪"
        st.sidebar.markdown(f"{icon} {label}")
    st.sidebar.divider()


# ── API key validation ─────────────────────────────────────────────────────────
def validate_and_set_api_key(key: str) -> bool:
    """Returns True if key looks valid and sets env var."""
    if not key:
        return False

    if len(key) < 20:   
        st.sidebar.error("⚠️ Invalid Gemini API key format. Keys start with 'AIza'.")
        return False
    os.environ["GOOGLE_API_KEY"] = key
    st.sidebar.success("✅ API key set")
    return True


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    init_session()

    # Sidebar
    st.sidebar.title("📊 Auto Data Science Pro")
    st.sidebar.write(
        "Upload a dataset and deploy an entire Data Science pipeline natively."
    )
    st.sidebar.divider()

    google_key = st.sidebar.text_input(
        "Google Gemini API Key", type="password",
        placeholder="AIza...",
        help="Required for AI Assistant and Smart Auto Mode."
    )
    validate_and_set_api_key(google_key)
    st.sidebar.divider()

    sidebar_pipeline_status()

    app_mode = st.sidebar.radio("Dashboard Sections", SECTION_LABELS)
    section  = SECTION_KEYS[app_mode]

    st.sidebar.divider()
    if st.sidebar.button("🔄 Reset Pipeline", help="Clears all loaded and processed data."):
        reset_pipeline()
        st.rerun()

    # ── 1. Upload ──────────────────────────────────────────────────────────────
    if section == "upload":
        st.header("Upload your CSV dataset")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file", type=["csv", "xlsx", "xls"] 
        )
        
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                st.session_state.raw_data     = df
                st.session_state.cleaned_data = None
                st.session_state.encoded_data = None
                st.session_state.model_results = None
                st.session_state.test_data     = None

        if st.session_state.raw_data is not None:
            df = st.session_state.raw_data
            st.success(f"Dataset loaded: **{df.shape[0]:,} rows × {df.shape[1]} columns**")
            st.write("### Preview")
            st.dataframe(df.head(10), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Rows",    f"{df.shape[0]:,}")
            col2.metric("Columns", df.shape[1])
            col3.metric("Missing values", int(df.isnull().sum().sum()))

    # ── 2. EDA ─────────────────────────────────────────────────────────────────
    elif section == "eda":
        st.header("Exploratory Data Analysis")
        if st.session_state.raw_data is None:
            st.warning("Please upload a dataset in Step 1.")
            st.stop()

        df = st.session_state.raw_data
        show_data_info(df)
        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            show_missing_values(df)
        with col2:
            show_summary_statistics(df)

    # ── 3. Smart Auto Mode ─────────────────────────────────────────────────────
    elif section == "smart":
        st.header("🚀 Smart Auto Mode")
        st.write(
            "Sit back and let the pipeline handle cleaning, encoding, and scaling "
            "automatically based on cardinality constraints and standard deviation analysis."
        )

        if st.session_state.raw_data is None:
            st.warning("Please upload a dataset first.")
            st.stop()

        target_col = st.selectbox(
            "Select your target variable (required for feature weight analysis):",
            st.session_state.raw_data.columns.tolist()
        )

        if st.button("🚀 Run Smart Pipeline", type="primary"):
            with st.spinner("Processing..."):
                df_out, logs, i_score, f_score = smart_auto_pipeline(
                    st.session_state.raw_data, target_col
                )
                st.session_state.encoded_data  = df_out
                st.session_state.cleaned_data  = df_out
                st.session_state.pipeline_logs = logs

            st.success(
                f"Done! Data readiness score: **{i_score:.1f}% → {f_score:.1f}%**"
            )

            with st.expander("View processing logs", expanded=True):
                for line in logs:
                    st.write(line)

            st.divider()
            st.write("### Processed output")
            st.dataframe(df_out.head(), use_container_width=True)
            st.write(f"Shape: **{df_out.shape[0]:,} rows × {df_out.shape[1]} columns**")

            
            if st.session_state.get("test_data") is not None:
                test_df = st.session_state["test_data"]
                st.info(
                    f"🧪 Held-out test set: **{test_df.shape[0]:,} rows × {test_df.shape[1]} columns** "
                    "— saved for evaluation in the ML section."
                )
            
            csv = df_out.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download processed data",
                data=csv, file_name="smart_processed.csv", mime="text/csv"
            )

    # ── 4. Advanced Cleaning ───────────────────────────────────────────────────
    elif section == "cleaning":
        st.header("Advanced Data Cleaning")

        if st.session_state.raw_data is None:
            st.warning("Please upload a dataset first.")
            st.stop()

        col1, col2 = st.columns(2)
        with col1:
            num_strategy = st.selectbox(
                "Numerical imputation method",
                ["Median", "Mean", "KNN Imputation", "Forward/Backward Fill (Time Series)"]
            )
        with col2:
            outlier_method = st.selectbox("Outlier detection method", ["IQR", "Z-score"])
            outlier_action = st.selectbox("Outlier handling", ["Cap", "Remove"])

        if st.button("Run Cleaning", type="primary"):
            with st.spinner("Cleaning data..."):
                cleaned = auto_clean_data(
                    st.session_state.raw_data,
                    num_strategy=num_strategy,
                    outlier_method=outlier_method,
                    outlier_action=outlier_action
                )
            st.session_state.cleaned_data = cleaned
            # FIX: Reset downstream state when data changes
            st.session_state.encoded_data  = None
            st.session_state.model_results = None
            st.success("Cleaning complete.")

        if st.session_state.cleaned_data is not None:
            st.write("### Cleaned dataset")
            st.dataframe(st.session_state.cleaned_data.head(), use_container_width=True)

            col1, col2 = st.columns(2)
            col1.metric("Rows",    f"{st.session_state.cleaned_data.shape[0]:,}")
            col2.metric("Columns", st.session_state.cleaned_data.shape[1])

            csv = st.session_state.cleaned_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download cleaned data (.csv)",
                data=csv, file_name="cleaned_dataset.csv", mime="text/csv"
            )

    # ── 5. Feature Engineering ─────────────────────────────────────────────────
    elif section == "features":
        st.header("Feature Engineering")
        data_to_use = (
            st.session_state.cleaned_data
            if st.session_state.cleaned_data is not None
            else st.session_state.raw_data
        )

        if data_to_use is None:
            st.warning("Please upload (and optionally clean) a dataset first.")
            st.stop()
        
# FIX: Show which data source is being used
        source = "cleaned" if st.session_state.cleaned_data is not None else "raw"
        st.info(f"ℹ️ Using **{source}** data. Run cleaning first for best results.")

        col1, col2 = st.columns(2)
        with col1:
            encoding_strategy = st.selectbox(
                "Categorical encoding", ["Label Encoding", "One-Hot Encoding"]
            )
            add_poly = st.checkbox("Add polynomial features (degree 2)")
        with col2:
            scaling_strategy = st.selectbox(
                "Numerical scaling", ["None", "Standard Scaling", "Min-Max Normalization"]
            )
            add_ratios = st.checkbox("Generate top feature ratios")

        if st.button("Apply Feature Engineering", type="primary"):
            with st.spinner("Engineering features..."):
                logs                 = []
                df_engineered        = auto_feature_engineering(data_to_use, poly=add_poly, ratios=add_ratios, logs=logs)
                df_encoded, encoders = encode_categorical(df_engineered, strategy=encoding_strategy, logs=logs)
                df_scaled, scaler    = scale_numerical(df_encoded, strategy=scaling_strategy, logs=logs)
            st.session_state.encoded_data = df_scaled
            st.session_state["encoders"]  = encoders
            st.session_state["scaler"]    = scaler
            st.session_state.model_results = None 
            st.success("Feature engineering complete.")
 

        if logs:
            with st.expander("📋 View feature engineering log", expanded=False):
                for i, entry in enumerate(logs, 1):
                    st.write(f"**{i}.** {entry}")
        
        if st.session_state.encoded_data is not None:
            st.write("### Transformed dataset")
            st.dataframe(st.session_state.encoded_data.head(), use_container_width=True)
 st.write(
                f"Shape: **{st.session_state.encoded_data.shape[0]:,} rows × "
                f"{st.session_state.encoded_data.shape[1]} columns**"
            )
            
            csv = st.session_state.encoded_data.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download engineered data (.csv)",
                data=csv, file_name="engineered_dataset.csv", mime="text/csv"
            )

    # ── 6. Visualization ───────────────────────────────────────────────────────
    elif section == "viz":
        st.header("Interactive Visualizations")
        data_to_plot = (
            st.session_state.cleaned_data
            if st.session_state.cleaned_data is not None
            else st.session_state.raw_data
        )

        if data_to_plot is None:
            st.warning("Please upload a dataset first.")
            st.stop()

        tab1, tab2, tab3 = st.tabs(["Histograms", "Correlation Matrix", "Pairplot Scatter"])
        with tab1:
            plot_histogram(data_to_plot)
        with tab2:
            plot_correlation_heatmap(data_to_plot)
        with tab3:
            plot_pairplot(data_to_plot)

    # ── 7. Advanced Analysis ───────────────────────────────────────────────────
    elif section == "advanced":
        st.header("📊 Advanced Analysis")
        data_to_plot = (
            st.session_state.cleaned_data
            if st.session_state.cleaned_data is not None
            else st.session_state.raw_data
        )

        if data_to_plot is None:
            st.warning("Please upload a dataset first.")
            st.stop()

        render_advanced_analysis(data_to_plot)

    # ── 8. Time Series ─────────────────────────────────────────────────────────
    elif section == "timeseries":
        st.header("Time Series Analysis")
        data_to_ts = (
            st.session_state.cleaned_data
            if st.session_state.cleaned_data is not None
            else st.session_state.raw_data
        )

        if data_to_ts is None:
            st.warning("Please upload a dataset first.")
            st.stop()

        run_time_series_analysis(data_to_ts)

    # ── 9. Machine Learning ────────────────────────────────────────────────────
    elif section == "ml":
        st.header("Machine Learning & Explainability")

        if st.session_state.encoded_data is not None:
            prepare_modeling(st.session_state.encoded_data)

        elif st.session_state.cleaned_data is not None:
            st.info(
                "ℹ️ Using cleaned (but not encoded) data. "
                "For best results, run **Feature Tracking** or **Smart Auto Mode** first."
            )
            prepare_modeling(st.session_state.cleaned_data)

        else:
            st.warning("Please upload and process a dataset before running ML.")

    # ── 10. AI Assistant ───────────────────────────────────────────────────────
    elif section == "ai":
        if not os.environ.get("GOOGLE_API_KEY"):
            st.warning("Please enter your Google Gemini API key in the sidebar first.")
            st.stop()

        data_to_chat = (
            st.session_state.cleaned_data
            if st.session_state.cleaned_data is not None
            else st.session_state.raw_data
        )


        if data_to_chat is None:
            st.warning("Please upload a dataset first before using the AI Assistant.")
            st.stop()
    
        render_ai_assistant(data_tochat)

    # ── 11. Report Generator ───────────────────────────────────────────────────
    elif section == "report":
        render_report_section()


if __name__ == "__main__":
    main()