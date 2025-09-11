# (Full file content begins)
import io
import os
import math
import sys
import json
import base64
import traceback
from io import BytesIO

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Try to import ReportLab; if not available PDF generation will be disabled
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except Exception:
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    RLImage = None
    getSampleStyleSheet = None
    REPORTLAB_AVAILABLE = False

# -------------------------
# Helper: Plotly -> PNG conversion and Matplotlib fallbacks
# -------------------------
def _fig_to_png_bytes(fig):
    """
    Convert a Plotly figure to PNG bytes using the built-in to_image method.
    Return bytes or None on failure.
    """
    try:
        # Plotly's to_image may not be available in constrained envs
        if hasattr(fig, "to_image"):
            return fig.to_image(format="png", engine="kaleido")
        else:
            # Try saving via fig.write_image if present
            buf = BytesIO()
            fig.write_image(buf, format="png")
            buf.seek(0)
            return buf.read()
    except Exception:
        try:
            # fallback: render raw HTML -> PNG is fragile; return None
            return None
        except Exception:
            try:
                plt.close("all")
            except Exception:
                pass
            return None

def _mpl_fig_to_png_bytes(fig):
    """
    Convert a Matplotlib figure to PNG bytes.
    """
    try:
        buf = BytesIO()
        FigureCanvas(fig).print_png(buf)
        buf.seek(0)
        return buf.getvalue()
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None

# Minimal Matplotlib renderers for fallback (trend, pareto)
def _mpl_png_trend_from_df(df, x_field, pa_field, delay_field, title="Trend"):
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax2 = ax.twinx()
        df_plot = df.copy()
        df_plot.plot(kind="bar", x=x_field, y=pa_field, ax=ax, legend=False)
        df_plot.plot(kind="line", x=x_field, y=delay_field, ax=ax2, legend=False)
        ax.set_ylabel("PA")
        ax2.set_ylabel("Delay Hours")
        ax.set_title(title)
        plt.tight_layout()
        return _mpl_fig_to_png_bytes(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None

def _mpl_png_pareto_from_df(df, x_field, y_field, title="Pareto"):
    try:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        df_sorted = df.sort_values(by=y_field, ascending=False)
        ax.bar(df_sorted[x_field], df_sorted[y_field])
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return _mpl_fig_to_png_bytes(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None

def _mpl_png_bar_from_df(df, x_field, y_field, title="Bar"):
    try:
        fig, ax = plt.subplots(figsize=(6, 4))
        df_sorted = df.sort_values(by=y_field, ascending=False)
        ax.bar(df_sorted[x_field], df_sorted[y_field])
        ax.set_title(title)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        return _mpl_fig_to_png_bytes(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None

# -------------------------
# Helper: create PDF bytes from title, KPI text, and list of PNG bytes
# -------------------------
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

def _create_pdf_bytes(title, kpi_text, png_byte_list):
    """
    Builds a simple PDF with:
      - title
      - a KPI table parsed from kpi_text (lines "Metric: value")
      - images provided as PNG byte blobs
    Returns bytes or None on error.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    try:
        normal_style = getSampleStyleSheet().get("Normal")
    except Exception:
        normal_style = None

    elements = []
    # Title
    try:
        styles = getSampleStyleSheet()
        elements.append(Paragraph(title, styles.get("Title", styles["Normal"])))
        elements.append(Spacer(1, 0.15 * inch))
    except Exception:
        pass

    # Build KPI table rows
    try:
        kpi_rows = []
        if kpi_text and isinstance(kpi_text, str) and kpi_text.strip():
            for line in str(kpi_text).splitlines():
                parts = [p.strip() for p in line.split(":", 1)]
                if len(parts) == 2:
                    kpi_rows.append([parts[0], parts[1]])
                elif line.strip():
                    kpi_rows.append([line.strip(), ""])
        if not kpi_rows:
            kpi_rows = [["Physical Availability (PA)", "N/A"],
                        ["Mechanical Availability (MA)", "N/A"],
                        ["Total Delay (hrs)", "N/A"]]

        data = [["Metric", "Value"]] + kpi_rows
        tbl = Table(data, colWidths=[3.0 * inch, 3.0 * inch])
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4f81bd")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ]))
        elements.append(tbl)
        elements.append(Spacer(1, 0.15 * inch))
    except Exception:
        if normal_style and kpi_text:
            elements.append(Paragraph(kpi_text.replace("\n", "<br/>"), normal_style))

    # Add images
    for png_bytes in (png_byte_list or []):
        try:
            if not png_bytes:
                continue
            bio = BytesIO(png_bytes)
            # adjust image width to document width (approx)
            img_width_inch = 6.5
            rl_img = RLImage(bio, width=img_width_inch * inch)
            elements.append(rl_img)
            elements.append(Spacer(1, 0.12 * inch))
        except Exception:
            continue

    # --- Build PDF
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=30, leftMargin=30,
            topMargin=30, bottomMargin=18
        )
        doc.build(elements)
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None

# -------------------------
# Prepare session_state holders for cached PNGs (used in PDF)
# -------------------------
pdf_keys = ['pdf_fig_trend','pdf_fig_pareto','pdf_fig_mttr_w','pdf_fig_mttr_m','pdf_fig_mtbf_w','pdf_fig_mtbf_m']
for k in pdf_keys:
    if k not in st.session_state:
        st.session_state[k] = None
# flag to request PDF generation after KPIs are computed
if '_pdf_request' not in st.session_state:
    st.session_state['_pdf_request'] = False
if '_last_pdf' not in st.session_state:
    st.session_state['_last_pdf'] = None

# -------------------------
# Streamlit page config and header
# -------------------------
st.set_page_config(page_title="Physical Availability - Data Delay Time", layout="wide")

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .css-1d391kg { width: 340px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Logo + Title (unchanged)
LOGO_URL = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/refs/heads/main/images/alamtri_logo.jpeg"
logo_col, title_col = st.columns([1, 8])
with logo_col:
    st.image(LOGO_URL, width=150)
with title_col:
    st.title("Physical Availability Dashboard — Data Delay Time")

# create tabs (Main shown implicitly; Reliability as second tab)
tabs = st.tabs(["Main Dashboard", "Reliability"])

# (Other parts of your code remain unchanged here — data loading, helpers, reliability functions, etc.)
# For brevity in this code dump, the rest of your original script follows exactly as it was,
# up until the PDF-handling sidebar block which we update minimally.

# --- Main script continues ---
# call the MTBF/MTTR computation (cached)
_mtbf_mttr_res = compute_mtbf_mttr_from_url(RAW_URL)

# Expose reliability globals
MTBF_GLOBAL_HOURS = _mtbf_mttr_res.get("MTBF_hours_overall") if isinstance(_mtbf_mttr_res, dict) else None
MTTR_GLOBAL_HOURS = _mtbf_mttr_res.get("MTTR_hours_overall") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_GLOBAL_HOURS_ROUNDED = round(MTBF_GLOBAL_HOURS, 2) if (MTBF_GLOBAL_HOURS is not None) else None
MTTR_GLOBAL_HOURS_ROUNDED = round(MTTR_GLOBAL_HOURS, 2) if (MTTR_GLOBAL_HOURS is not None) else None

MTBF_SOURCE_SHEET = _mtbf_mttr_res.get("sheet_name") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_OP_HOURS_COL = _mtbf_mttr_res.get("operational_hours_column") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_MAINT_DELAY_COL = _mtbf_mttr_res.get("maintenance_delay_column") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_TOTAL_ROWS = _mtbf_mttr_res.get("total_rows") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_TOTAL_OP_HOURS = _mtbf_mttr_res.get("total_operational_hours") if isinstance(_mtbf_mttr_res, dict) else None

# (rest of the unchanged UI building code goes here: sidebars, filters, etc.)
# Build month list from PERIOD_MONTH chronologically
months_available = []
if "PERIOD_MONTH" in df.columns:
    unique_pm = pd.Series(df["PERIOD_MONTH"].dropna().astype(str).str.strip().unique())
    parsed = pd.to_datetime(unique_pm, format="%b %Y", errors="coerce")
    if parsed.notna().any():
        months_df = pd.DataFrame({"PERIOD_MONTH": unique_pm.values, "period_dt": parsed.values})
        months_df = months_df.sort_values("period_dt")
        months_available = months_df["PERIOD_MONTH"].tolist()

# fallback if not found
if not months_available and "MONTH" in df.columns and "YEAR" in df.columns:
    month_to_idx = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
    tmp = []
    for _, r in df.iterrows():
        m = (r.get("MONTH") or "")
        y = r.get("YEAR")
        if pd.isna(y):
            continue
        try:
            y_int = int(y)
        except Exception:
            continue
        m_upper = m.upper()
        if m_upper not in month_to_idx:
            continue
        tmp.append((y_int, month_to_idx[m_upper], f"{m_upper} {y_int}"))
    tmp_sorted = sorted(set(tmp), key=lambda x: (x[0], x[1]))
    months_available = [t[2] for t in tmp_sorted]

# default latest month index
default_idx = 0
if months_available:
    default_idx = len(months_available) - 1

# Sidebar controls (unchanged)
granularity = st.sidebar.selectbox("Time granularity", options=["WEEK", "PERIOD_MONTH"], index=1)

# -------------------------
# PDF sidebar button (modified)
# -------------------------
pdf_keys = globals().get(
    "pdf_keys",
    [
        "pdf_fig_trend",
        "pdf_fig_pareto",
        "pdf_fig_mttr_w",
        "pdf_fig_mttr_m",
        "pdf_fig_mtbf_w",
        "pdf_fig_mtbf_m",
    ],
)

if REPORTLAB_AVAILABLE:
    # Generate PDF when user clicks button
    if st.sidebar.button("Generate PDF"):
        # Defer PDF creation until after KPI computations (to ensure KPI text is populated)
        st.session_state["_pdf_request"] = True
        st.sidebar.info("PDF requested — it will be generated with current KPIs and charts.")

    if st.session_state.get("_last_pdf") is not None:
        st.sidebar.download_button(
            "Download latest PDF",
            data=st.session_state["_last_pdf"],
            file_name="PA_Report.pdf",
            mime="application/pdf",
        )

# (the rest of your app UI -- filters, KPI computations, charts -- run here in order)
# -------------------------
# Apply selected filters to df (month & years)
# -------------------------
filtered = df.copy()
if selected_month != "All" and selected_month != "":
    filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]
if selected_years:
    if "YEAR" in filtered.columns:
        filtered = filtered[filtered["YEAR"].isin(selected_years)].copy()

# -------------------------
# KPI calculations (unchanged)
# -------------------------
total_delay = filtered["DELAY"].sum()

available_time = None
try:
    if "AVAILABLE_TIME_MONTH" in filtered.columns and filtered["AVAILABLE_TIME_MONTH"].notna().any():
        available_time = filtered.groupby("PERIOD_MONTH", dropna=True)["AVAILABLE_TIME_MONTH"].max().dropna().sum()
    elif "AVAILABLE_HOURS" in filtered.columns and filtered["AVAILABLE_HOURS"].notna().any():
        available_time = filtered.groupby("PERIOD_MONTH", dropna=True)["AVAILABLE_HOURS"].max().dropna().sum()
    else:
        available_time = None
except Exception:
    available_time = None

PA = max(0, 1 - total_delay / available_time) if (available_time and available_time > 0) else None
maintenance_delay = filtered[filtered["CATEGORY"] == "Maintenance"]["DELAY"].sum() if "CATEGORY" in filtered.columns else 0
MA = max(0, 1 - maintenance_delay / available_time) if (available_time and available_time > 0) else None

pa_target = filtered["PA_TARGET"].dropna().unique().tolist() if "PA_TARGET" in filtered.columns else []
ma_target = filtered["MA_TARGET"].dropna().unique().tolist() if "MA_TARGET" in filtered.columns else []
try:
    pa_target = pa_target[0] if isinstance(pa_target, (list, tuple)) and pa_target else (pa_target if isinstance(pa_target, (int, float)) else 0.9)
except Exception:
    pa_target = 0.9
try:
    ma_target = ma_target[0] if isinstance(ma_target, (list, tuple)) and ma_target else (ma_target if isinstance(ma_target, (int, float)) else 0.95)
except Exception:
    ma_target = 0.95

if isinstance(pa_target, (int, float)) and pa_target > 1:
    pa_target = pa_target / 100.0
if isinstance(ma_target, (int, float)) and ma_target > 1:
    ma_target = ma_target / 100.0

# Build KPI header text for PDF (now that PA/MA exist)
try:
    _pdf_kpi_text = f"PA: {PA:.2%}\nMA: {MA:.2%}\nTotal Delay (selected): {total_delay:.2f} hrs\n"
    if available_time:
        _pdf_kpi_text += f"Total Available Time (selected): {available_time:.2f} hrs\n"
except Exception:
    _pdf_kpi_text = ""

# -------------------------
# YTD calculations (unchanged)
# -------------------------
ytd_PA = ytd_MA = None
ytd_total_delay = None
try:
    latest_filtered = filtered if not filtered.empty else df
    if not latest_filtered.empty:
        latest_year = int(latest_filtered["YEAR"].dropna().max())
        period_dt_all = pd.to_datetime(latest_filtered["PERIOD_MONTH"], format="%b %Y", errors="coerce")
        if period_dt_all.notna().any():
            latest_period_dt = period_dt_all.max()
            df_period_dt = pd.to_datetime(df["PERIOD_MONTH"], format="%b %Y", errors="coerce")
            ytd_df = df[(df["YEAR"] == latest_year) & (df_period_dt <= latest_period_dt)]
        else:
            ytd_df = df[df["YEAR"] == latest_year]
        ytd_total_delay = ytd_df["DELAY"].sum()
        if "AVAILABLE_TIME_MONTH" in ytd_df.columns and ytd_df["AVAILABLE_TIME_MONTH"].notna().any():
            ytd_available_time = ytd_df.groupby("PERIOD_MONTH")["AVAILABLE_TIME_MONTH"].max().dropna().sum()
        elif "AVAILABLE_HOURS" in ytd_df.columns and ytd_df["AVAILABLE_HOURS"].notna().any():
            ytd_available_time = ytd_df.groupby("PERIOD_MONTH")["AVAILABLE_HOURS"].max().dropna().sum()
        else:
            ytd_available_time = None
        if ytd_available_time and ytd_available_time > 0:
            ytd_PA = max(0, 1 - ytd_total_delay / ytd_available_time)
            ytd_maintenance_delay = ytd_df[ytd_df["CATEGORY"] == "Maintenance"]["DELAY"].sum()
            ytd_MA = max(0, 1 - ytd_maintenance_delay / ytd_available_time)
        else:
            ytd_PA = None
            ytd_MA = None
except Exception:
    ytd_PA = ytd_MA = None
    ytd_total_delay = None

# (KPI display - unchanged)
try:
    min_y = int(df["YEAR"].dropna().min())
    max_y = int(df["YEAR"].dropna().max())
    min_caption = f"01/01/{min_y}"
    max_caption = f"31/12/{max_y}"
except Exception:
    min_caption = None
    max_caption = None

st.caption(f"Data obtained from {min_caption} to {max_caption}" if min_caption and max_caption else "Data obtained from unknown date range")

st.metric("Physical Availability (PA)", f"{PA:.2%}" if PA is not None else "N/A", delta=f"Target {pa_target:.2%}")
st.metric("Mechanical Availability (MA)", f"{MA:.2%}" if MA is not None else "N/A", delta=f"Target {ma_target:.2%}")
st.metric("Total Delay Hours (selected)", f"{total_delay:.2f} hrs")
st.metric("Total Available Time (selected)", f"{available_time:.2f} hrs" if available_time else "N/A")

if ytd_PA is not None:
    st.write("")
    st.caption(f"YTD (up to selected): PA {ytd_PA:.2%} | MA {ytd_MA:.2%} | Delay {ytd_total_delay:.2f} hrs")
else:
    st.write("")

# (Chart generation code follows and each figure caches PNG into st.session_state keys as before)
# Example: after creating fig_trend you already cache a PNG into st.session_state['pdf_fig_trend']
# (Your existing code that creates fig_trend, fig_pareto, fig_mttr_w, fig_mttr_m, fig_mtbf_w, fig_mtbf_m stays intact.)

# If a PDF was requested earlier, generate it now after all figures have been created and cached
if st.session_state.get("_pdf_request"):
    figs_for_pdf = []
    # Use the pdf_keys to fetch cached PNGs; if missing, attempt to render from figure objects
    for k in pdf_keys:
        val = st.session_state.get(k, None)
        if isinstance(val, (bytes, bytearray)):
            figs_for_pdf.append(val)
            continue
        # try to map pdf_ key to a figure name, e.g. pdf_fig_trend -> fig_trend
        fig_name = k.replace('pdf_', '')
        fig_obj = globals().get(fig_name)
        if fig_obj is None:
            continue
        try:
            png = _fig_to_png_bytes(fig_obj)
            if png:
                st.session_state[k] = png
                figs_for_pdf.append(png)
        except Exception:
            pass

    # Now create the PDF using the KPI text populated earlier
    pdf_bytes = _create_pdf_bytes("Physical Availability Report", _pdf_kpi_text, figs_for_pdf)
    if pdf_bytes:
        st.session_state["_last_pdf"] = pdf_bytes
        st.sidebar.success("PDF generated and ready to download.")
    else:
        st.sidebar.error("Failed to generate PDF. Check server logs and installed dependencies (ReportLab).")
    # Reset the request flag
    st.session_state["_pdf_request"] = False

st.caption("MTBF and MTTR shown is based on the period selected")

# End of script
# (No further modifications)
