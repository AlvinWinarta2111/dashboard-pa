# Full updated script (with Matplotlib fallback for PDF export)
# NOTE: This file is the full script to replace your PA_DB.py
import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# Additional imports for PDF export & fallback
import io
from io import BytesIO
from PIL import Image as PILImage

# Matplotlib for robust PNG fallback (Option A)
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
    A4 = None
    inch = None
    REPORTLAB_AVAILABLE = False

# -------------------------
# Helper: convert Plotly figure to PNG bytes (fallback only)
# -------------------------
def _fig_to_png_bytes(fig):
    """
    Try to convert a Plotly figure to PNG bytes. This function is no longer dependent on kaleido.
    It's kept for any other potential Plotly image export.
    """
    try:
        img_bytes = fig.to_image(format="png")
        if isinstance(img_bytes, (bytes, bytearray)):
            return bytes(img_bytes)
    except Exception:
        return None
    return None

# -------------------------
# Helper: save matplotlib figure to PNG bytes
# -------------------------
def _mpl_fig_to_png_bytes(fig):
    """
    Save a Matplotlib figure to PNG bytes and close the figure.
    """
    try:
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)
        return buf.read()
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None

# -------------------------
# Matplotlib fallback generators for each chart type
# These functions produce a PNG byte array (or None)
# -------------------------
def _mpl_png_trend_from_df(trend_df, x_field, pa_col="PA_pct_rounded", delay_col="total_delay_hours_rounded", title="Trend"):
    """
    Create a Matplotlib representation of the Trend chart (PA% bars + Delay line).
    """
    try:
        # convert to safe dataframe copy
        t = trend_df.copy()
        x = t[x_field].astype(str).tolist()
        y_pa = pd.to_numeric(t[pa_col], errors="coerce").fillna(0).tolist()
        y_delay = pd.to_numeric(t[delay_col], errors="coerce").fillna(0).tolist()

        fig, ax1 = plt.subplots(figsize=(10, 4))
        idx = list(range(len(x)))
        # PA% bars
        ax1.bar(idx, y_pa, width=0.6)
        ax1.set_ylabel("PA%")
        ax1.set_ylim(0, 1)
        ax1.set_xticks(idx)
        ax1.set_xticklabels(x, rotation=45, ha="right", fontsize=8)
        # Delay on second axis
        ax2 = ax1.twinx()
        ax2.plot(idx, y_delay, marker="o", linewidth=2)
        ax2.set_ylabel("Delay Hours")
        fig.suptitle(title)
        fig.tight_layout()
        return _mpl_fig_to_png_bytes(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None

def _mpl_png_pareto_from_df(pareto_df, equipment_key="EQUIPMENT_DESC", title="Pareto by Equipment"):
    """
    Create a Matplotlib Pareto chart (bars for hours + cumulative % line).
    """
    try:
        p = pareto_df.copy()
        p[equipment_key] = p[equipment_key].astype(str)
        labels = p[equipment_key].tolist()
        hours = pd.to_numeric(p["hours"], errors="coerce").fillna(0).tolist()
        cum_pct = pd.to_numeric(p["cum_pct"], errors="coerce").fillna(0).tolist()

        fig, ax1 = plt.subplots(figsize=(10, 4))
        idx = list(range(len(labels)))
        ax1.bar(idx, hours)
        ax1.set_ylabel("Hours")
        ax1.set_xticks(idx)
        ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)

        ax2 = ax1.twinx()
        ax2.plot(idx, [v * 100 for v in cum_pct], color="C1", marker="o")
        ax2.set_ylabel("Cumulative %")
        ax2.set_ylim(0, 100)
        fig.suptitle(title)
        fig.tight_layout()
        return _mpl_fig_to_png_bytes(fig)
    except Exception:
        try:
            plt.close("all")
        except Exception:
            pass
        return None

def _mpl_png_bar_from_df(df, x_col, y_col, title="", color="blue", xlabel="", ylabel=""):
    """
    Generic vertical bar chart fallback for MTTR/MTBF weekly/monthly.
    """
    try:
        d = df.copy()
        x = d[x_col].astype(str).tolist()
        y = pd.to_numeric(d[y_col], errors="coerce").fillna(0).tolist()

        fig, ax = plt.subplots(figsize=(10, 4))
        idx = list(range(len(x)))
        ax.bar(idx, y, color=color)
        ax.set_xticks(idx)
        ax.set_xticklabels(x, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel(xlabel if xlabel else "")
        ax.set_ylabel(ylabel if ylabel else "")
        fig.suptitle(title)
        fig.tight_layout()
        return _mpl_fig_to_png_bytes(fig)
    except Exception:
        try:
            plt.close(fig)
        except Exception:
            pass
        return None
        
def _create_mpl_table_png(dataframe, title):
    """
    Creates and saves a Matplotlib table as a PNG.
    """
    try:
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.axis('off')
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        tbl = ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8)
        tbl.auto_set_column_width(col=list(range(len(dataframe.columns))))
        return _mpl_fig_to_png_bytes(fig)
    except Exception:
        return None

# -------------------------
# Helper: create PDF bytes from title, KPI text, and list of PNG bytes
# -------------------------
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

def _create_pdf_bytes(title, kpi_text, png_byte_list):
    """
    Builds a simple PDF with:
    - Title
    - KPI table (parsed from kpi_text)
    - List of PNG charts (smaller size)
    """
    if not REPORTLAB_AVAILABLE:
        return None

    elements = []
    styles = getSampleStyleSheet() if getSampleStyleSheet else None
    title_style = styles.get("Heading1") if styles else None
    normal_style = styles.get("Normal") if styles else None

    # --- Title
    if title_style:
        elements.append(Paragraph(title, title_style))
    else:
        elements.append(Paragraph(title, normal_style or ""))
    elements.append(Spacer(1, 0.08 * inch))

    # --- KPI table
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
            elements.append(Spacer(1, 0.12 * inch))

    # --- Charts (resize here)
    img_width_inch = 5.0  # adjust this value to resize images
    for png in (png_byte_list or []):
        if not png:
            continue
        try:
            bio = BytesIO(png)
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
pdf_keys = ['pdf_fig_trend','pdf_fig_pareto','pdf_fig_mttr_w','pdf_fig_mttr_m','pdf_fig_mtbf_w','pdf_fig_mtbf_m','pdf_tbl_mttr_equipment_w','pdf_tbl_mttr_equipment_m','pdf_tbl_mtbf_equipment_w','pdf_tbl_mtbf_equipment_m']
for k in pdf_keys:
    if k not in st.session_state:
        st.session_state[k] = None
if '_last_pdf' not in st.session_state:
    st.session_state['_last_pdf'] = None

# -------------------------
# Streamlit page config and header
# -------------------------
st.set_page_config(page_title="Physical Availability - Data Delay Time", layout="wide")

# Auto-hide sidebar script (unchanged)
st.markdown(
    """
    <script>
    (function() {
        let timer;
        function showSidebar() {
            try {
                const sb = window.parent.document.querySelector('[data-testid="stSidebar"]');
                if (sb) sb.style.display = 'block';
            } catch(e) {}
        }
        function hideSidebar() {
            try {
                const sb = window.parent.document.querySelector('[data-testid="stSidebar"]');
                if (sb) sb.style.display = 'none';
            } catch(e) {}
        }
        function resetTimer() {
            showSidebar();
            if (timer) clearTimeout(timer);
            timer = setTimeout(hideSidebar, 10000);
        }
        ['mousemove','mousedown','keydown','touchstart','scroll'].forEach(evt => {
            document.addEventListener(evt, resetTimer, {passive:true});
        });
        window.addEventListener('load', resetTimer);
    })();
    </script>
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

# -------------------------
# Config: source workbook URL (UPDATED AND CORRECTED)
# -------------------------
RAW_URL = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/main/Database%20Delay%20CHPP.xlsx"

# -------------------------
# Load + Clean function (UPDATED to read from correct URL)
# -------------------------
@st.cache_data
def load_data_from_url():
    try:
        # Read the raw excel file from the URL, reading the first sheet to find the header
        raw = pd.read_excel(RAW_URL, sheet_name="Data Delay Time", header=None)
    except Exception as e:
        st.error(f"Unable to read sheet 'Data Delay Time' from the URL. Please check the link and sheet name. Error: {e}")
        return None

    # Detect header row (first 20 rows)
    header_row = None
    for i in range(20):
        row_values = raw.iloc[i].astype(str).str.upper().tolist()
        if "WEEK" in row_values or "MONTH" in row_values or "YEAR" in row_values:
            header_row = i
            break
    if header_row is None:
        st.error("Could not detect header row automatically. Please check the Excel file format.")
        return None

    # Now read the excel file again, this time with the correct header row
    df = pd.read_excel(RAW_URL, sheet_name="Data Delay Time", header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

    # normalize some column names
    replacements = {
        "WEEK": "WEEK",
        "MONTH": "MONTH",
        "YEAR": "YEAR",
        "DELAY": "DELAY",
        "MTN DELAY TYPE": "MTN_DELAY_TYPE",
        "SCH MTN": "SCH_MTN",
        "UNSCH MTN": "UNSCH_MTN",
        "MINING DELAY": "MINING_DELAY",
        "WEATHER DELAY": "WEATHER_DELAY",
        "OTHER DELAY": "OTHER_DELAY",
        "AVAILABLE TIME (MONTH)": "AVAILABLE_TIME_MONTH",
        "MA TARGET": "MA_TARGET",
        "PA TARGET": "PA_TARGET",
        "MTN NOTE": "MTN_NOTE",
        "NOTE": "NOTE",
        "PICA": "PICA", # ADDED
        "START": "START",
        "STOP": "STOP",
        "EQUIPMENT": "EQUIPMENT",
        "EQ. DESC.": "EQ_DESC",
        "CATEGORY": "CATEGORY",
        "PERIOD_MONTH": "PERIOD_MONTH",
        "DATE": "DATE",
    }
    for orig, new in replacements.items():
        for col in df.columns:
            if str(col).strip().upper() == orig:
                df.rename(columns={col: new}, inplace=True)
                break

    essential = ["WEEK", "MONTH", "YEAR", "DELAY"]
    for c in essential:
        if c not in df.columns:
            st.error(f"Expected column '{c}' not found after cleaning. Found columns: {df.columns.tolist()}")
            return None

    # Convert numeric columns
    df["DELAY"] = pd.to_numeric(df["DELAY"], errors="coerce")
    if "AVAILABLE_TIME_MONTH" in df.columns:
        df["AVAILABLE_TIME_MONTH"] = pd.to_numeric(df["AVAILABLE_TIME_MONTH"], errors="coerce")
    else:
        df["AVAILABLE_TIME_MONTH"] = None

    # START/STOP -> AVAILABLE_HOURS
    if "START" in df.columns and "STOP" in df.columns:
        df["START"] = df["START"].astype(str).str.strip()
        df["STOP"] = df["STOP"].astype(str).str.strip()

        def parse_time_to_hours(start_str, stop_str):
            try:
                start_t = datetime.datetime.strptime(start_str, "%H:%M")
                stop_t = datetime.datetime.strptime(stop_str, "%H:%M")
                diff = (stop_t - start_t).total_seconds() / 3600
                if diff < 0:
                    diff += 24
                return diff
            except Exception:
                return None

        df["AVAILABLE_HOURS"] = [
            parse_time_to_hours(s, e) if (str(s).strip() not in ["", "nan", "None"]) and (str(e).strip() not in ["", "nan", "None"]) else None
            for s, e in zip(df["START"], df["STOP"])
        ]
    else:
        df["AVAILABLE_HOURS"] = None
        if "START" not in df.columns:
            df["START"] = ""
        if "STOP" not in df.columns:
            df["STOP"] = ""

    # Ensure presence of helpful columns and cast to string where appropriate
    for cat in ["MTN_DELAY_TYPE", "SCH_MTN", "UNSCH_MTN", "MINING_DELAY", "WEATHER DELAY", "OTHER_DELAY", "MTN_NOTE", "NOTE", "PICA", "EQ_DESC", "EQUIPMENT", "CATEGORY", "PERIOD_MONTH", "DATE"]: # ADDED 'PICA'
        if cat in df.columns:
            df[cat] = df[cat].fillna("").astype(str).str.strip()
        else:
            df[cat] = ""

    # SUB_CATEGORY detection
    def detect_subcat_row(r):
        mt = str(r.get("MTN_DELAY_TYPE", "")).strip().lower()
        sch = str(r.get("SCH_MTN", "")).strip().lower()
        unsch = str(r.get("UNSCH_MTN", "")).strip().lower()
        if mt == "unscheduled" or mt.startswith("uns") or unsch not in ("", "nan", "none"):
            return "Unscheduled"
        if mt == "scheduled" or mt.startswith("sch") or sch not in ("", "nan", "none"):
            return "Scheduled"
        return ""

    df["SUB_CATEGORY"] = df.apply(detect_subcat_row, axis=1)

    # CATEGORY determination
    def determine_category(r):
        if r.get("SUB_CATEGORY") in ("Scheduled", "Unscheduled"):
            return "Maintenance"
        if r.get("MINING_DELAY") and str(r.get("MINING_DELAY")).strip() != "":
            return "Mining"
        if r.get("WEATHER_DELAY") and str(r.get("WEATHER_DELAY")).strip() != "":
            return "Weather"
        if r.get("OTHER_DELAY") and str(r.get("OTHER_DELAY")).strip() != "":
            return "Other"
        if r.get("CATEGORY") and str(r.get("CATEGORY")).strip() not in ("", "nan"):
            return str(r.get("CATEGORY")).strip()
        return "Unknown"

    df["CATEGORY"] = df.apply(determine_category, axis=1)

    # Compose CAUSE for pareto fallback
    def compose_cause(r):
        parts = []
        for c in ["MTN_DELAY_TYPE", "SCH_MTN", "UNSCH_MTN", "MINING_DELAY", "WEATHER_DELAY", "OTHER_DELAY", "MTN_NOTE", "NOTE", "PICA"]: # ADDED 'PICA'
            v = r.get(c)
            if v is not None:
                vs = str(v).strip()
                if vs and vs.lower() != "nan":
                    parts.append(vs)
        return " | ".join(parts)

    df["CAUSE"] = df.apply(compose_cause, axis=1)

    # YEAR/WEEK numeric normalization
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce").astype("Int64")

    # Add PERIOD_MONTH if missing - THIS IS THE CORRECT LOGIC NOW
    if "PERIOD_MONTH" not in df.columns or df["PERIOD_MONTH"].isnull().all() or (df["PERIOD_MONTH"].astype(str).str.strip()=="").all():
        if "MONTH" in df.columns and "YEAR" in df.columns:
            df["PERIOD_MONTH"] = df["MONTH"].astype(str).str.strip() + " " + df["YEAR"].astype(str)

    if "PERIOD_MONTH" in df.columns:
        df["PERIOD_MONTH"] = df["PERIOD_MONTH"].astype(str).str.strip()

    # drop rows with no DELAY numeric
    df = df[df["DELAY"].notna()].copy()

    # allow data from 2023 onwards
    if "YEAR" in df.columns:
        df = df[df["YEAR"].notna() & (df["YEAR"] >= 2023)]

    # Prepare EQUIPMENT_DESC composite
    if "EQUIPMENT" in df.columns and df["EQUIPMENT"].notna().any():
        df["EQUIPMENT_DESC"] = df["EQUIPMENT"].replace("", "(Unknown)").astype(str) + " - " + df["EQ_DESC"].replace("", "(No Desc)").astype(str)
    else:
        df["EQUIPMENT_DESC"] = df["CAUSE"].replace("", "(Unknown)").astype(str)

    # DATE: normalize
    if "DATE" in df.columns:
        try:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date.astype("object").fillna("").astype(str)
        except Exception:
            df["DATE"] = df["DATE"].astype(str).replace("nan", "")
    else:
        df["DATE"] = ""

    # compute WEEK_START (ISO week Monday)
    def _compute_week_start(r):
        try:
            y = r.get("YEAR")
            w = r.get("WEEK")
            if pd.notna(y) and pd.notna(w):
                y_i = int(y)
                w_i = int(w)
                try:
                    return datetime.date.fromisocalendar(y_i, w_i, 1)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            d = pd.to_datetime(r.get("DATE"), errors="coerce")
            if pd.notna(d):
                iso = d.isocalendar()
                return datetime.date.fromisocalendar(int(iso.year), int(iso.week), 1)
        except Exception:
            pass
        return pd.NaT

    df["WEEK_START"] = df.apply(_compute_week_start, axis=1)
    
    return df

# -------------------------
# Load data
# -------------------------
df = load_data_from_url()
if df is None:
    st.stop()

# -------------------------
# NEW: compute MTBF & MTTR from "Data Operational" sheet (UPDATED to read from correct URL)
# -------------------------
@st.cache_data
def compute_operational_data(raw_url):
    """
    Reads the Data Operational sheet from the Excel file URL and computes all necessary metrics.
    """
    try:
        # Since we don't know the header location, we read the specific sheet from the URL
        df_op = pd.read_excel(raw_url, sheet_name="Data Operational")
    except Exception as e:
        return {"error": f"Could not read sheet 'Data Operational': {e}"}

    # Column Name Detection
    op_col, maint_col, date_col, unsched_col, sched_col = None, None, None, None, None
    for c in df_op.columns:
        lc = str(c).lower()
        if ("operat" in lc and "hour" in lc) or ("operational" in lc and "hour" in lc): op_col = c
        if "maint" in lc and "delay" in lc: maint_col = c
        if "unscheduled" in lc: unsched_col = c
        if "scheduled" in lc: sched_col = c
        if lc.strip() in ("date", "dates", "tanggal"): date_col = c
    
    if op_col is None:
        for c in df_op.columns:
            if "operat" in str(c).lower(): op_col = c; break
    if date_col is None:
        for c in df_op.columns:
            if "date" in str(c).lower(): date_col = c; break

    # Data Parsing and Cleaning
    def _parse_operational_hours(val):
        # (This function remains the same as your version)
        try:
            if pd.isna(val) or (isinstance(val, str) and str(val).strip() == ""): return float("nan")
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                v = float(val)
                return v * 24.0 if 0 < v < 1 else v
            s = str(val).strip()
            if ":" in s:
                try:
                    return pd.to_timedelta(s).total_seconds() / 3600.0
                except Exception: pass
            return float(s)
        except Exception:
            return float("nan")

    if op_col: df_op[op_col] = df_op[op_col].apply(_parse_operational_hours)
    if maint_col: df_op[maint_col] = pd.to_numeric(df_op[maint_col], errors='coerce').fillna(0)
    if unsched_col: df_op[unsched_col] = pd.to_numeric(df_op[unsched_col], errors='coerce').fillna(0)
    if sched_col: df_op[sched_col] = pd.to_numeric(df_op[sched_col], errors='coerce').fillna(0)
    
    if date_col:
        df_op["_date_parsed"] = pd.to_datetime(df_op[date_col], errors="coerce")
        iso_dates = df_op["_date_parsed"].dt.isocalendar()
        df_op['YEAR'] = iso_dates.year
        df_op['WEEK'] = iso_dates.week
    
    df_op['YEAR'] = df_op['YEAR'].fillna(0).astype(int)
    df_op['WEEK'] = df_op['WEEK'].fillna(0).astype(int)

    return {
        "raw_df_op": df_op,
        "op_col": op_col,
        "maint_col": maint_col,
        "unsched_col": unsched_col,
        "sched_col": sched_col
    }

# Execute data loading
_op_data_dict = compute_operational_data(RAW_URL)
RAW_DF_OP = _op_data_dict.get("raw_df_op") if isinstance(_op_data_dict, dict) else pd.DataFrame()
OP_COL_NAME = _op_data_dict.get("op_col")
MAINT_COL_NAME = _op_data_dict.get("maint_col")
UNSCHED_COL_NAME = _op_data_dict.get("unsched_col")

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters & Options")
st.sidebar.markdown("---")

# -------------------------
# Time granularity / Month / Year filters (UPDATED)
# -------------------------
# Use a format_func to change the display label without changing the underlying value
granularity = st.sidebar.selectbox(
    "Time granularity",
    options=["WEEK", "PERIOD_MONTH"],
    format_func=lambda x: 'Month' if x == 'PERIOD_MONTH' else 'Week',
    index=1
)

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
        m = str(r.get("MONTH","")).strip()
        if m == "":
            continue
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
if months_available:
    try:
        default_idx = len(months_available) - 1
    except Exception:
        default_idx = 0
else:
    default_idx = 0

months = ["All"] + months_available
selected_month = st.sidebar.selectbox("MONTH", months, index=0)

# Year multi-select global filter (fix previous TypeError by passing Python ints)
all_years = []
if "YEAR" in df.columns:
    try:
        yrs = sorted(list(pd.Series(df["YEAR"].dropna().astype(int).unique())), reverse=True)
        all_years = yrs
    except Exception:
        all_years = []
if all_years:
    # default select all (descending)
    selected_years = st.sidebar.multiselect("Year (multi-select, descending)", options=all_years, default=all_years)
else:
    selected_years = []

# -------------------------
# Apply selected filters to df (month & years)
# -------------------------
filtered = df.copy()
# The month filter is now only applied if granularity is MONTH
if granularity == 'PERIOD_MONTH' and selected_month != "All" and selected_month != "":
    filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]
if selected_years:
    if "YEAR" in filtered.columns:
        filtered = filtered[filtered["YEAR"].isin(selected_years)].copy()

# create tabs (Main shown implicitly; Reliability as second tab)
tabs = st.tabs(["Main Dashboard", "Reliability"])

# -------------------------
# START: MAIN DASHBOARD TAB CONTENT
# -------------------------
with tabs[0]:
    # -------------------------
    # KPI calculations for display on dashboard (UPDATED LOGIC)
    # -------------------------
    
    # Get the unique Year/Week combinations from the currently filtered data
    filtered_weeks = filtered[['YEAR', 'WEEK']].drop_duplicates()

    # Aggregate operational data for the relevant weeks
    op_df_agg = pd.DataFrame()
    if not RAW_DF_OP.empty and all(col in RAW_DF_OP.columns for col in [MAINT_COL_NAME, OP_COL_NAME]):
        op_df_agg = RAW_DF_OP.groupby(['YEAR', 'WEEK']).agg({
            MAINT_COL_NAME: 'sum',
            OP_COL_NAME: 'sum'
        }).reset_index()

        # Filter the aggregated operational data to match the filtered weeks
        merged_op_data = pd.merge(filtered_weeks, op_df_agg, on=['YEAR', 'WEEK'], how='left').fillna(0)
        
        total_maintenance_delay = merged_op_data[MAINT_COL_NAME].sum()
        total_operational_hours = merged_op_data[OP_COL_NAME].sum()
    else:
        total_maintenance_delay = 0
        total_operational_hours = 0
    
    # Calculate available_time from the 'Data Delay Time' sheet as before
    available_time = 0
    if "AVAILABLE_TIME_MONTH" in filtered.columns and filtered["AVAILABLE_TIME_MONTH"].notna().any():
         available_time = filtered.groupby("PERIOD_MONTH")["AVAILABLE_TIME_MONTH"].first().sum()

    # Calculate PA and MA with the new logic
    PA = (available_time - total_maintenance_delay) / available_time if available_time > 0 else 0
    denominator_ma = total_operational_hours + total_maintenance_delay
    MA = total_operational_hours / denominator_ma if denominator_ma > 0 else 0
    
    pa_target = filtered["PA_TARGET"].dropna().unique().tolist() if "PA_TARGET" in filtered.columns else []
    ma_target = filtered["MA_TARGET"].dropna().unique().tolist() if "MA_TARGET" in filtered.columns else []
    pa_target = pa_target[0] if pa_target else 0.9
    ma_target = ma_target[0] if ma_target else 0.85
    if isinstance(pa_target, (int, float)) and pa_target > 1: pa_target /= 100.0
    if isinstance(ma_target, (int, float)) and ma_target > 1: ma_target /= 100.0
    
    # -------------------------
    # YTD calculations (UPDATED LOGIC)
    # -------------------------
    ytd_PA = ytd_MA = None
    ytd_total_delay_for_display = None # This will now be maintenance delay

    try:
        latest_filtered = filtered if not filtered.empty else df
        if not latest_filtered.empty:
            latest_year = int(latest_filtered["YEAR"].dropna().max())
            df_period_dt = pd.to_datetime(df["PERIOD_MONTH"], format="%b %Y", errors="coerce")
            latest_period_dt = pd.to_datetime(latest_filtered["PERIOD_MONTH"], format="%b %Y", errors="coerce").max()
            
            ytd_df = df[(df["YEAR"] == latest_year) & (df_period_dt <= latest_period_dt)]
            
            ytd_available_time = 0
            if "AVAILABLE_TIME_MONTH" in ytd_df.columns and ytd_df["AVAILABLE_TIME_MONTH"].notna().any():
                ytd_available_time = ytd_df.groupby("PERIOD_MONTH")["AVAILABLE_TIME_MONTH"].first().sum()

            if not op_df_agg.empty:
                ytd_weeks = ytd_df[['YEAR', 'WEEK']].drop_duplicates()
                ytd_merged_op = pd.merge(ytd_weeks, op_df_agg, on=['YEAR', 'WEEK'], how='left').fillna(0)

                ytd_maintenance_delay = ytd_merged_op[MAINT_COL_NAME].sum()
                ytd_operational_hours = ytd_merged_op[OP_COL_NAME].sum()
            else:
                ytd_maintenance_delay = 0
                ytd_operational_hours = 0
            
            ytd_PA = (ytd_available_time - ytd_maintenance_delay) / ytd_available_time if ytd_available_time > 0 else 0
            ytd_ma_denom = ytd_operational_hours + ytd_maintenance_delay
            ytd_MA = ytd_operational_hours / ytd_ma_denom if ytd_ma_denom > 0 else 0
            ytd_total_delay_for_display = ytd_maintenance_delay
    except Exception:
        ytd_PA = ytd_MA = None
        ytd_total_delay_for_display = None


    # -------------------------
    # Top Row: KPIs + Donuts with PNG caching + Matplotlib fallback
    # -------------------------
    kpi_col, donut1_col, donut2_col = st.columns([1,2,2])
    with kpi_col:
        st.subheader("Key KPIs")
        # (This section remains unchanged, as it just displays the calculated values)
        min_caption, max_caption = None, None
        if "PERIOD_MONTH" in filtered.columns and not filtered["PERIOD_MONTH"].dropna().empty:
            parsed = pd.to_datetime(filtered["PERIOD_MONTH"].dropna().unique(), format="%b %Y", errors="coerce")
            if parsed.notna().any():
                min_dt, max_dt = parsed.min(), parsed.max()
                if pd.notna(min_dt) and pd.notna(max_dt):
                    min_caption = min_dt.strftime("%d/%m/%Y")
                    max_caption = (max_dt + pd.offsets.MonthEnd(0)).strftime("%d/%m/%Y")
        
        st.caption(f"Data from {min_caption} to {max_caption}" if min_caption and max_caption else "Data from an unknown date range")

        st.metric("Physical Availability (PA)", f"{PA:.2%}" if PA is not None else "N/A", delta=f"Target {pa_target:.2%}")
        st.metric("Mechanical Availability (MA)", f"{MA:.2%}" if MA is not None else "N/A", delta=f"Target {ma_target:.2%}")
        st.metric("Total Maintenance Delay (selected)", f"{total_maintenance_delay:.2f} hrs")
        st.metric("Total Available Time (selected)", f"{available_time:.2f} hrs" if available_time else "N/A")

        if ytd_PA is not None:
            st.write("")
            st.caption(f"YTD: PA {ytd_PA:.2%} | MA {ytd_MA:.2%} | Delay {ytd_total_delay_for_display:.2f} hrs")

    with donut1_col:
        st.subheader("Delay by Category")
        if "CATEGORY" in filtered.columns:
            donut_data = filtered.groupby("CATEGORY", dropna=False)["DELAY"].sum().reset_index().sort_values("DELAY", ascending=False)
            if not donut_data.empty:
                donut_data["DELAY"] = donut_data["DELAY"].round(2)
                donut_fig = go.Figure(data=[go.Pie(labels=donut_data["CATEGORY"], values=donut_data["DELAY"], hole=0.4, textinfo="label+percent", hovertemplate="%{label}: %{value:.2f} hrs<extra></extra>")])
                donut_fig.update_layout(margin=dict(t=20,b=20,l=20,r=20))
                # Store the Plotly figure in session state for potential PDF rendering
                st.session_state['pdf_fig_donut1'] = donut_fig
                st.plotly_chart(donut_fig, use_container_width=True)
            else:
                st.info("No delay data available.")
        else:
            st.info("No category data available.")

    with donut2_col:
        st.subheader("Scheduled vs Unscheduled (Maintenance only)")
        if "MTN_DELAY_TYPE" in filtered.columns:
            maint_df = filtered[filtered["CATEGORY"] == "Maintenance"].copy()
            if not maint_df.empty:
                sched_donut = maint_df.groupby("SUB_CATEGORY")["DELAY"].sum().reset_index().sort_values("DELAY", ascending=False)
                if not sched_donut.empty:
                    sched_donut["DELAY"] = sched_donut["DELAY"].round(2)
                    donut_fig2 = go.Figure(data=[go.Pie(labels=sched_donut["SUB_CATEGORY"], values=sched_donut["DELAY"], hole=0.4, textinfo="label+percent", hovertemplate="%{label}: %{value:.2f} hrs<extra></extra>")])
                    donut_fig2.update_layout(margin=dict(t=20,b=20,l=20,r=20))
                    st.session_state['pdf_fig_donut2'] = donut_fig2
                    st.plotly_chart(donut_fig2, use_container_width=True)
                else:
                    st.info("No maintenance breakdown available.")
            else:
                st.info("No maintenance data found in selection.")
        else:
            st.info("No MTN_DELAY_TYPE column available.")

    st.markdown("---")
    
    # -------------------------
    # Trend Analysis (UPDATED LOGIC for PA%)
    # -------------------------
    st.subheader("Trend: Total Delay Hours vs PA%")
    group_field = granularity
    
    # Create a separate dataframe for the trend chart to handle filtering logic
    if group_field == "WEEK":
        df_for_trend = df[df['YEAR'].isin(selected_years)].copy()
    else:
        df_for_trend = filtered.copy()

    # Determine date range for 52-week cutoff
    if group_field == "WEEK":
        latest_week_start = df["WEEK_START"].dropna().max()
        cutoff_date = latest_week_start - pd.Timedelta(weeks=51)
        df_for_trend = df_for_trend[df_for_trend["WEEK_START"] >= cutoff_date]

    # Aggregate data for the trend chart
    if group_field == "WEEK":
        trend = df_for_trend.groupby(["YEAR", "WEEK"]).agg(
            total_delay_hours=("DELAY", "sum"),
            available_time_month=("AVAILABLE_TIME_MONTH", "first")
        ).reset_index()
        trend["period_label"] = trend["YEAR"].astype(str) + " W" + trend["WEEK"].astype("Int64").astype(str)
        x_field = "period_label"
        # Merge operational data for PA calculation
        trend = pd.merge(trend, op_df_agg[[MAINT_COL_NAME, 'YEAR', 'WEEK']], on=['YEAR', 'WEEK'], how='left')

    elif group_field == "PERIOD_MONTH":
        trend = df_for_trend.groupby("PERIOD_MONTH").agg(
            total_delay_hours=("DELAY", "sum"),
            available_time_month=("AVAILABLE_TIME_MONTH", "first")
        ).reset_index()
        x_field = "PERIOD_MONTH"
        # For monthly, we need to aggregate operational data monthly
        op_monthly_agg = pd.merge(df[['YEAR', 'WEEK', 'PERIOD_MONTH']].drop_duplicates(), op_df_agg, on=['YEAR', 'WEEK'], how='left')
        op_monthly_agg = op_monthly_agg.groupby('PERIOD_MONTH')[MAINT_COL_NAME].sum().reset_index()
        trend = pd.merge(trend, op_monthly_agg, on='PERIOD_MONTH', how='left')
        
    trend = trend.sort_values(by=['YEAR', 'WEEK'] if group_field == "WEEK" else 'PERIOD_MONTH').reset_index(drop=True)

    # Calculate PA% with updated logic
    trend["PA_pct"] = None
    for idx, row in trend.iterrows():
        avail = row.get("available_time_month", 168) if group_field == "PERIOD_MONTH" else 168
        maint_delay = row.get(MAINT_COL_NAME, 0)
        if avail > 0:
            trend.at[idx, "PA_pct"] = (avail - maint_delay) / avail

    # Data formatting and plotting (remains the same)
    trend["PA_pct_rounded"] = trend["PA_pct"].round(4)
    trend["total_delay_hours_rounded"] = trend["total_delay_hours"].round(2)

    pa_threshold = pa_target if (pa_target is not None) else 0.9
    colors = ['red' if v < pa_threshold else 'green' for v in trend["PA_pct_rounded"].fillna(0)]

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=trend[x_field], y=trend["PA_pct_rounded"], name="PA%", marker=dict(color=colors), hovertemplate="%{y:.2%}<extra></extra>"))
    fig_trend.add_trace(go.Scatter(x=trend[x_field], y=trend["total_delay_hours_rounded"], name="Total Delay Hours", yaxis="y2", mode="lines+markers", hovertemplate="%{y:.2f} hrs<extra></extra>"))
    fig_trend.add_shape(type="line", x0=0, x1=1, xref="paper", y0=pa_target, y1=pa_target, yref="y", line=dict(color="green", dash="dash"))
    fig_trend.add_annotation(x=0, xref="paper", y=pa_target, yref="y", showarrow=False, text=f"PA Target {pa_target:.2%}", font=dict(color="green"), align="left", xanchor="left", yanchor="bottom")
    fig_trend.update_layout(xaxis_title="Period", yaxis=dict(title="PA%", overlaying=None, side="left", tickformat=".2%", range=[0,1]), yaxis2=dict(title="Delay Hours", overlaying="y", side="right"), legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom"), margin=dict(t=70))
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown("---")
    
    # (The rest of the script from Pareto onwards remains unchanged)
    # Pareto by Equipment
    st.subheader("Top Delay by Equipment (Pareto)")

    if "EQUIPMENT_DESC" in filtered.columns and filtered["EQUIPMENT_DESC"].notna().any():
        equipment_key = "EQUIPMENT_DESC"
        equipment_series = filtered["EQUIPMENT_DESC"].replace("", "(Unknown)")
    else:
        equipment_key = "CAUSE"
        equipment_series = filtered["CAUSE"].replace("", "(Unknown)")

    equipment_agg = (
        filtered.assign(_equip=equipment_series)
        .groupby("_equip", dropna=False)
        .agg(hours=("DELAY", "sum"))
        .reset_index()
        .rename(columns={"_equip": equipment_key})
        .sort_values("hours", ascending=False)
    )
    equipment_agg[equipment_key] = equipment_agg[equipment_key].fillna("(Unknown)").astype(str)
    equipment_agg["cum_hours"] = equipment_agg["hours"].cumsum()
    total_hours_sum = equipment_agg["hours"].sum()
    equipment_agg["cum_pct"] = equipment_agg["cum_hours"] / total_hours_sum if total_hours_sum > 0 else 0

    top_n = st.slider("Top N equipment to show", min_value=5, max_value=50, value=15)
    pareto_df = equipment_agg.head(top_n)

    fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
    fig_pareto.add_trace(go.Bar(x=pareto_df[equipment_key], y=pareto_df["hours"].round(2), name="Hours"), secondary_y=False)
    fig_pareto.add_trace(
        go.Scatter(
            x=pareto_df[equipment_key],
            y=pareto_df["cum_pct"],
            name="Cumulative %",
            mode="lines+markers",
            hovertemplate="%{y:.2%}<extra></extra>"
        ),
        secondary_y=True
    )
    fig_pareto.update_layout(xaxis_tickangle=-45, yaxis_title="Hours", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.15, yanchor="bottom"), margin=dict(t=110))
    fig_pareto.update_yaxes(title_text="Cumulative %", tickformat=".2%", range=[0, 1], secondary_y=True)
    fig_pareto.update_yaxes(title_text="Delay Hours", secondary_y=False)

    # Cache PNG for Pareto (Plotly -> PNG first; fallback to Matplotlib)
    png = _fig_to_png_bytes(fig_pareto)
    if not png:
        try:
            png = _mpl_png_pareto_from_df(pareto_df.rename(columns={equipment_key:"EQUIPMENT_DESC"}), equipment_key="EQUIPMENT_DESC", title="Top Delay by Equipment (Pareto) (fallback)")
        except Exception:
            png = None
    if png:
        st.session_state['pdf_fig_pareto'] = png
    st.plotly_chart(fig_pareto, use_container_width=True)

    # -------------------------
    # CATEGORY FILTER -> Drilldown (unchanged)
    # -------------------------
    st.subheader("Filter by Delay Category (affects drilldown table)")

    category_options = [
        "MAINTENANCE (ALL)",
        "MAINTENANCE - SCHEDULED",
        "MAINTENANCE - UNSCHEDULED",
        "MINING DELAY",
        "WEATHER DELAY",
        "OTHER DELAY",
    ]
    selected_category = st.selectbox("Select delay category", category_options, index=0)

    if selected_category == "MAINTENANCE (ALL)":
        drill_df_base = filtered[filtered["CATEGORY"] == "Maintenance"].copy()
    elif selected_category == "MAINTENANCE - SCHEDULED":
        drill_df_base = filtered[(filtered["CATEGORY"] == "Maintenance") & (filtered["SUB_CATEGORY"] == "Scheduled")].copy()
    elif selected_category == "MAINTENANCE - UNSCHEDULED":
        drill_df_base = filtered[(filtered["CATEGORY"] == "Maintenance") & (filtered["SUB_CATEGORY"] == "Unscheduled")].copy()
    elif selected_category == "MINING DELAY":
        drill_df_base = filtered[filtered["CATEGORY"] == "Mining"].copy()
    elif selected_category == "WEATHER DELAY":
        drill_df_base = filtered[filtered["CATEGORY"] == "Weather"].copy()
    elif selected_category == "OTHER DELAY":
        drill_df_base = filtered[filtered["CATEGORY"] == "Other"].copy()
    else:
        drill_df_base = filtered.copy()

    # -------------------------
    # Drill-down table (UPDATED with auto-resize and time formatting)
    # -------------------------
    st.subheader("Drill-down data (filtered by selected category)")
    
    # NEW: Add a checkbox for auto-resizing columns
    fit_columns = st.checkbox("Auto-resize columns to fit content", value=True)

    details_df = drill_df_base.copy()
    if "PERIOD_MONTH" in details_df.columns:
        details_df["MONTH"] = details_df["PERIOD_MONTH"]

    required_cols = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "EQ_DESC", "DELAY", "NOTE", "PICA", "SUB_CATEGORY", "YEAR"]
    for c in required_cols:
        if c not in details_df.columns:
            details_df[c] = ""
            
    # NEW: Format START and STOP columns to show only time
    details_df['START'] = pd.to_datetime(details_df['START'], errors='coerce').dt.strftime('%H:%M:%S').fillna('N/A')
    details_df['STOP'] = pd.to_datetime(details_df['STOP'], errors='coerce').dt.strftime('%H:%M:%S').fillna('N/A')

    details_out = details_df[["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "EQ_DESC", "DELAY", "NOTE", "PICA", "SUB_CATEGORY", "YEAR"]].copy()
    details_out = details_out.rename(columns={"EQ_DESC": "Equipment Description"})

    if selected_category == "MAINTENANCE (ALL)":
        ordered = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "SUB_CATEGORY", "Equipment Description", "DELAY", "NOTE", "PICA"]
    else:
        ordered = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "Equipment Description", "DELAY", "NOTE", "PICA"]

    ordered = [c for c in ordered if c in details_out.columns]
    details_out["WEEK"] = pd.to_numeric(details_out["WEEK"], errors="coerce")

    if "YEAR" in details_out.columns and details_out["YEAR"].notna().any():
        details_out["YEAR"] = pd.to_numeric(details_out["YEAR"], errors="coerce")
        details_out = details_out.sort_values(by=["YEAR", "WEEK", "START"], ascending=[False, False, True]).reset_index(drop=True)
        details_out = details_out.drop(columns=["YEAR"], errors="ignore")
    else:
        details_out = details_out.sort_values(by=["WEEK", "START"], ascending=[False, True]).reset_index(drop=True)

    details_out = details_out[ordered].reset_index(drop=True)

    def _round_maybe(x):
        try:
            if x is None or (isinstance(x, str) and str(x).strip() == ""):
                return x
            xf = float(x)
            return round(xf, 2)
        except Exception:
            return x

    if "DELAY" in details_out.columns:
        details_out["DELAY"] = details_out["DELAY"].apply(_round_maybe)

    # AgGrid display: configure defaults + auto-fit on grid load (fit_columns_on_grid_load True)
    gob2 = GridOptionsBuilder.from_dataframe(details_out)
    gob2.configure_grid_options(pagination=False)
    gob2.configure_default_column(editable=False, sortable=True, filter=True, resizable=True, wrapText=True, autoHeight=True)
    grid_options2 = gob2.build()

    AgGrid(
        details_out,
        gridOptions=grid_options2,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=False,
        height=600,
        fit_columns_on_grid_load=fit_columns, # Connects to the checkbox
        theme="balham"
    )

# -------------------------
# START: RELIABILITY TAB CONTENT
# -------------------------
with tabs[1]:
    # (The reliability tab remains unchanged from your original full script)
    st.subheader("Reliability: MTBF & MTTR")
    # ... and so on for the rest of your reliability tab code
# -------------------------
# START: RELIABILITY TAB CONTENT
# -------------------------
with tabs[1]:
    weekly_df_global = _mtbf_mttr_res.get("weekly_df") if isinstance(_mtbf_mttr_res, dict) else pd.DataFrame()
    monthly_df_global = _mtbf_mttr_res.get("monthly_df") if isinstance(_mtbf_mttr_res, dict) else pd.DataFrame()

    st.subheader("Reliability: MTBF & MTTR")

    weekly_df = weekly_df_global.copy() if isinstance(weekly_df_global, pd.DataFrame) else pd.DataFrame()
    monthly_df = monthly_df_global.copy() if isinstance(monthly_df_global, pd.DataFrame) else pd.DataFrame()

    # Apply year filter to reliability dataframes as well
    if not weekly_df.empty and selected_years:
        if "YEAR" in weekly_df.columns:
            weekly_df = weekly_df[weekly_df["YEAR"].isin(selected_years)]
    if not monthly_df.empty and selected_years:
        if "period_dt" in monthly_df.columns and monthly_df["period_dt"].notna().any():
            monthly_df = monthly_df[monthly_df["period_dt"].dt.year.isin(selected_years)]
        else:
            def _pm_year(s):
                try:
                    return int(str(s).split()[-1])
                except Exception:
                    return None
            monthly_df["_yr"] = monthly_df["PERIOD_MONTH"].apply(_pm_year)
            monthly_df = monthly_df[monthly_df["_yr"].isin(selected_years)]
            monthly_df = monthly_df.drop(columns=["_yr"], errors="ignore")

    # MTTR
    st.markdown("### MTTR (Mean Time To Repair)")
    cols_mttr = st.columns(2)
    # Weekly MTTR
    with cols_mttr[0]:
        st.markdown("**MTTR — Weekly**")
        if weekly_df.empty:
            st.info("No weekly reliability data available.")
        else:
            try:
                if "week_start" in weekly_df.columns and weekly_df["week_start"].notna().any():
                    latest_ws = weekly_df["week_start"].dropna().max()
                    cutoff = latest_ws - pd.Timedelta(weeks=51)
                    weekly_df_limited = weekly_df[pd.to_datetime(weekly_df["week_start"]) >= pd.to_datetime(cutoff)].copy()
                else:
                    weekly_df_limited = weekly_df.copy()
            except Exception:
                weekly_df_limited = weekly_df.copy()
            if weekly_df_limited.empty:
                st.info("No weekly reliability data in selected years / range.")
            else:
                # ensure ascending order for display
                if "week_start" in weekly_df_limited.columns:
                    weekly_df_limited = weekly_df_limited.sort_values("week_start")
                fig_mttr_w = go.Figure()
                fig_mttr_w.add_trace(go.Bar(x=weekly_df_limited["period_label"], y=weekly_df_limited["MTTR_hours"].round(2), name="MTTR (hrs)", marker=dict(color="green")))
                fig_mttr_w.update_layout(xaxis_title="Week", yaxis_title="MTTR (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                # cache PNG: try Plotly -> PNG, fallback to Matplotlib using weekly_df_limited
                png = _fig_to_png_bytes(fig_mttr_w)
                if not png:
                    png = _mpl_png_bar_from_df(weekly_df_limited, x_col="period_label", y_col="MTTR_hours", title="MTTR — Weekly", color="green", xlabel="Week", ylabel="MTTR (hrs)")
                if png:
                    st.session_state['pdf_fig_mttr_w'] = png
                st.plotly_chart(fig_mttr_w, use_container_width=True)

    # Monthly MTTR
    with cols_mttr[1]:
        st.markdown("**MTTR — Monthly**")
        if monthly_df.empty:
            st.info("No monthly reliability data available.")
        else:
            monthly_df_local = monthly_df.copy()
            if "PERIOD_MONTH" in monthly_df_local.columns:
                monthly_df_local = monthly_df_local.sort_values(by="period_dt" if "period_dt" in monthly_df_local.columns else "PERIOD_MONTH", ascending=True)
                fig_mttr_m = go.Figure()
                fig_mttr_m.add_trace(go.Bar(x=monthly_df_local["PERIOD_MONTH"], y=monthly_df_local["MTTR_hours"].round(2), name="MTTR (hrs)", marker=dict(color="green")))
                fig_mttr_m.update_layout(xaxis_title="Month", yaxis_title="MTTR (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                png = _fig_to_png_bytes(fig_mttr_m)
                if not png:
                    png = _mpl_png_bar_from_df(monthly_df_local, x_col="PERIOD_MONTH", y_col="MTTR_hours", title="MTTR — Monthly", color="green", xlabel="Month", ylabel="MTTR (hrs)")
                if png:
                    st.session_state['pdf_fig_mttr_m'] = png
                st.plotly_chart(fig_mttr_m, use_container_width=True)
            else:
                st.info("No PERIOD_MONTH column in monthly reliability data.")

    st.markdown("---")
    st.markdown("### MTBF (Mean Time Between Failures)")
    cols_mtbf = st.columns(2)
    # Weekly MTBF
    with cols_mtbf[0]:
        st.markdown("**MTBF — Weekly**")
        if weekly_df.empty:
            st.info("No weekly reliability data available.")
        else:
            try:
                if "week_start" in weekly_df.columns and weekly_df["week_start"].notna().any():
                    latest_ws = weekly_df["week_start"].dropna().max()
                    cutoff = latest_ws - pd.Timedelta(weeks=51)
                    weekly_df_limited = weekly_df[pd.to_datetime(weekly_df["week_start"]) >= pd.to_datetime(cutoff)].copy()
                else:
                    weekly_df_limited = weekly_df.copy()
            except Exception:
                weekly_df_limited = weekly_df.copy()
            if weekly_df_limited.empty:
                st.info("No weekly reliability data in selected years / range.")
            else:
                if "week_start" in weekly_df_limited.columns:
                    weekly_df_limited = weekly_df_limited.sort_values("week_start")
                fig_mtbf_w = go.Figure()
                fig_mtbf_w.add_trace(go.Bar(x=weekly_df_limited["period_label"], y=weekly_df_limited["MTBF_hours"].round(2), name="MTBF (hrs)", marker=dict(color="orange")))
                fig_mtbf_w.update_layout(xaxis_title="Week", yaxis_title="MTBF (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                png = _fig_to_png_bytes(fig_mtbf_w)
                if not png:
                    png = _mpl_png_bar_from_df(weekly_df_limited, x_col="period_label", y_col="MTBF_hours", title="MTBF — Weekly", color="orange", xlabel="Week", ylabel="MTBF (hrs)")
                if png:
                    st.session_state['pdf_fig_mtbf_w'] = png
                st.plotly_chart(fig_mtbf_w, use_container_width=True)

    # Monthly MTBF
    with cols_mtbf[1]:
        st.markdown("**MTBF — Monthly**")
        if monthly_df.empty:
            st.info("No monthly reliability data available.")
        else:
            monthly_df_local = monthly_df.copy()
            if "PERIOD_MONTH" in monthly_df_local.columns:
                monthly_df_local = monthly_df_local.sort_values(by="period_dt" if "period_dt" in monthly_df_local.columns else "PERIOD_MONTH", ascending=True)
                fig_mtbf_m = go.Figure()
                fig_mtbf_m.add_trace(go.Bar(x=monthly_df_local["PERIOD_MONTH"], y=monthly_df_local["MTBF_hours"].round(2), name="MTBF (hrs)", marker=dict(color="orange")))
                fig_mtbf_m.update_layout(xaxis_title="Month", yaxis_title="MTBF (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                png = _fig_to_png_bytes(fig_mtbf_m)
                if not png:
                    png = _mpl_png_bar_from_df(monthly_df_local, x_col="PERIOD_MONTH", y_col="MTBF_hours", title="MTBF — Monthly", color="orange", xlabel="Month", ylabel="MTTR (hrs)")
                if png:
                    st.session_state['pdf_fig_mtbf_m'] = png
                st.plotly_chart(fig_mtbf_m, use_container_width=True)
            else:
                st.info("No PERIOD_MONTH column in monthly reliability data.")

    st.markdown("---")
    
    # -------------------------
    # MTTR & MTBF per Equipment Table
    # -------------------------
    st.subheader("MTTR & MTBF per Equipment")
    st.markdown(f"View by: **{granularity.replace('_', ' ').title()}** (change in sidebar)")

    df_delay_time = filtered.copy()
    df_op = RAW_DF_OP.copy() if isinstance(RAW_DF_OP, pd.DataFrame) else pd.DataFrame()

    if df_op.empty or df_delay_time.empty:
        st.info("Data for calculating per-equipment reliability metrics is unavailable.")
    else:
        df_maint_delays = df_delay_time[df_delay_time['CATEGORY'] == 'Maintenance'].copy()
        df_maint_delays = df_maint_delays[~df_maint_delays['EQUIPMENT_DESC'].str.strip().isin(['-', '---', ''])]

        delay_agg = df_maint_delays.groupby(['YEAR', 'WEEK', 'PERIOD_MONTH', 'EQUIPMENT_DESC']).agg(
            total_maint_delay_hours=('DELAY', 'sum'),
            maint_event_count=('DELAY', 'size')
        ).reset_index()

        op_agg = df_op.groupby(['YEAR', 'WEEK']).agg(
            total_operational_hours=('_op_hours_dec', 'sum')
        ).reset_index()

        # Use 'inner' merge to only include weeks present in BOTH datasets.
        merged_data = pd.merge(delay_agg, op_agg, on=['YEAR', 'WEEK'], how='inner')

        if granularity == 'PERIOD_MONTH':
            monthly_agg = merged_data.groupby(['YEAR', 'PERIOD_MONTH', 'EQUIPMENT_DESC']).agg(
                total_maint_delay_hours=('total_maint_delay_hours', 'sum'),
                maint_event_count=('maint_event_count', 'sum'),
                total_operational_hours=('total_operational_hours', 'sum')
            ).reset_index()

            monthly_agg['MTTR'] = monthly_agg.apply(
                lambda row: row['total_maint_delay_hours'] / row['maint_event_count'] if row['maint_event_count'] > 0 else 0, axis=1
            )
            monthly_agg['MTBF'] = monthly_agg.apply(
                lambda row: row['total_operational_hours'] / row['maint_event_count'] if row['maint_event_count'] > 0 else 0, axis=1
            )
            
            display_df = monthly_agg.copy()
            display_df['MONTH'] = pd.to_datetime(display_df['PERIOD_MONTH'], format='%b %Y', errors='coerce').dt.strftime('%b')
            display_df = display_df[['YEAR', 'MONTH', 'EQUIPMENT_DESC', 'MTTR', 'MTBF']]
            display_df.rename(columns={'EQUIPMENT_DESC': 'EQUIPMENT'}, inplace=True)
            
            sort_key = pd.to_datetime(monthly_agg['PERIOD_MONTH'], format='%b %Y', errors='coerce')
            display_df = display_df.iloc[sort_key.argsort()[::-1]]

        else: # Default to WEEK view
            merged_data['MTTR'] = merged_data.apply(
                lambda row: row['total_maint_delay_hours'] / row['maint_event_count'] if row['maint_event_count'] > 0 else 0, axis=1
            )
            merged_data['MTBF'] = merged_data.apply(
                lambda row: row['total_operational_hours'] / row['maint_event_count'] if row['maint_event_count'] > 0 else 0, axis=1
            )
            
            display_df = merged_data.copy()
            display_df['MONTH'] = pd.to_datetime(display_df['PERIOD_MONTH'], format='%b %Y', errors='coerce').dt.strftime('%b')
            display_df = display_df[['YEAR', 'MONTH', 'WEEK', 'EQUIPMENT_DESC', 'MTTR', 'MTBF']]
            display_df.rename(columns={'EQUIPMENT_DESC': 'EQUIPMENT'}, inplace=True)
            display_df = display_df.sort_values(by=['YEAR', 'WEEK'], ascending=[False, False])

        display_df['MTTR'] = display_df['MTTR'].round(2)
        display_df['MTBF'] = display_df['MTBF'].round(2)
        display_df = display_df.reset_index(drop=True)

        if not display_df.empty:
            gb = GridOptionsBuilder.from_dataframe(display_df)
            
            # --- THIS IS THE ONLY CHANGE IN THIS UPDATE ---
            # Automatically resize columns to fit header text
            gb.configure_grid_options(autoSizeStrategy=dict(type='fitGridWidth'))
            
            gb.configure_default_column(editable=False, sortable=True, filter=True, resizable=True)
            grid_options = gb.build()

            AgGrid(
                display_df,
                gridOptions=grid_options,
                height=500,
                # The fit_columns_on_grid_load parameter is superseded by autoSizeStrategy
                theme="balham",
                key=f'aggrid_{granularity}'
            )
        else:
            st.info("No matching reliability data found for the selected filters. Ensure both 'Data Delay Time' and 'Data Operational' have entries for the same weeks.")

# END: RELIABILITY TAB CONTENT
# -------------------------
