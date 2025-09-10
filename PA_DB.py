# Full updated script with PDF chart caching (kaleido -> html2image fallback).
# Important notes:
#  - This script expects your original workbook and columns unchanged.
#  - Required libraries (for PDF export with charts):
#      - reportlab (for PDF building)
#      - kaleido (preferred for plotly image export)
#      - html2image (fallback; requires an installed headless Chromium/Chrome)
#      - Pillow (already used by HTML -> image processing in some flows)
#    If you don't want PDF export, the dashboard still runs without them.
#
#  - I placed debug-friendly inline comments (# UPDATED / # DEBUG) to help trace behavior.
#
#  - Tabs have been placed near the top. The main dashboard content is rendered under the "Main Dashboard" tab.
#
#  - Chart caching: save_and_cache_fig_for_pdf(key, fig) will:
#      1) attempt to convert fig -> PNG (kaleido first, then html2image)
#      2) store PNG bytes in st.session_state[key]
#      3) display the figure on the dashboard as normal
#
#  - When generating PDF, the app collects cached PNGs from session_state and builds a simple ReportLab PDF.
#
# Please test locally first. If charts are not showing in PDF, check kaleido and/or html2image environment.

import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tempfile
import os
import shutil
from io import BytesIO

# AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# PDF and image helpers
try:
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import inch
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False
    # Keep names defined to avoid NameErrors
    SimpleDocTemplate = Paragraph = Spacer = RLImage = None
    getSampleStyleSheet = A4 = inch = None

# Try kaleido availability (plotly -> image)
KALEIDO_AVAILABLE = True
try:
    # Attempt a tiny call to check if kaleido is importable by plotly
    import plotly
    # No explicit import needed; fig.to_image will raise if missing
except Exception:
    KALEIDO_AVAILABLE = False

# Try html2image fallback (requires headless browser available in environment)
HTML2IMAGE_AVAILABLE = True
try:
    from html2image import Html2Image
except Exception:
    HTML2IMAGE_AVAILABLE = False

# Helper: convert a Plotly figure to PNG bytes.
# UPDATED: Try kaleido first; if that fails, fall back to html2image (if available).
def _fig_to_png_bytes(fig, width=1200, height=700):
    """
    Convert a plotly figure to PNG bytes.
    Returns bytes or None on failure.
    """
    # Accept already-bytes
    if isinstance(fig, (bytes, bytearray)):
        return bytes(fig)
    # Try direct plotly -> png using kaleido (preferred)
    try:
        # plotly >= 5: fig.to_image(format='png') uses kaleido if installed.
        img_bytes = fig.to_image(format="png")
        if isinstance(img_bytes, (bytes, bytearray)):
            return bytes(img_bytes)
    except Exception:
        # KALEIDO may not be available or fail — continue to fallback
        pass

    # Fallback: use html2image to render fig.to_html -> screenshot
    if HTML2IMAGE_AVAILABLE:
        try:
            html_str = fig.to_html(full_html=True, include_plotlyjs="cdn")
            hti = Html2Image(size=(width, height))
            # Create a temporary directory to save screenshot
            tmpdir = tempfile.mkdtemp(prefix="hti_")
            out_path = os.path.join(tmpdir, "fig.png")
            # html2image.screenshot accepts html_str parameter
            hti.screenshot(html_str=html_str, save_as=out_path)
            with open(out_path, "rb") as f:
                data = f.read()
            # cleanup
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
            return data
        except Exception:
            # final fallback: return None
            try:
                shutil.rmtree(tmpdir)
            except Exception:
                pass
            return None
    # No method succeeded
    return None

# UPDATED: helper that both displays and caches PNG bytes for PDF inclusion
def save_and_cache_fig_for_pdf(session_key_png, fig, width=1200, height=700):
    """
    Display fig with st.plotly_chart and cache PNG bytes in st.session_state[session_key_png].
    - session_key_png: key in st.session_state to store PNG bytes (e.g., 'pdf_fig_trend')
    """
    # attempt conversion; store bytes if successful
    png = None
    try:
        png = _fig_to_png_bytes(fig, width=width, height=height)
    except Exception:
        png = None
    if png:
        st.session_state[session_key_png] = png
    # Always display the figure to the user
    st.plotly_chart(fig, use_container_width=True)

# Initialize session_state slots for cached PNGs (used by PDF)
for k in ['pdf_fig_trend','pdf_fig_pareto','pdf_fig_mttr_w','pdf_fig_mttr_m','pdf_fig_mtbf_w','pdf_fig_mtbf_m']:
    if k not in st.session_state:
        st.session_state[k] = None

# Page config
st.set_page_config(page_title="Physical Availability - Data Delay Time", layout="wide")

# -------------------------
# Auto-hide sidebar after 10 seconds of no interaction
# -------------------------
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
        // Reset timer on common user interactions:
        ['mousemove','mousedown','keydown','touchstart','scroll'].forEach(evt => {
            document.addEventListener(evt, resetTimer, {passive:true});
        });
        // also start on load
        window.addEventListener('load', resetTimer);
    })();
    </script>
    """,
    unsafe_allow_html=True,
)

# Logo + Title
LOGO_URL = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/refs/heads/main/images/alamtri_logo.jpeg"
logo_col, title_col = st.columns([1, 8])
with logo_col:
    st.image(LOGO_URL, width=150)
with title_col:
    st.title("Physical Availability Dashboard — Data Delay Time")

# -------------------------
# Config: source workbook URL
# -------------------------
RAW_URL = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/refs/heads/main/Draft_New%20Version_Weekly_Report_Maintenance_CHPP.xlsx"

# -------------------------
# Load + clean function (Data Delay Time sheet)
# -------------------------
@st.cache_data(ttl=600)
def load_data_from_url():
    try:
        raw = pd.read_excel(RAW_URL, sheet_name="Data Delay Time", header=None)
    except Exception as e:
        st.error(f"Unable to read sheet 'Data Delay Time': {e}")
        return None

    # Detect header row (header starts in first 20 rows containing WEEK/MONTH/YEAR)
    header_row = None
    for i in range(20):
        row_values = raw.iloc[i].astype(str).str.upper().tolist()
        if "WEEK" in row_values or "MONTH" in row_values or "YEAR" in row_values:
            header_row = i
            break
    if header_row is None:
        st.error("Could not detect header row automatically. Please check the Excel file format.")
        return None

    df = pd.read_excel(RAW_URL, sheet_name="Data Delay Time", header=header_row)
    df.columns = [str(c).strip() for c in df.columns]

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

    df["DELAY"] = pd.to_numeric(df["DELAY"], errors="coerce")
    if "AVAILABLE_TIME_MONTH" in df.columns:
        df["AVAILABLE_TIME_MONTH"] = pd.to_numeric(df["AVAILABLE_TIME_MONTH"], errors="coerce")
    else:
        df["AVAILABLE_TIME_MONTH"] = None

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

    for cat in ["MTN_DELAY_TYPE", "SCH_MTN", "UNSCH_MTN", "MINING_DELAY", "WEATHER_DELAY", "OTHER_DELAY", "MTN_NOTE", "NOTE", "EQ_DESC", "EQUIPMENT", "CATEGORY", "PERIOD_MONTH", "DATE"]:
        if cat in df.columns:
            df[cat] = df[cat].fillna("").astype(str).str.strip()
        else:
            df[cat] = ""

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

    def compose_cause(r):
        parts = []
        for c in ["MTN_DELAY_TYPE", "SCH_MTN", "UNSCH_MTN", "MINING_DELAY", "WEATHER_DELAY", "OTHER_DELAY", "MTN_NOTE", "NOTE"]:
            v = r.get(c)
            if v is not None:
                vs = str(v).strip()
                if vs and vs.lower() != "nan":
                    parts.append(vs)
        return " | ".join(parts)
    df["CAUSE"] = df.apply(compose_cause, axis=1)

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce").astype("Int64")

    if "PERIOD_MONTH" not in df.columns or df["PERIOD_MONTH"].isnull().all() or (df["PERIOD_MONTH"].astype(str).str.strip()=="").all():
        if "MONTH" in df.columns and "YEAR" in df.columns:
            df["PERIOD_MONTH"] = df["MONTH"].astype(str).str.strip() + " " + df["YEAR"].astype(str)

    if "PERIOD_MONTH" in df.columns:
        df["PERIOD_MONTH"] = df["PERIOD_MONTH"].astype(str).str.strip()

    df = df[df["DELAY"].notna()].copy()

    # Allow 2023 onwards
    if "YEAR" in df.columns:
        df = df[df["YEAR"].notna() & (df["YEAR"] >= 2023)]

    if "EQUIPMENT" in df.columns and df["EQUIPMENT"].notna().any():
        df["EQUIPMENT_DESC"] = df["EQUIPMENT"].replace("", "(Unknown)").astype(str) + " - " + df["EQ_DESC"].replace("", "(No Desc)").astype(str)
    else:
        df["EQUIPMENT_DESC"] = df["CAUSE"].replace("", "(Unknown)").astype(str)

    if "DATE" in df.columns:
        try:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date.astype("object").fillna("").astype(str)
        except Exception:
            df["DATE"] = df["DATE"].astype(str).replace("nan", "")
    else:
        df["DATE"] = ""

    # Compute WEEK_START (ISO Monday) to support recent 52-week limiting
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

    # Assign each ISO-week to the month containing the week end (latest month of that week)
    try:
        week_start_dt = pd.to_datetime(df["WEEK_START"], errors="coerce")
        week_end_dt = week_start_dt + pd.Timedelta(days=6)
        df.loc[week_start_dt.notna(), "PERIOD_MONTH"] = week_end_dt.dt.strftime("%b %Y")
    except Exception:
        pass

    return df

# -------------------------
# Load data
# -------------------------
df = load_data_from_url()
if df is None:
    st.stop()

# -------------------------
# NEW: compute MTBF & MTTR from "Data Operational" sheet
# -------------------------
@st.cache_data(ttl=600)
def compute_mtbf_mttr_from_url(raw_url):
    """
    Reads the 'Data Operational' sheet and computes per-week and per-month MTBF/MTTR.
    Uses count of rows as denominator (user requested).
    Returns dict containing weekly_df and monthly_df among other metadata.
    """
    try:
        xls = pd.ExcelFile(raw_url)
    except Exception as e:
        return {"error": f"Could not open workbook: {e}"}

    # find sheet name
    sheet_name = None
    for s in xls.sheet_names:
        if s.strip().lower() == "data operational":
            sheet_name = s
            break
    if sheet_name is None:
        for s in xls.sheet_names:
            if "operational" in s.strip().lower():
                sheet_name = s
                break
    if sheet_name is None:
        return {"error": "Data Operational sheet not found"}

    try:
        df_op = pd.read_excel(raw_url, sheet_name=sheet_name)
    except Exception as e:
        return {"error": f"Could not read sheet '{sheet_name}': {e}"}

    # detect OPERATIONAL HOURS and MAINTENANCE DELAY columns
    op_col = None
    maint_col = None
    date_col = None
    for c in df_op.columns:
        lc = str(c).lower()
        if ("operat" in lc and "hour" in lc) or ("operational" in lc and "hour" in lc):
            op_col = c
            break
    if op_col is None:
        for c in df_op.columns:
            lc = str(c).lower()
            if "operat" in lc:
                op_col = c
                break

    for c in df_op.columns:
        lc = str(c).lower()
        if "maint" in lc and "delay" in lc:
            maint_col = c
            break
    if maint_col is None:
        for c in df_op.columns:
            lc = str(c).lower()
            if "maintenance" in lc and "delay" in lc:
                maint_col = c
                break

    # detect date-like column for grouping
    for c in df_op.columns:
        lc = str(c).lower()
        if lc.strip() in ("date", "dates", "tanggal"):
            date_col = c
            break
    if date_col is None:
        for c in df_op.columns:
            lc = str(c).lower()
            if "date" in lc or "week" in lc or "year" in lc:
                date_col = c
                break

    # parse operational hours to decimal hours
    def _parse_operational_hours(val):
        try:
            if pd.isna(val) or (isinstance(val, str) and str(val).strip() == ""):
                return float("nan")
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                v = float(val)
                if 0 < v < 1:
                    return v * 24.0
                return v
            s = str(val).strip()
            if ":" in s:
                try:
                    td = pd.to_timedelta(s)
                    return td.total_seconds() / 3600.0
                except Exception:
                    pass
            try:
                return float(s)
            except Exception:
                return float("nan")
        except Exception:
            return float("nan")

    if op_col is not None:
        df_op["_op_hours_dec"] = df_op[op_col].apply(_parse_operational_hours)
    else:
        df_op["_op_hours_dec"] = pd.Series([float("nan")] * len(df_op))

    if maint_col is not None:
        df_op["_maint_delay_num"] = pd.to_numeric(df_op[maint_col], errors="coerce")
    else:
        df_op["_maint_delay_num"] = pd.Series([float("nan")] * len(df_op))

    # parse date column if possible
    if date_col is not None:
        try:
            df_op["_date_parsed"] = pd.to_datetime(df_op[date_col], errors="coerce")
        except Exception:
            df_op["_date_parsed"] = pd.Series([pd.NaT] * len(df_op))
    else:
        df_op["_date_parsed"] = pd.Series([pd.NaT] * len(df_op))

    # YEAR/WEEK via ISO calendar
    def _from_date_to_iso(y):
        try:
            if pd.isna(y):
                return (None, None)
            d = pd.to_datetime(y)
            iso = d.isocalendar()
            return (int(iso.year), int(iso.week))
        except Exception:
            return (None, None)

    df_op["_yr_wk"] = df_op["_date_parsed"].apply(lambda r: _from_date_to_iso(r))
    df_op["YEAR"] = df_op["_yr_wk"].apply(lambda t: t[0] if isinstance(t, tuple) else None)
    df_op["WEEK"] = df_op["_yr_wk"].apply(lambda t: t[1] if isinstance(t, tuple) else None)

    # WEEK_START and PERIOD_MONTH (use week-end month)
    try:
        # Week start using Period W (ISO)
        df_op["WEEK_START"] = df_op["_date_parsed"].dt.to_period("W").apply(lambda p: p.start_time.date())
        ws = pd.to_datetime(df_op["WEEK_START"], errors="coerce")
        we = ws + pd.Timedelta(days=6)
        df_op["PERIOD_MONTH"] = we.dt.strftime("%b %Y")
    except Exception:
        df_op["PERIOD_MONTH"] = df_op["_date_parsed"].dt.strftime("%b %Y")

    # Group weekly: sum hours, sum delay, count rows
    try:
        weekly_group = df_op.groupby(["YEAR", "WEEK"], dropna=False).agg(
            total_operational_hours=("_op_hours_dec", "sum"),
            total_maintenance_delay=("_maint_delay_num", "sum"),
            count_rows=("_op_hours_dec", "count")
        ).reset_index()
    except Exception:
        # fallback
        weekly_group = df_op.groupby(["YEAR", "WEEK"], dropna=False).agg(
            total_operational_hours=("_op_hours_dec", "sum"),
            total_maintenance_delay=("_maint_delay_num", "sum"),
            count_rows=("_op_hours_dec", "count")
        ).reset_index()

    weekly_group["MTBF_hours"] = weekly_group.apply(lambda r: (r["total_operational_hours"] / r["count_rows"]) if (r["count_rows"] and r["count_rows"] > 0) else float("nan"), axis=1)
    weekly_group["MTTR_hours"] = weekly_group.apply(lambda r: (r["total_maintenance_delay"] / r["count_rows"]) if (r["count_rows"] and r["count_rows"] > 0) else float("nan"), axis=1)

    def _wk_label_from_row(r):
        try:
            y = int(r["YEAR"])
            w = int(r["WEEK"])
            return f"{y} W{int(w)}"
        except Exception:
            return ""
    weekly_group["period_label"] = weekly_group.apply(_wk_label_from_row, axis=1)

    def _wk_start(r):
        try:
            return datetime.date.fromisocalendar(int(r["YEAR"]), int(r["WEEK"]), 1)
        except Exception:
            return pd.NaT
    weekly_group["week_start"] = weekly_group.apply(_wk_start, axis=1)
    weekly_group = weekly_group.sort_values("week_start").reset_index(drop=True)

    # Monthly grouping
    monthly_group = df_op.groupby("PERIOD_MONTH", dropna=False).agg(
        total_operational_hours=("_op_hours_dec", "sum"),
        total_maintenance_delay=("_maint_delay_num", "sum"),
        count_rows=("_op_hours_dec", "count")
    ).reset_index()
    monthly_group["MTBF_hours"] = monthly_group.apply(lambda r: (r["total_operational_hours"] / r["count_rows"]) if (r["count_rows"] and r["count_rows"] > 0) else float("nan"), axis=1)
    monthly_group["MTTR_hours"] = monthly_group.apply(lambda r: (r["total_maintenance_delay"] / r["count_rows"]) if (r["count_rows"] and r["count_rows"] > 0) else float("nan"), axis=1)
    try:
        monthly_group["period_dt"] = pd.to_datetime(monthly_group["PERIOD_MONTH"], format="%b %Y", errors="coerce")
    except Exception:
        monthly_group["period_dt"] = pd.NaT
    monthly_group = monthly_group.sort_values("period_dt").reset_index(drop=True)

    return {
        "sheet_name": sheet_name,
        "operational_hours_column": op_col,
        "maintenance_delay_column": maint_col,
        "date_column": date_col,
        "weekly_df": weekly_group,
        "monthly_df": monthly_group,
        "total_rows": len(df_op),
        "total_operational_hours": float(df_op["_op_hours_dec"].sum(skipna=True)),
        "total_maintenance_delay": float(df_op["_maint_delay_num"].sum(skipna=True)),
        "MTBF_hours_overall": (float(df_op["_op_hours_dec"].sum(skipna=True)) / len(df_op)) if len(df_op) > 0 else None,
        "MTTR_hours_overall": (float(df_op["_maint_delay_num"].sum(skipna=True)) / len(df_op)) if len(df_op) > 0 else None
    }

# call the MTBF/MTTR computation (cached)
_mtbf_mttr_res = compute_mtbf_mttr_from_url(RAW_URL)

# expose simple globals for potential display (unchanged from earlier)
MTBF_GLOBAL_HOURS = _mtbf_mttr_res.get("MTBF_hours_overall") if isinstance(_mtbf_mttr_res, dict) else None
MTTR_GLOBAL_HOURS = _mtbf_mttr_res.get("MTTR_hours_overall") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_GLOBAL_HOURS_ROUNDED = round(MTBF_GLOBAL_HOURS, 2) if (MTBF_GLOBAL_HOURS is not None) else None
MTTR_GLOBAL_HOURS_ROUNDED = round(MTTR_GLOBAL_HOURS, 2) if (MTTR_GLOBAL_HOURS is not None) else None

# -------------------------
# Sidebar Filters & PDF Export UI
# -------------------------
st.sidebar.header("Filters & Options")

# PDF Export UI (button) - we'll create the PDF bytes only when the user clicks
st.sidebar.markdown("---")
st.sidebar.subheader("Export report (PDF)")

# Note: KPI text for PDF is composed at the moment of generation (to ensure it uses current selections)
if REPORTLAB_AVAILABLE:
    if st.sidebar.button("Generate PDF"):
        # Build KPI header text dynamically (safe)
        try:
            kpi_text = ""
            try:
                kpi_text += f"PA: {PA:.2%}\n"
            except Exception:
                kpi_text += "PA: N/A\n"
            try:
                kpi_text += f"MA: {MA:.2%}\n"
            except Exception:
                kpi_text += "MA: N/A\n"
            try:
                kpi_text += f"Total Delay (selected): {total_delay:.2f} hrs\n"
            except Exception:
                kpi_text += "Total Delay (selected): N/A\n"
            try:
                if available_time:
                    kpi_text += f"Total Available Time (selected): {available_time:.2f} hrs\n"
            except Exception:
                pass
        except Exception:
            kpi_text = ""
        # Collect cached PNGs (in display order)
        figs_bytes = []
        for key in ['pdf_fig_trend','pdf_fig_pareto','pdf_fig_mttr_w','pdf_fig_mttr_m','pdf_fig_mtbf_w','pdf_fig_mtbf_m']:
            v = st.session_state.get(key)
            if v:
                figs_bytes.append(v)
        # Build PDF (simple)
        def _build_pdf_bytes(title, kpi_text, png_bytes_list):
            if not REPORTLAB_AVAILABLE:
                return None
            elements = []
            styles = getSampleStyleSheet() if getSampleStyleSheet else None
            if styles:
                title_style = styles.get("Heading1")
                normal_style = styles.get("Normal")
            else:
                title_style = None
                normal_style = None
            if title_style is not None:
                elements.append(Paragraph(title, title_style))
            if normal_style is not None and kpi_text:
                elements.append(Spacer(1, 0.1 * inch))
                elements.append(Paragraph(kpi_text.replace("\n","<br/>"), normal_style))
            elements.append(Spacer(1, 0.2 * inch))
            for b in png_bytes_list:
                try:
                    bio = BytesIO(b)
                    rl_img = RLImage(bio, width=6.5 * inch)
                    elements.append(rl_img)
                    elements.append(Spacer(1, 0.2 * inch))
                except Exception:
                    # skip broken images
                    continue
            try:
                buf = BytesIO()
                doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
                doc.build(elements)
                buf.seek(0)
                return buf.read()
            except Exception:
                return None

        pdf_bytes = _build_pdf_bytes("Physical Availability Report", kpi_text, figs_bytes)
        if pdf_bytes:
            st.session_state['_last_pdf'] = pdf_bytes
            st.sidebar.success("PDF generated and ready to download.")
        else:
            st.sidebar.error("Failed to generate PDF. Check server logs and installed dependencies.")
    if st.session_state.get('_last_pdf') is not None:
        st.sidebar.download_button("Download latest PDF", data=st.session_state['_last_pdf'], file_name="PA_report.pdf", mime="application/pdf")
else:
    st.sidebar.info("PDF export unavailable: ReportLab not installed in this environment.")

# Granularity control (sidebar)
granularity = st.sidebar.selectbox("Time granularity", options=["WEEK", "PERIOD_MONTH"], index=1)

# Build month list (chronologically)
months_available = []
if "PERIOD_MONTH" in df.columns:
    unique_pm = pd.Series(df["PERIOD_MONTH"].dropna().astype(str).str.strip().unique())
    parsed = pd.to_datetime(unique_pm, format="%b %Y", errors="coerce")
    if parsed.notna().any():
        months_df = pd.DataFrame({"PERIOD_MONTH": unique_pm.values, "period_dt": parsed.values})
        months_df = months_df.sort_values("period_dt")
        months_available = months_df["PERIOD_MONTH"].tolist()
if not months_available and "MONTH" in df.columns and "YEAR" in df.columns:
    month_to_idx = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
    tmp = []
    for _, r in df.iterrows():
        m = str(r.get("MONTH","")).strip()
        if m=="":
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
    tmp_sorted = sorted(set(tmp), key=lambda x:(x[0],x[1]))
    months_available = [t[2] for t in tmp_sorted]

if months_available:
    try:
        default_idx = len(months_available)-1
    except Exception:
        default_idx = 0
else:
    default_idx = 0
months = ["All"] + months_available
selected_month = st.sidebar.selectbox("MONTH", months, index=0)

# Year filter multi-select (global)
all_years = []
if "YEAR" in df.columns:
    try:
        yrs = pd.Series(df["YEAR"].dropna().astype(int).unique()).sort_values(ascending=False)
        all_years = yrs.tolist()
    except Exception:
        all_years = []
if all_years:
    # DESCENDING default selection (user requested)
    selected_years = st.sidebar.multiselect(
        "Year (multi-select, descending)",
        options=all_years,
        default=all_years,
        format_func=str  # safer, converts everything to string for display
    )

else:
    selected_years = []

# -------------------------
# Tabs: Main Dashboard and Reliability (moved near top as requested)
# -------------------------
tabs = st.tabs(["Main Dashboard", "Reliability"])

# -------------------------
# MAIN DASHBOARD (put entire main content under tabs[0])
# -------------------------
with tabs[0]:
    # Apply initial filtering
    filtered = df.copy()
    if selected_month != "All" and selected_month != "":
        filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]
    if selected_years:
        if "YEAR" in filtered.columns:
            filtered = filtered[filtered["YEAR"].isin(selected_years)].copy()

    # KPI calculations
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
    pa_target = pa_target[0] if pa_target else 0.9
    ma_target = ma_target[0] if ma_target else 0.85
    if isinstance(pa_target, (int,float)) and pa_target>1:
        pa_target = pa_target/100.0
    if isinstance(ma_target, (int,float)) and ma_target>1:
        ma_target = ma_target/100.0

    # YTD calculations
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

    # Top row: KPIs + Donuts
    kpi_col, donut1_col, donut2_col = st.columns([1,2,2])
    with kpi_col:
        st.subheader("Key KPIs")
        min_caption = None
        max_caption = None
        if "PERIOD_MONTH" in filtered.columns and not filtered["PERIOD_MONTH"].dropna().empty:
            parsed = pd.to_datetime(filtered["PERIOD_MONTH"].dropna().unique(), format="%b %Y", errors="coerce")
            if parsed.notna().any():
                min_dt = parsed.min()
                max_dt = parsed.max()
                if pd.notna(min_dt) and pd.notna(max_dt):
                    min_caption = min_dt.strftime("%d/%m/%Y")
                    max_caption = (max_dt + pd.offsets.MonthEnd(0)).strftime("%d/%m/%Y")
        if min_caption is None and "YEAR" in filtered.columns and filtered["YEAR"].notna().any():
            min_y = int(filtered["YEAR"].min())
            max_y = int(filtered["YEAR"].max())
            min_caption = f"01/01/{min_y}"
            max_caption = f"31/12/{max_y}"

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

    with donut1_col:
        st.subheader("Delay by Category")
        if "CATEGORY" in filtered.columns:
            donut_data = filtered.groupby("CATEGORY", dropna=False)["DELAY"].sum().reset_index().sort_values("DELAY", ascending=False)
            if not donut_data.empty:
                donut_data["DELAY"] = donut_data["DELAY"].round(2)
                donut_fig = go.Figure(data=[go.Pie(labels=donut_data["CATEGORY"], values=donut_data["DELAY"], hole=0.4, textinfo="label+percent", hovertemplate="%{label}: %{value:.2f} hrs<extra></extra>")])
                donut_fig.update_layout(margin=dict(t=20,b=20,l=20,r=20))
                # Display donut (we do not include donut in PDF by default)
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
                    st.plotly_chart(donut_fig2, use_container_width=True)
                else:
                    st.info("No maintenance breakdown available.")
            else:
                st.info("No maintenance data found in selection.")
        else:
            st.info("No MTN_DELAY_TYPE column available.")

    st.markdown("---")

    # -------------------------
    # Trend Analysis
    # -------------------------
    st.subheader("Trend: Total Delay Hours vs PA%")
    group_field = granularity

    # 52-week cutoff logic (global)
    if group_field == "WEEK":
        latest_week_start_global = df["WEEK_START"].dropna().max() if "WEEK_START" in df.columns else pd.NaT
        if pd.isna(latest_week_start_global):
            latest_week_start = filtered["WEEK_START"].dropna().max() if "WEEK_START" in filtered.columns else pd.NaT
        else:
            latest_week_start = latest_week_start_global
        if pd.isna(latest_week_start):
            filtered_for_trend = filtered.copy()
        else:
            cutoff_date = latest_week_start - datetime.timedelta(weeks=51)
            filtered_for_trend = filtered[filtered["WEEK_START"].notna() & (pd.to_datetime(filtered["WEEK_START"]) >= pd.to_datetime(cutoff_date))].copy()
            if filtered_for_trend.empty:
                filtered_for_trend = filtered.copy()
    else:
        filtered_for_trend = filtered.copy()

    if group_field == "WEEK":
        trend = filtered_for_trend.groupby(["YEAR","WEEK"], dropna=False).agg(
            total_delay_hours=("DELAY","sum"),
            available_time_month=("AVAILABLE_TIME_MONTH","max"),
            available_hours=("AVAILABLE_HOURS","max")
        ).reset_index()
        trend["period_label"] = trend["YEAR"].astype(str) + " W" + trend["WEEK"].astype("Int64").astype(str)
        def _week_start_from_row(r):
            try:
                return datetime.date.fromisocalendar(int(r["YEAR"]), int(r["WEEK"]), 1)
            except Exception:
                return pd.NaT
        trend["week_start"] = trend.apply(_week_start_from_row, axis=1)
        trend = trend.sort_values(by=["week_start"])
        x_field = "period_label"
    elif group_field == "PERIOD_MONTH":
        trend = filtered.groupby("PERIOD_MONTH", dropna=False).agg(
            total_delay_hours=("DELAY","sum"),
            available_time_month=("AVAILABLE_TIME_MONTH","max"),
            available_hours=("AVAILABLE_HOURS","max")
        ).reset_index()
        trend["period_dt"] = pd.to_datetime(trend["PERIOD_MONTH"], format="%b %Y", errors="coerce")
        trend = trend.sort_values(by=["period_dt", "PERIOD_MONTH"])
        x_field = "PERIOD_MONTH"
    else:
        trend = filtered.groupby(group_field).agg(total_delay_hours=("DELAY","sum"), available_time_month=("AVAILABLE_TIME_MONTH","max"), available_hours=("AVAILABLE_HOURS","max")).reset_index()
        x_field = group_field

    trend["PA_pct"] = None
    trend["available_for_pa"] = None
    for idx, row in trend.iterrows():
        avail_month = row.get("available_time_month", None)
        avail_hours = row.get("available_hours", None)
        if pd.notna(avail_month) and avail_month > 0:
            avail = avail_month
        elif pd.notna(avail_hours) and avail_hours > 0:
            avail = avail_hours
        else:
            avail = None
        trend.at[idx,"available_for_pa"] = avail
        if avail and avail > 0:
            trend.at[idx,"PA_pct"] = max(0, 1 - row["total_delay_hours"] / avail)

    trend["PA_pct"] = pd.to_numeric(trend["PA_pct"], errors="coerce")
    trend["PA_pct_rounded"] = trend["PA_pct"].round(4)
    trend["total_delay_hours"] = pd.to_numeric(trend["total_delay_hours"], errors="coerce")
    trend["total_delay_hours_rounded"] = trend["total_delay_hours"].round(2)

    pa_threshold = pa_target if (pa_target is not None) else 0.9
    colors = []
    for v in trend["PA_pct_rounded"]:
        if pd.isna(v):
            colors.append("lightgrey")
        elif v < pa_threshold:
            colors.append("red")
        else:
            colors.append("green")

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=trend[x_field], y=trend["PA_pct_rounded"], name="PA%", marker=dict(color=colors), hovertemplate="%{y:.2%}<extra></extra>"))
    fig_trend.add_trace(go.Scatter(x=trend[x_field], y=trend["total_delay_hours_rounded"], name="Total Delay Hours", yaxis="y2", mode="lines+markers", hovertemplate="%{y:.2f} hrs<extra></extra>"))
    fig_trend.add_shape(type="line", x0=0, x1=1, xref="paper", y0=pa_target, y1=pa_target, yref="y", line=dict(color="green", dash="dash"))
    fig_trend.add_annotation(x=0, xref="paper", y=pa_target, yref="y", showarrow=False, text=f"PA Target {pa_target:.2%}", font=dict(color="green"), align="left", xanchor="left", yanchor="bottom")
    fig_trend.update_layout(xaxis_title="Period", yaxis=dict(title="PA%", overlaying=None, side="left", tickformat=".2%", range=[0,1]), yaxis2=dict(title="Delay Hours", overlaying="y", side="right"), legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom"), margin=dict(t=70))

    # UPDATED: use helper to cache PNG bytes for PDF; stored to st.session_state['pdf_fig_trend']
    save_and_cache_fig_for_pdf('pdf_fig_trend', fig_trend)

    st.markdown("---")

    # -------------------------
    # Pareto by Equipment
    # -------------------------
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
    fig_pareto.add_trace(go.Scatter(x=pareto_df[equipment_key], y=pareto_df["cum_pct"], name="Cumulative %", mode="lines+markers", hovertemplate="%{y:.2%}<extra></extra>"), secondary_y=True)
    fig_pareto.update_layout(xaxis_tickangle=-45, yaxis_title="Hours", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.15, yanchor="bottom"), margin=dict(t=110))
    fig_pareto.update_yaxes(title_text="Cumulative %", tickformat=".2%", range=[0, 1], secondary_y=True)
    fig_pareto.update_yaxes(title_text="Delay Hours", secondary_y=False)

    # UPDATED: cache PNG for PDF and display
    save_and_cache_fig_for_pdf('pdf_fig_pareto', fig_pareto)

    st.markdown("---")

    # -------------------------
    # CATEGORY FILTER and Drilldown table (unchanged logic)
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

    st.subheader("Drill-down data (filtered by selected category)")
    details_df = drill_df_base.copy()

    if "PERIOD_MONTH" in details_df.columns:
        details_df["MONTH"] = details_df["PERIOD_MONTH"]

    required_cols = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "EQ_DESC", "DELAY", "NOTE", "SUB_CATEGORY", "YEAR"]
    for c in required_cols:
        if c not in details_df.columns:
            details_df[c] = ""

    details_out = details_df[["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "EQ_DESC", "DELAY", "NOTE", "SUB_CATEGORY", "YEAR"]].copy()
    details_out = details_out.rename(columns={"EQ_DESC":"Equipment Description"})

    if selected_category == "MAINTENANCE (ALL)":
        ordered = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "SUB_CATEGORY", "Equipment Description", "DELAY", "NOTE"]
    else:
        ordered = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "Equipment Description", "DELAY", "NOTE"]
    ordered = [c for c in ordered if c in details_out.columns]

    details_out["WEEK"] = pd.to_numeric(details_out["WEEK"], errors="coerce")
    if "YEAR" in details_out.columns and details_out["YEAR"].notna().any():
        details_out["YEAR"] = pd.to_numeric(details_out["YEAR"], errors="coerce")
        details_out = details_out.sort_values(by=["YEAR","WEEK","START"], ascending=[True, True, True]).reset_index(drop=True)
        details_out = details_out.drop(columns=["YEAR"], errors="ignore")
    else:
        details_out = details_out.sort_values(by=["WEEK","START"], ascending=[True, True]).reset_index(drop=True)
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

    # Show using AgGrid and auto-resize columns (fit_columns_on_grid_load)
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
        fit_columns_on_grid_load=True,
        theme="balham"
    )

# -------------------------
# RELIABILITY TAB (tabs[1])
# -------------------------
with tabs[1]:
    st.subheader("Reliability: MTBF & MTTR")

    weekly_df_global = _mtbf_mttr_res.get("weekly_df") if isinstance(_mtbf_mttr_res, dict) else pd.DataFrame()
    monthly_df_global = _mtbf_mttr_res.get("monthly_df") if isinstance(_mtbf_mttr_res, dict) else pd.DataFrame()

    weekly_df = weekly_df_global.copy() if isinstance(weekly_df_global, pd.DataFrame) else pd.DataFrame()
    monthly_df = monthly_df_global.copy() if isinstance(monthly_df_global, pd.DataFrame) else pd.DataFrame()

    # Apply global year filter
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
                if "week_start" in weekly_df_limited.columns:
                    weekly_df_limited = weekly_df_limited.sort_values("week_start", ascending=True)
                fig_mttr_w = go.Figure()
                fig_mttr_w.add_trace(go.Bar(x=weekly_df_limited["period_label"], y=weekly_df_limited["MTTR_hours"].round(2), name="MTTR (hrs)", marker=dict(color="green")))
                fig_mttr_w.update_layout(xaxis_title="Week", yaxis_title="MTTR (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                # cache for PDF + display
                save_and_cache_fig_for_pdf('pdf_fig_mttr_w', fig_mttr_w)
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
                save_and_cache_fig_for_pdf('pdf_fig_mttr_m', fig_mttr_m)
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
                    weekly_df_limited = weekly_df_limited.sort_values("week_start", ascending=True)
                fig_mtbf_w = go.Figure()
                fig_mtbf_w.add_trace(go.Bar(x=weekly_df_limited["period_label"], y=weekly_df_limited["MTBF_hours"].round(2), name="MTBF (hrs)", marker=dict(color="orange")))
                fig_mtbf_w.update_layout(xaxis_title="Week", yaxis_title="MTBF (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                save_and_cache_fig_for_pdf('pdf_fig_mtbf_w', fig_mtbf_w)
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
                save_and_cache_fig_for_pdf('pdf_fig_mtbf_m', fig_mtbf_m)
            else:
                st.info("No PERIOD_MONTH column in monthly reliability data.")

    st.markdown("---")
    st.caption("MTBF and MTTR shown is based on the period selected")

# End of script
