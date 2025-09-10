# Full updated PA_DB.py
# Carefully prepared: main dashboard + reliability tab + PDF export + inline comments for debugging
# Keep backup of previous version before replacing.

import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# AgGrid for drilldown table
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

# For PDF creation & PNG export
import io
from io import BytesIO
from PIL import Image as PILImage

# Try to import ReportLab (PDF). If not installed, graceful degrade.
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
# Page config
# -------------------------
st.set_page_config(page_title="Physical Availability - Data Delay Time", layout="wide")

# -------------------------
# Helpers: figure -> PNG bytes (kaleido)
# -------------------------
def _fig_to_png_bytes(fig):
    try:
        return fig.to_image(format="png")  # no engine arg
    except Exception as e:
        st.warning(f"Chart export failed: {e}")
        return None

def _create_pdf_bytes(title, kpi_text, png_bytes_list):
    """
    Create a PDF containing the title, KPI text and a list of PNG bytes (one image per page/flow).
    Returns raw PDF bytes or None on failure.
    NOTE: ReportLab must be available for this to work.
    """
    if not REPORTLAB_AVAILABLE:
        return None

    elements = []
    styles = getSampleStyleSheet() if getSampleStyleSheet else None
    title_style = styles.get("Heading1") if styles else None
    normal_style = styles.get("Normal") if styles else None

    if title_style:
        elements.append(Paragraph(title, title_style))
    if normal_style and kpi_text:
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(Paragraph(kpi_text.replace("\n", "<br/>"), normal_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Add images (PNG bytes) for charts
    for png in png_bytes_list or []:
        if png is None:
            continue
        try:
            bio = BytesIO(png)
            rl_img = RLImage(bio, width=6.5 * inch)  # fit to page width (approx)
            elements.append(rl_img)
            elements.append(Spacer(1, 0.2 * inch))
        except Exception:
            # skip images that cannot be embedded
            continue

    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=18)
        doc.build(elements)
        buffer.seek(0)
        return buffer.read()
    except Exception:
        return None

# -------------------------
# Auto-hide sidebar after 10 seconds (client-side script). Kept as before.
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

# Logo + Title (unchanged)
LOGO_URL = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/refs/heads/main/images/alamtri_logo.jpeg"
logo_col, title_col = st.columns([1, 8])
with logo_col:
    st.image(LOGO_URL, width=150)
with title_col:
    st.title("Physical Availability Dashboard — Data Delay Time")

# -------------------------
# Config: data source
# -------------------------
RAW_URL = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/refs/heads/main/Draft_New%20Version_Weekly_Report_Maintenance_CHPP.xlsx"

# -------------------------
# Load + Clean function (Data Delay Time sheet)
# -------------------------
@st.cache_data(ttl=600)
def load_data_from_url():
    """Load and clean the 'Data Delay Time' sheet. Returns cleaned DataFrame or None on error."""
    try:
        raw = pd.read_excel(RAW_URL, sheet_name="Data Delay Time", header=None)
    except Exception as e:
        st.error(f"Unable to read sheet 'Data Delay Time': {e}")
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

    # Read again with header
    df = pd.read_excel(RAW_URL, sheet_name="Data Delay Time", header=header_row)
    # Normalize column names
    df.columns = [str(c).strip() for c in df.columns]

    # Map a set of known column labels to normalized ones
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

    # Ensure minimal columns
    essential = ["WEEK", "MONTH", "YEAR", "DELAY"]
    for c in essential:
        if c not in df.columns:
            st.error(f"Expected column '{c}' not found after cleaning. Found columns: {df.columns.tolist()}")
            return None

    # Numeric conversions
    df["DELAY"] = pd.to_numeric(df["DELAY"], errors="coerce")
    if "AVAILABLE_TIME_MONTH" in df.columns:
        df["AVAILABLE_TIME_MONTH"] = pd.to_numeric(df["AVAILABLE_TIME_MONTH"], errors="coerce")
    else:
        df["AVAILABLE_TIME_MONTH"] = None

    # START/STOP parsing to AVAILABLE_HOURS if present
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

    # Normalize textual columns
    for cat in ["MTN_DELAY_TYPE", "SCH_MTN", "UNSCH_MTN", "MINING_DELAY", "WEATHER_DELAY", "OTHER_DELAY", "MTN_NOTE", "NOTE", "EQ_DESC", "EQUIPMENT", "CATEGORY", "PERIOD_MONTH", "DATE"]:
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

    # Compose CAUSE
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

    # YEAR/WEEK numeric normalization
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce").astype("Int64")

    # PERIOD_MONTH fallback if missing
    if "PERIOD_MONTH" not in df.columns or df["PERIOD_MONTH"].isnull().all() or (df["PERIOD_MONTH"].astype(str).str.strip() == "").all():
        if "MONTH" in df.columns and "YEAR" in df.columns:
            df["PERIOD_MONTH"] = df["MONTH"].astype(str).str.strip() + " " + df["YEAR"].astype(str)

    if "PERIOD_MONTH" in df.columns:
        df["PERIOD_MONTH"] = df["PERIOD_MONTH"].astype(str).str.strip()

    # drop rows with missing DELAY numeric
    df = df[df["DELAY"].notna()].copy()

    # UPDATED: allow data from 2023 onwards (user requested 2023+)
    if "YEAR" in df.columns:
        df = df[df["YEAR"].notna() & (df["YEAR"] >= 2023)]

    # Prepare EQUIPMENT_DESC composite
    if "EQUIPMENT" in df.columns and df["EQUIPMENT"].notna().any():
        df["EQUIPMENT_DESC"] = df["EQUIPMENT"].replace("", "(Unknown)").astype(str) + " - " + df["EQ_DESC"].replace("", "(No Desc)").astype(str)
    else:
        df["EQUIPMENT_DESC"] = df["CAUSE"].replace("", "(Unknown)").astype(str)

    # DATE: keep string (YYYY-MM-DD) if possible
    if "DATE" in df.columns:
        try:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date.astype("object").fillna("").astype(str)
        except Exception:
            df["DATE"] = df["DATE"].astype(str).replace("nan", "")
    else:
        df["DATE"] = ""

    # UPDATED: compute WEEK_START (iso-week Monday) to support accurate 52-week limiting
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

    # UPDATED: assign each ISO-week to the month containing the week end (latest month)
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
# Compute MTBF/MTTR from "Data Operational" sheet
# -------------------------
@st.cache_data(ttl=600)
def compute_mtbf_mttr_from_url(raw_url):
    """
    Reads the Data Operational sheet (case-insensitive search).
    Computes per-week and per-month MTBF and MTTR using:
      - MTBF (hrs) = total_operational_hours / count_rows (per period)
      - MTTR (hrs) = total_maintenance_delay / count_rows (per period)
    COUNT is a simple row count as requested.
    Returns dictionary with weekly_df, monthly_df and other metadata.
    """
    try:
        xls = pd.ExcelFile(raw_url)
    except Exception as e:
        return {"error": f"Could not open workbook: {e}"}

    # find Data Operational sheet name by exact or fuzzy match
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

    # Detect relevant columns (case-insensitive)
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

    # date-like column detection for grouping
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
            # numeric values (including Excel time fraction)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                v = float(val)
                # Excel time fraction -> hours
                if 0 < v < 1:
                    return v * 24.0
                return v
            s = str(val).strip()
            # time-like string e.g. "08:30" or "8:30:00"
            if ":" in s:
                try:
                    td = pd.to_timedelta(s)
                    return td.total_seconds() / 3600.0
                except Exception:
                    pass
            # numeric string
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

    # parse date column if present
    if date_col is not None:
        try:
            df_op["_date_parsed"] = pd.to_datetime(df_op[date_col], errors="coerce", dayfirst=False)
        except Exception:
            df_op["_date_parsed"] = pd.Series([pd.NaT] * len(df_op))
    else:
        df_op["_date_parsed"] = pd.Series([pd.NaT] * len(df_op))

    # produce YEAR and WEEK columns using isocalendar (vectorized)
    try:
        iso = df_op["_date_parsed"].dt.isocalendar()
        # iso may be a DataFrame-like with columns year, week, day
        df_op["YEAR"] = iso["year"].astype("Int64")
        df_op["WEEK"] = iso["week"].astype("Int64")
    except Exception:
        # fallback using apply
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
        df_op["YEAR"] = df_op["_yr_wk"].apply(lambda t: t[0] if isinstance(t, tuple) else None).astype("Int64")
        df_op["WEEK"] = df_op["_yr_wk"].apply(lambda t: t[1] if isinstance(t, tuple) else None).astype("Int64")

    # compute WEEK_START and PERIOD_MONTH for monthly grouping (week end month)
    try:
        df_op["WEEK_START"] = df_op["_date_parsed"].dt.to_period("W").apply(lambda p: p.start_time.date())
        ws = pd.to_datetime(df_op["WEEK_START"], errors="coerce")
        we = ws + pd.Timedelta(days=6)
        df_op["PERIOD_MONTH"] = we.dt.strftime("%b %Y")
    except Exception:
        # fallback to month of date parsed
        try:
            df_op["PERIOD_MONTH"] = df_op["_date_parsed"].dt.strftime("%b %Y")
        except Exception:
            df_op["PERIOD_MONTH"] = ""

    # Now group weekly and monthly using consistent column names (no leading spaces!)
    try:
        weekly_group = df_op.groupby(["YEAR", "WEEK"], dropna=False).agg(
            total_operational_hours=("_op_hours_dec", "sum"),
            total_maintenance_delay=("_maint_delay_num", "sum"),
            count_rows=("_op_hours_dec", "count")
        ).reset_index()
    except Exception:
        # fallback if above fails for some reason
        weekly_group = pd.DataFrame()
        try:
            weekly_group = df_op.groupby(["YEAR", "WEEK"], dropna=False).agg({
                "_op_hours_dec": "sum",
                "_maint_delay_num": "sum",
            }).reset_index()
            weekly_group = weekly_group.rename(columns={"_op_hours_dec": "total_operational_hours", "_maint_delay_num": "total_maintenance_delay"})
            weekly_group["count_rows"] = df_op.groupby(["YEAR", "WEEK"], dropna=False).size().values
        except Exception:
            # give empty result but do not crash
            weekly_group = pd.DataFrame(columns=["YEAR", "WEEK", "total_operational_hours", "total_maintenance_delay", "count_rows"])

    # compute MTBF & MTTR per-week (hours)
    if not weekly_group.empty:
        weekly_group["MTBF_hours"] = weekly_group.apply(lambda r: (r["total_operational_hours"] / r["count_rows"]) if (r.get("count_rows") and r["count_rows"] > 0) else float("nan"), axis=1)
        weekly_group["MTTR_hours"] = weekly_group.apply(lambda r: (r["total_maintenance_delay"] / r["count_rows"]) if (r.get("count_rows") and r["count_rows"] > 0) else float("nan"), axis=1)
        # period label and week_start for ordering
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
    else:
        weekly_group = pd.DataFrame(columns=["YEAR","WEEK","total_operational_hours","total_maintenance_delay","count_rows","MTBF_hours","MTTR_hours","period_label","week_start"])

    # Monthly grouping
    try:
        monthly_group = df_op.groupby("PERIOD_MONTH", dropna=False).agg(
            total_operational_hours=("_op_hours_dec", "sum"),
            total_maintenance_delay=("_maint_delay_num", "sum"),
            count_rows=("_op_hours_dec", "count")
        ).reset_index()
    except Exception:
        monthly_group = pd.DataFrame(columns=["PERIOD_MONTH","total_operational_hours","total_maintenance_delay","count_rows"])

    if not monthly_group.empty:
        monthly_group["MTBF_hours"] = monthly_group.apply(lambda r: (r["total_operational_hours"] / r["count_rows"]) if (r.get("count_rows") and r["count_rows"] > 0) else float("nan"), axis=1)
        monthly_group["MTTR_hours"] = monthly_group.apply(lambda r: (r["total_maintenance_delay"] / r["count_rows"]) if (r.get("count_rows") and r["count_rows"] > 0) else float("nan"), axis=1)
        try:
            monthly_group["period_dt"] = pd.to_datetime(monthly_group["PERIOD_MONTH"], format="%b %Y", errors="coerce")
        except Exception:
            monthly_group["period_dt"] = pd.NaT
        monthly_group = monthly_group.sort_values("period_dt").reset_index(drop=True)

    # Overall totals
    total_rows = len(df_op)
    total_operational_hours = float(df_op["_op_hours_dec"].sum(skipna=True)) if "_op_hours_dec" in df_op.columns else 0.0
    total_maintenance_delay = float(df_op["_maint_delay_num"].sum(skipna=True)) if "_maint_delay_num" in df_op.columns else 0.0

    return {
        "sheet_name": sheet_name,
        "operational_hours_column": op_col,
        "maintenance_delay_column": maint_col,
        "date_column": date_col,
        "weekly_df": weekly_group,
        "monthly_df": monthly_group,
        "total_rows": total_rows,
        "total_operational_hours": total_operational_hours,
        "total_maintenance_delay": total_maintenance_delay,
        "MTBF_hours_overall": (total_operational_hours / total_rows) if total_rows > 0 else None,
        "MTTR_hours_overall": (total_maintenance_delay / total_rows) if total_rows > 0 else None
    }

# execute reliability compute (cached)
_mtbf_mttr_res = compute_mtbf_mttr_from_url(RAW_URL)

# Expose reliability globals (non-colliding names)
MTBF_GLOBAL_HOURS = _mtbf_mttr_res.get("MTBF_hours_overall") if isinstance(_mtbf_mttr_res, dict) else None
MTTR_GLOBAL_HOURS = _mtbf_mttr_res.get("MTTR_hours_overall") if isinstance(_mtbf_mttr_res, dict) else None

# -------------------------
# Build month list (chronological) and year list (descending)
# -------------------------
months_available = []
if "PERIOD_MONTH" in df.columns:
    unique_pm = pd.Series(df["PERIOD_MONTH"].dropna().astype(str).str.strip().unique())
    parsed = pd.to_datetime(unique_pm, format="%b %Y", errors="coerce")
    if parsed.notna().any():
        months_df = pd.DataFrame({"PERIOD_MONTH": unique_pm.values, "period_dt": parsed.values})
        months_df = months_df.sort_values("period_dt")
        months_available = months_df["PERIOD_MONTH"].tolist()

# fallback to month + year logic
if not months_available and "MONTH" in df.columns and "YEAR" in df.columns:
    month_to_idx = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12
    }
    tmp = []
    for _, r in df.iterrows():
        m = str(r.get("MONTH", "")).strip()
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

# Years list (descending)
all_years = []
if "YEAR" in df.columns:
    try:
        yrs = pd.Series(df["YEAR"].dropna().astype(int).unique()).sort_values(ascending=False)
        all_years = yrs.tolist()
    except Exception:
        all_years = []

# -------------------------
# Sidebar controls (global): Year multi-select, Month select, Granularity, Export PDF
# -------------------------
st.sidebar.header("Filters & Options")

# Year multi-select (descending default: all years)
if all_years:
    selected_years = st.sidebar.multiselect("Year (multi-select, descending)", options=all_years, default=all_years)
else:
    selected_years = []

# Granularity preserved (WEEK or PERIOD_MONTH)
granularity = st.sidebar.selectbox("Time granularity", options=["WEEK", "PERIOD_MONTH"], index=1)

# Month filter (keep default to "All")
months = ["All"] + months_available
selected_month = st.sidebar.selectbox("MONTH", months, index=0)

# PDF export controls (will use cached PNGs)
st.sidebar.markdown("---")
st.sidebar.subheader("Export report (PDF)")
_pdf_status_msg = st.sidebar.empty()  # placeholder for status messages

# Prepare placeholders in session_state for PNG caching (so we can use them in PDF)
for key in ['pdf_fig_trend','pdf_fig_pareto','pdf_fig_mttr_w','pdf_fig_mttr_m','pdf_fig_mtbf_w','pdf_fig_mtbf_m']:
    if key not in st.session_state:
        st.session_state[key] = None

# NOTE: The PDF generation button is located in the sidebar but it will use PNGs cached by plotting code below.

# -------------------------
# Apply global filters to base dataframe (filtered)
# -------------------------
filtered = df.copy()
if selected_month != "All" and selected_month != "":
    filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]

# Apply global year filter if any selected
if selected_years:
    if "YEAR" in filtered.columns:
        filtered = filtered[filtered["YEAR"].isin(selected_years)].copy()

# -------------------------
# MAIN DASHBOARD & RELIABILITY tabs
# -------------------------
tabs = st.tabs(["Main Dashboard", "Reliability"])  # user chooses tab manually

# =========================
# TAB 0: Main Dashboard
# =========================
with tabs[0]:
    # -------------------------
    # KPI Calculations (before category drilldown)
    # -------------------------
    total_delay = filtered["DELAY"].sum()

    # AVAILABLE_TIME aggregation
    available_time = None
    try:
        if "AVAILABLE_TIME_MONTH" in filtered.columns and filtered["AVAILABLE_TIME_MONTH"].notna().any():
            available_time = (
                filtered.groupby("PERIOD_MONTH", dropna=True)["AVAILABLE_TIME_MONTH"]
                .max()
                .dropna()
                .sum()
            )
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
    if isinstance(pa_target, (int, float)) and pa_target > 1:
        pa_target = pa_target / 100.0
    if isinstance(ma_target, (int, float)) and ma_target > 1:
        ma_target = ma_target / 100.0

    # -------------------------
    # Year-to-date (YTD) calculations
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

    # -------------------------
    # KPIs + Donuts
    # -------------------------
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

        # Use 2-digit percentage formatting as requested
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
                # cache for PDF if possible
                png = _fig_to_png_bytes(donut_fig)
                if png:
                    st.session_state['pdf_fig_pareto'] = png
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
    # Trend Analysis (PA% bars, Delay line)
    # -------------------------
    st.subheader("Trend: Total Delay Hours vs PA%")
    group_field = granularity

    # UPDATED: Use global latest week for 52-week cutoff when grouping by week
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

    # compute PA_pct per period
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

    # ---------- swap visuals & coloring (PA bars, Delay line) ----------
    trend["PA_pct"] = pd.to_numeric(trend["PA_pct"], errors="coerce")
    # Use more precise internal rounding then display as percent .2%
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
    # Bar shows PA% (fraction 0..1)
    fig_trend.add_trace(
        go.Bar(
            x=trend[x_field],
            y=trend["PA_pct_rounded"],
            name="PA%",
            marker=dict(color=colors),
            hovertemplate="%{y:.2%}<extra></extra>"
        )
    )
    # Line shows delay hours on secondary y axis
    fig_trend.add_trace(
        go.Scatter(
            x=trend[x_field],
            y=trend["total_delay_hours_rounded"],
            name="Total Delay Hours",
            yaxis="y2",
            mode="lines+markers",
            hovertemplate="%{y:.2f} hrs<extra></extra>"
        )
    )

    # draw PA target line on left axis
    fig_trend.add_shape(type="line", x0=0, x1=1, xref="paper", y0=pa_target, y1=pa_target, yref="y", line=dict(color="green", dash="dash"))
    fig_trend.add_annotation(x=0, xref="paper", y=pa_target, yref="y", showarrow=False, text=f"PA Target {pa_target:.2%}", font=dict(color="green"), align="left", xanchor="left", yanchor="bottom")

    fig_trend.update_layout(
        xaxis_title="Period",
        yaxis=dict(title="PA%", overlaying=None, side="left", tickformat=".2%", range=[0,1]),
        yaxis2=dict(title="Delay Hours", overlaying="y", side="right"),
        # legend below title
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom"),
        margin=dict(t=70)
    )
    # cache PNG for PDF
    png = _fig_to_png_bytes(fig_trend)
    if png:
        st.session_state['pdf_fig_trend'] = png
    st.plotly_chart(fig_trend, use_container_width=True)

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
    # move legend above chart (below selection bar) to avoid overlapping equipment names
    fig_pareto.update_layout(xaxis_tickangle=-45, yaxis_title="Hours", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.15, yanchor="bottom"), margin=dict(t=110))
    fig_pareto.update_yaxes(title_text="Cumulative %", tickformat=".2%", range=[0, 1], secondary_y=True)
    fig_pareto.update_yaxes(title_text="Delay Hours", secondary_y=False)

    # cache PNG for PDF
    png = _fig_to_png_bytes(fig_pareto)
    if png:
        st.session_state['pdf_fig_pareto'] = png
    st.plotly_chart(fig_pareto, use_container_width=True)

    # -------------------------
    # CATEGORY Filter -> Drilldown data
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
    # Drill-down table (filtered by selected category)
    # -------------------------
    # Title includes selected years info (applied globally)
    years_label = "All years" if not selected_years else ", ".join([str(y) for y in selected_years])
    st.subheader(f"Drill-down data (filtered by selected category) — {years_label}")

    details_df = drill_df_base.copy()

    # Show the assigned month in the drilldown table using PERIOD_MONTH (week -> latest month)
    if "PERIOD_MONTH" in details_df.columns:
        details_df["MONTH"] = details_df["PERIOD_MONTH"]

    # Ensure required columns exist
    required_cols = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "EQ_DESC", "DELAY", "NOTE", "SUB_CATEGORY", "YEAR"]
    for c in required_cols:
        if c not in details_df.columns:
            details_df[c] = ""

    # Build details_out for display
    details_out = details_df[["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "EQ_DESC", "DELAY", "NOTE", "SUB_CATEGORY", "YEAR"]].copy()
    details_out = details_out.rename(columns={"EQ_DESC": "Equipment Description"})

    # Reorder columns depending on category selection
    if selected_category == "MAINTENANCE (ALL)":
        ordered = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "SUB_CATEGORY", "Equipment Description", "DELAY", "NOTE"]
    else:
        ordered = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "Equipment Description", "DELAY", "NOTE"]
    ordered = [c for c in ordered if c in details_out.columns]

    # Convert WEEK to numeric for stable sorting
    details_out["WEEK"] = pd.to_numeric(details_out["WEEK"], errors="coerce")

    # Sort descending (USER REQUESTED descending is better)
    if "YEAR" in details_out.columns and details_out["YEAR"].notna().any():
        details_out["YEAR"] = pd.to_numeric(details_out["YEAR"], errors="coerce")
        # Descending YEAR and WEEK, keep START ascending for time ordering within week
        details_out = details_out.sort_values(by=["YEAR", "WEEK", "START"], ascending=[False, False, True]).reset_index(drop=True)
        details_out = details_out.drop(columns=["YEAR"], errors="ignore")
    else:
        details_out = details_out.sort_values(by=["WEEK", "START"], ascending=[False, True]).reset_index(drop=True)

    # Keep ordered columns only
    details_out = details_out[ordered].reset_index(drop=True)

    # Round DELAY to 2 decimals for display
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

    # Show table via AgGrid
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
    # After plots: PDF generation UI in sidebar uses cached PNGs. Show button and generate PDF on press.
    # (We place generation here to ensure the PNGs above are cached before user generates)
    # -------------------------
    # Build KPI summary for PDF header (short)
    try:
        _pdf_kpi_text = ""
        _pdf_kpi_text += f"PA: {PA:.2%}\n" if PA is not None else ""
        _pdf_kpi_text += f"MA: {MA:.2%}\n" if MA is not None else ""
        _pdf_kpi_text += f"Total Delay (selected): {total_delay:.2f} hrs\n"
        if available_time:
            _pdf_kpi_text += f"Total Available Time (selected): {available_time:.2f} hrs\n"
    except Exception:
        _pdf_kpi_text = ""

    # PDF generate button (in sidebar) - user can press anytime after visuals rendered
    if REPORTLAB_AVAILABLE:
        if st.sidebar.button("Generate PDF"):
            figs_pngs = []
            for k in ['pdf_fig_trend','pdf_fig_pareto','pdf_fig_mttr_w','pdf_fig_mttr_m','pdf_fig_mtbf_w','pdf_fig_mtbf_m']:
                val = st.session_state.get(k)
                if val:
                    figs_pngs.append(val)
            pdf_bytes = _create_pdf_bytes("Physical Availability Report", _pdf_kpi_text, figs_pngs)
            if pdf_bytes:
                st.session_state['_last_pdf'] = pdf_bytes
                _pdf_status_msg.success("PDF generated and ready to download.")
            else:
                _pdf_status_msg.error("Failed to generate PDF. Check server logs and dependencies.")
        if st.session_state.get('_last_pdf') is not None:
            st.sidebar.download_button("Download latest PDF", data=st.session_state['_last_pdf'], file_name="PA_report.pdf", mime="application/pdf")
    else:
        st.sidebar.info("PDF export unavailable: ReportLab not installed in environment.")

# =========================
# TAB 1: Reliability
# =========================
with tabs[1]:
    st.subheader("Reliability: MTBF & MTTR")

    # Get computed weekly/monthly dfs from compute results
    weekly_df_global = _mtbf_mttr_res.get("weekly_df") if isinstance(_mtbf_mttr_res, dict) else pd.DataFrame()
    monthly_df_global = _mtbf_mttr_res.get("monthly_df") if isinstance(_mtbf_mttr_res, dict) else pd.DataFrame()

    weekly_df = weekly_df_global.copy() if isinstance(weekly_df_global, pd.DataFrame) else pd.DataFrame()
    monthly_df = monthly_df_global.copy() if isinstance(monthly_df_global, pd.DataFrame) else pd.DataFrame()

    # Apply global year filter to reliability dataframes
    if not weekly_df.empty and selected_years:
        if "YEAR" in weekly_df.columns:
            weekly_df = weekly_df[weekly_df["YEAR"].isin(selected_years)]
    if not monthly_df.empty and selected_years:
        if "period_dt" in monthly_df.columns and monthly_df["period_dt"].notna().any():
            monthly_df = monthly_df[monthly_df["period_dt"].dt.year.isin(selected_years)]
        else:
            # attempt to extract year from PERIOD_MONTH string
            def _pm_year(s):
                try:
                    return int(str(s).split()[-1])
                except Exception:
                    return None
            if "PERIOD_MONTH" in monthly_df.columns:
                monthly_df["_yr"] = monthly_df["PERIOD_MONTH"].apply(_pm_year)
                monthly_df = monthly_df[monthly_df["_yr"].isin(selected_years)]
                monthly_df = monthly_df.drop(columns=["_yr"], errors="ignore")

    # MTTR charts (weekly + monthly)
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
                # Ensure ascending order on x-axis (week_start)
                if "week_start" in weekly_df_limited.columns:
                    weekly_df_limited = weekly_df_limited.sort_values("week_start")
                fig_mttr_w = go.Figure()
                # Bars only (no line), green color for MTTR
                fig_mttr_w.add_trace(go.Bar(
                    x=weekly_df_limited["period_label"],
                    y=weekly_df_limited["MTTR_hours"].round(2),
                    name="MTTR (hrs)",
                    marker=dict(color="green"),
                    hovertemplate="%{y:.2f} hrs<extra></extra>"
                ))
                fig_mttr_w.update_layout(xaxis_title="Week", yaxis_title="MTTR (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                # cache PNG for PDF
                png = _fig_to_png_bytes(fig_mttr_w)
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
                # ascending order for x-axis
                monthly_df_local = monthly_df_local.sort_values(by="period_dt" if "period_dt" in monthly_df_local.columns else "PERIOD_MONTH", ascending=True)
                fig_mttr_m = go.Figure()
                fig_mttr_m.add_trace(go.Bar(
                    x=monthly_df_local["PERIOD_MONTH"],
                    y=monthly_df_local["MTTR_hours"].round(2),
                    name="MTTR (hrs)",
                    marker=dict(color="green"),
                    hovertemplate="%{y:.2f} hrs<extra></extra>"
                ))
                fig_mttr_m.update_layout(xaxis_title="Month", yaxis_title="MTTR (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                png = _fig_to_png_bytes(fig_mttr_m)
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
                # Bars only, orange color
                fig_mtbf_w.add_trace(go.Bar(
                    x=weekly_df_limited["period_label"],
                    y=weekly_df_limited["MTBF_hours"].round(2),
                    name="MTBF (hrs)",
                    marker=dict(color="orange"),
                    hovertemplate="%{y:.2f} hrs<extra></extra>"
                ))
                fig_mtbf_w.update_layout(xaxis_title="Week", yaxis_title="MTBF (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                png = _fig_to_png_bytes(fig_mtbf_w)
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
                fig_mtbf_m.add_trace(go.Bar(
                    x=monthly_df_local["PERIOD_MONTH"],
                    y=monthly_df_local["MTBF_hours"].round(2),
                    name="MTBF (hrs)",
                    marker=dict(color="orange"),
                    hovertemplate="%{y:.2f} hrs<extra></extra>"
                ))
                fig_mtbf_m.update_layout(xaxis_title="Month", yaxis_title="MTBF (hours)", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02), margin=dict(t=60))
                png = _fig_to_png_bytes(fig_mtbf_m)
                if png:
                    st.session_state['pdf_fig_mtbf_m'] = png
                st.plotly_chart(fig_mtbf_m, use_container_width=True)
            else:
                st.info("No PERIOD_MONTH column in monthly reliability data.")

    st.markdown("---")
    # Short caption — simplified per your request
    st.caption("MTBF and MTTR shown are based on the selected period.")

# End of file
