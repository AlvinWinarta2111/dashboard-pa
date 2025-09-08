import streamlit as st
import pandas as pd
import datetime
from datetime import timedelta, date
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

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

# Logo URL
LOGO_URL = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/refs/heads/main/images/alamtri_logo.jpeg"

# Layout for title and logo
logo_col, title_col = st.columns([1, 8])

with logo_col:
    st.image(LOGO_URL, width=150)

with title_col:
    st.title("Physical Availability Dashboard — Data Delay Time")

# -------------------------
# Config
# -------------------------
RAW_URL = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/refs/heads/main/Draft_New%20Version_Weekly_Report_Maintenance_CHPP.xlsx"

# -------------------------
# Load + Clean Function
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
    # normalize column names
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

    # Ensure minimal expected columns exist
    essential = ["WEEK", "MONTH", "YEAR", "DELAY"]
    for c in essential:
        if c not in df.columns:
            st.error(f"Expected column '{c}' not found after cleaning. Found columns: {df.columns.tolist()}")
            return None

    # Numeric conversion where needed
    df["DELAY"] = pd.to_numeric(df["DELAY"], errors="coerce")
    if "AVAILABLE_TIME_MONTH" in df.columns:
        df["AVAILABLE_TIME_MONTH"] = pd.to_numeric(df["AVAILABLE_TIME_MONTH"], errors="coerce")
    else:
        df["AVAILABLE_TIME_MONTH"] = None

    # Keep START/STOP as strings for display and compute AVAILABLE_HOURS (if both present)
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
    for cat in ["MTN_DELAY_TYPE", "SCH_MTN", "UNSCH_MTN", "MINING_DELAY", "WEATHER_DELAY", "OTHER_DELAY", "MTN_NOTE", "NOTE", "EQ_DESC", "EQUIPMENT", "CATEGORY", "PERIOD_MONTH", "DATE"]:
        if cat in df.columns:
            # keep raw text, strip whitespace
            df[cat] = df[cat].fillna("").astype(str).str.strip()
        else:
            df[cat] = ""

    # ---------------------
    # SUB_CATEGORY detection (Scheduled / Unscheduled) - robust
    # ---------------------
    def detect_subcat_row(r):
        mt = str(r.get("MTN_DELAY_TYPE", "")).strip().lower()
        sch = str(r.get("SCH_MTN", "")).strip().lower()
        unsch = str(r.get("UNSCH_MTN", "")).strip().lower()

        # Explicit "unscheduled" wins
        if mt == "unscheduled" or mt.startswith("uns") or unsch not in ("", "nan", "none"):
            return "Unscheduled"

        # Explicit "scheduled" wins
        if mt == "scheduled" or mt.startswith("sch") or sch not in ("", "nan", "none"):
            return "Scheduled"

        # Nothing matched → leave blank
        return ""

    df["SUB_CATEGORY"] = df.apply(detect_subcat_row, axis=1)

    # ---------------------
    # CATEGORY determination — Maintenance when SUB_CATEGORY set
    # ---------------------
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

    # Add PERIOD_MONTH if missing using MONTH + YEAR (keep MONTH as original 3-letter)
    if "PERIOD_MONTH" not in df.columns or df["PERIOD_MONTH"].isnull().all() or (df["PERIOD_MONTH"].astype(str).str.strip()=="").all():
        if "MONTH" in df.columns and "YEAR" in df.columns:
            df["PERIOD_MONTH"] = df["MONTH"].astype(str).str.strip() + " " + df["YEAR"].astype(str)

    # trim
    if "PERIOD_MONTH" in df.columns:
        df["PERIOD_MONTH"] = df["PERIOD_MONTH"].astype(str).str.strip()

    # drop rows with no DELAY numeric
    df = df[df["DELAY"].notna()].copy()

    # ---------- (CHANGE) allow data from 2023 onwards ----------
    if "YEAR" in df.columns:
        df = df[df["YEAR"].notna() & (df["YEAR"] >= 2023)]

    # Prepare EQUIPMENT_DESC composite
    if "EQUIPMENT" in df.columns and df["EQUIPMENT"].notna().any():
        df["EQUIPMENT_DESC"] = df["EQUIPMENT"].replace("", "(Unknown)").astype(str) + " - " + df["EQ_DESC"].replace("", "(No Desc)").astype(str)
    else:
        df["EQUIPMENT_DESC"] = df["CAUSE"].replace("", "(Unknown)").astype(str)

    # DATE: keep as string (YYYY-MM-DD) if possible
    if "DATE" in df.columns:
        try:
            df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date.astype("object").fillna("").astype(str)
        except Exception:
            df["DATE"] = df["DATE"].astype(str).replace("nan", "")
    else:
        df["DATE"] = ""

    # ---------- (NEW) compute WEEK_START (ISO week Monday) to support accurate 52-week limiting ----------
    def _compute_week_start(r):
        # try YEAR+WEEK first
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
        # fallback to DATE column if available
        try:
            d = pd.to_datetime(r.get("DATE"), errors="coerce")
            if pd.notna(d):
                iso = d.isocalendar()
                return datetime.date.fromisocalendar(int(iso.year), int(iso.week), 1)
        except Exception:
            pass
        return pd.NaT

    df["WEEK_START"] = df.apply(_compute_week_start, axis=1)

    # ---------- (NEW) assign each ISO-week to the month containing the week end (latest month of that week) ----------
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
# NEW: compute MTBF & MTTR from "Data Operational" sheet (COUNT = total rows)
# -------------------------
@st.cache_data(ttl=600)
def compute_mtbf_mttr_from_url(raw_url):
    """
    Reads the 'Data Operational' sheet from the workbook at raw_url,
    parses OPERATIONAL HOURS into decimal hours and sums MAINTENANCE DELAY.
    Uses COUNT = total number of rows in the Data Operational sheet as denominator.
    Returns a dict with results and detected column names.
    """
    try:
        xls = pd.ExcelFile(raw_url)
    except Exception as e:
        return {"error": f"Could not open workbook: {e}"}

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

    # attempt to detect OPERATIONAL HOURS and MAINTENANCE DELAY columns (case-insensitive)
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

    # try to find a date column
    for c in df_op.columns:
        lc = str(c).lower()
        if lc in ("date", "tanggal") or "date" in lc:
            date_col = c
            break

    # parsing function for operational hours -> decimal hours
    def _parse_operational_hours(val):
        try:
            if pd.isna(val) or (isinstance(val, str) and str(val).strip() == ""):
                return float("nan")
            # numeric values (including Excel time fraction)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                v = float(val)
                # Excel time fraction
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

    # parse date column to datetime if present
    if date_col is not None:
        df_op["_date_parsed"] = pd.to_datetime(df_op[date_col], errors="coerce")
    else:
        # try to find a column named DATE regardless of case
        _found = None
        for c in df_op.columns:
            if str(c).strip().lower() == "date":
                _found = c
                break
        if _found:
            df_op["_date_parsed"] = pd.to_datetime(df_op[_found], errors="coerce")
        else:
            df_op["_date_parsed"] = pd.Series([pd.NaT] * len(df_op))

    total_rows_op_sheet = len(df_op)
    total_operational_hours_op_sheet = float(df_op["_op_hours_dec"].sum(skipna=True))
    total_maintenance_delay_op_sheet = float(df_op["_maint_delay_num"].sum(skipna=True))

    if total_rows_op_sheet > 0:
        mtbf_hours = total_operational_hours_op_sheet / total_rows_op_sheet
        mttr_hours = total_maintenance_delay_op_sheet / total_rows_op_sheet
    else:
        mtbf_hours = None
        mttr_hours = None

    # return also the processed df for further grouping (but not cached as large payload)
    return {
        "sheet_name": sheet_name,
        "operational_hours_column": op_col,
        "maintenance_delay_column": maint_col,
        "date_column": date_col,
        "total_rows": total_rows_op_sheet,
        "total_operational_hours": total_operational_hours_op_sheet,
        "total_maintenance_delay": total_maintenance_delay_op_sheet,
        "MTBF_hours": mtbf_hours,
        "MTTR_hours": mttr_hours,
        "df_op_processed_sample": df_op.head(5).to_dict(orient="records"),  # lightweight sample
    }

# call the MTBF/MTTR computation (safe, cached)
_mtbf_mttr_res = compute_mtbf_mttr_from_url(RAW_URL)

# Expose variables (global names that won't clash with existing variables)
MTBF_GLOBAL_HOURS = _mtbf_mttr_res.get("MTBF_hours") if isinstance(_mtbf_mttr_res, dict) else None
MTTR_GLOBAL_HOURS = _mtbf_mttr_res.get("MTTR_hours") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_GLOBAL_HOURS_ROUNDED = round(MTBF_GLOBAL_HOURS, 2) if (MTBF_GLOBAL_HOURS is not None) else None
MTTR_GLOBAL_HOURS_ROUNDED = round(MTTR_GLOBAL_HOURS, 2) if (MTTR_GLOBAL_HOURS is not None) else None

# Also keep totals & column names (useful later if you want to display them)
MTBF_SOURCE_SHEET = _mtbf_mttr_res.get("sheet_name") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_OP_HOURS_COL = _mtbf_mttr_res.get("operational_hours_column") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_MAINT_DELAY_COL = _mtbf_mttr_res.get("maintenance_delay_column") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_TOTAL_ROWS = _mtbf_mttr_res.get("total_rows") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_TOTAL_OP_HOURS = _mtbf_mttr_res.get("total_operational_hours") if isinstance(_mtbf_mttr_res, dict) else None
MTBF_TOTAL_MAINT_DELAY = _mtbf_mttr_res.get("total_maintenance_delay") if isinstance(_mtbf_mttr_res, dict) else None

# -------------------------
# Load operational sheet fully for reliability time series (not cached inside compute above)
# -------------------------
@st.cache_data(ttl=600)
def load_operational_sheet(raw_url):
    """
    Returns a processed DataFrame df_op_proc with:
      - _op_hours_dec: operational hours as decimal hours
      - _maint_delay_num: maintenance delay (numeric)
      - DATE_PARSED: parsed timestamp if present
      - YEAR, WEEK columns (ISO week)
    """
    try:
        xls = pd.ExcelFile(raw_url)
    except Exception as e:
        return None, f"Could not open workbook: {e}"

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
        return None, "Data Operational sheet not found"

    try:
        df_op = pd.read_excel(raw_url, sheet_name=sheet_name)
    except Exception as e:
        return None, f"Could not read sheet '{sheet_name}': {e}"

    # detect columns
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
            if "operat" in str(c).lower():
                op_col = c
                break

    for c in df_op.columns:
        lc = str(c).lower()
        if "maint" in lc and "delay" in lc:
            maint_col = c
            break
    if maint_col is None:
        for c in df_op.columns:
            if "maintenance" in str(c).lower() and "delay" in str(c).lower():
                maint_col = c
                break

    for c in df_op.columns:
        if str(c).strip().lower() == "date" or "date" in str(c).lower():
            date_col = c
            break

    # parse op hours and maintenance delay
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

    # parse date
    if date_col is not None:
        df_op["_date_parsed"] = pd.to_datetime(df_op[date_col], errors="coerce')
    else:
        # try to find a column named DATE
        found = None
        for c in df_op.columns:
            if str(c).strip().lower() == "date":
                found = c
                break
        if found:
            df_op["_date_parsed"] = pd.to_datetime(df_op[found], errors="coerce")
        else:
            df_op["_date_parsed"] = pd.Series([pd.NaT] * len(df_op))

    # Create YEAR and ISO WEEK based on _date_parsed if present, fallback to other columns if needed
    # Ensure df_op is a plain DataFrame (reset index)
    df_op = df_op.reset_index(drop=True)

    # If there are existing YEAR or WEEK columns, prioritize parsed date to avoid mismatch
    try:
        # use isocalendar for correct ISO week/year
        df_op["_date_parsed"] = pd.to_datetime(df_op["_date_parsed"], errors="coerce")
        iso_series = df_op["_date_parsed"].apply(lambda dt: pd.NaT if pd.isna(dt) else dt.isocalendar())
        # build YEAR and WEEK columns safely
        def _extract_iso_year_week(x):
            try:
                if x is pd.NaT:
                    return (pd.NA, pd.NA)
                # x is a namedtuple (year, week, weekday)
                return (int(x.year), int(x.week))
            except Exception:
                return (pd.NA, pd.NA)

        iso_tuples = iso_series.apply(_extract_iso_year_week)
        df_op["YEAR"] = iso_tuples.apply(lambda t: t[0]).astype("Int64")
        df_op["WEEK"] = iso_tuples.apply(lambda t: t[1]).astype("Int64")
    except Exception:
        # fallback: try existing columns
        if "YEAR" in df_op.columns:
            df_op["YEAR"] = pd.to_numeric(df_op["YEAR"], errors="coerce").astype("Int64")
        if "WEEK" in df_op.columns:
            df_op["WEEK"] = pd.to_numeric(df_op["WEEK"], errors="coerce").astype("Int64")

    # Final safety: ensure YEAR & WEEK are normal columns (1-D)
    if "YEAR" not in df_op.columns:
        df_op["YEAR"] = pd.Series([pd.NA] * len(df_op)).astype("Int64")
    if "WEEK" not in df_op.columns:
        df_op["WEEK"] = pd.Series([pd.NA] * len(df_op)).astype("Int64")

    # Reset index to avoid grouping issues (this is the core fix for "Grouper for 'YEAR' not 1-dimensional")
    df_op = df_op.reset_index(drop=True)

    return df_op, None

# load operational sheet (for reliability metrics)
df_op_proc, df_op_err = load_operational_sheet(RAW_URL)
# df_op_proc may be None if sheet not found; we handle that where used

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters & Options")

# Year multi-select (global filter) - default all years selected
available_years = []
if "YEAR" in df.columns and df["YEAR"].notna().any():
    available_years = sorted(df["YEAR"].dropna().unique().astype(int).tolist(), reverse=True)  # descending default
# allow the case where df_op_proc has additional years
if df_op_proc is not None and "YEAR" in df_op_proc.columns and df_op_proc["YEAR"].notna().any():
    yrs_op = sorted(df_op_proc["YEAR"].dropna().unique().astype(int).tolist(), reverse=True)
    for y in yrs_op:
        if y not in available_years:
            available_years.append(y)
available_years = sorted(list(set(available_years)), reverse=True)

if available_years:
    selected_years = st.sidebar.multiselect("Filter years (global)", options=available_years, default=available_years)
else:
    selected_years = []

granularity = st.sidebar.selectbox("Time granularity", options=["WEEK", "PERIOD_MONTH"], index=1)

# Build month list from PERIOD_MONTH (chronologically) — this fixes mismatch when weeks were reassigned to later month
months_available = []
if "PERIOD_MONTH" in df.columns:
    unique_pm = pd.Series(df["PERIOD_MONTH"].dropna().astype(str).str.strip().unique())
    parsed = pd.to_datetime(unique_pm, format="%b %Y", errors="coerce")
    if parsed.notna().any():
        months_df = pd.DataFrame({"PERIOD_MONTH": unique_pm.values, "period_dt": parsed.values})
        months_df = months_df.sort_values("period_dt")
        months_available = months_df["PERIOD_MONTH"].tolist()
# fallback to previous MONTH+YEAR logic if PERIOD_MONTH parsing failed
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

# Default to latest available month if exists
if months_available:
    try:
        default_idx = len(months_available) - 1
    except Exception:
        default_idx = 0
else:
    default_idx = 0

months = ["All"] + months_available
selected_month = st.sidebar.selectbox("MONTH", months, index=0)

# Apply global filters to df (copy so we don't mutate original)
filtered = df.copy()

# Apply year filter if selected (global)
if selected_years:
    # selected_years is list of ints
    filtered = filtered[filtered["YEAR"].isin(selected_years)]

# Apply month filter if selected
if selected_month != "All" and selected_month != "":
    filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]

# Apply same year filter to operational dataframe if loaded
if df_op_proc is not None:
    df_op_for_reliability = df_op_proc.copy()
    if selected_years:
        # YEAR may be Int64 dtype; compare as int
        df_op_for_reliability = df_op_for_reliability[df_op_for_reliability["YEAR"].isin(selected_years)]
else:
    df_op_for_reliability = None

# -------------------------
# TAB LAYOUT: Main Dashboard | MTBF & MTTR | Reliability Metrics
# -------------------------
tab_main, tab_mtbf_mttr, tab_reliability = st.tabs(["Main Dashboard", "MTBF & MTTR", "Reliability Metrics"])

# -------------------------
# MAIN DASHBOARD CONTENT (kept as originally)
# -------------------------
with tab_main:
    # -------------------------
    # KPI Calculations (compute BEFORE applying category drilldown)
    # -------------------------
    total_delay = filtered["DELAY"].sum()

    # AVAILABLE_TIME aggregation: robust approach — max per PERIOD_MONTH then sum
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
    # Top Row: KPIs + Donuts
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

        # Show monthly (current filter) KPIs
        st.metric("Physical Availability (PA)", f"{PA:.2%}" if PA is not None else "N/A", delta=f"Target {pa_target:.2%}")
        st.metric("Mechanical Availability (MA)", f"{MA:.2%}" if MA is not None else "N/A", delta=f"Target {ma_target:.2%}")
        st.metric("Total Delay Hours (selected)", f"{total_delay:.2f} hrs")
        st.metric("Total Available Time (selected)", f"{available_time:.2f} hrs" if available_time else "N/A")

        # YTD row
        if ytd_PA is not None:
            st.write("")  # spacing
            st.caption(f"YTD (up to selected): PA {ytd_PA:.2%} | MA {ytd_MA:.2%} | Delay {ytd_total_delay:.2f} hrs")
        else:
            st.write("")

    with donut1_col:
        st.subheader("Delay by Category")
        if "CATEGORY" in filtered.columns:
            donut_data = filtered.groupby("CATEGORY", dropna=False)["DELAY"].sum().reset_index().sort_values("DELAY", ascending=False)
            if not donut_data.empty:
                # round values for display
                donut_data["DELAY"] = donut_data["DELAY"].round(2)
                donut_fig = go.Figure(data=[go.Pie(labels=donut_data["CATEGORY"], values=donut_data["DELAY"], hole=0.4, textinfo="label+percent", hovertemplate="%{label}: %{value:.2f} hrs<extra></extra>")])
                donut_fig.update_layout(margin=dict(t=20,b=20,l=20,r=20))
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
                    # round values for display
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

    # ---------- (UPDATED) Use GLOBAL latest week for 52-week cutoff ----------
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

    # ---------- swap visuals & coloring (PA bars, Delay line) ----------
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
    fig_trend.add_trace(
        go.Bar(
            x=trend[x_field],
            y=trend["PA_pct_rounded"],
            name="PA%",
            marker=dict(color=colors),
            hovertemplate="%{y:.2%}<extra></extra>"
        )
    )
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

    fig_trend.add_shape(type="line", x0=0, x1=1, xref="paper", y0=pa_target, y1=pa_target, yref="y", line=dict(color="green", dash="dash"))
    fig_trend.add_annotation(x=0, xref="paper", y=pa_target, yref="y", showarrow=False, text=f"PA Target {pa_target:.2%}", font=dict(color="green"), align="left", xanchor="left", yanchor="bottom")

    fig_trend.update_layout(
        xaxis_title="Period",
        yaxis=dict(title="PA%", overlaying=None, side="left", tickformat=".2%", range=[0,1]),
        yaxis2=dict(title="Delay Hours", overlaying="y", side="right"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.02, yanchor="bottom"),
        margin=dict(t=70)
    )
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
    # Move legend above chart (so it does not overlap equipment names)
    fig_pareto.update_layout(xaxis_tickangle=-45, yaxis_title="Hours", legend=dict(orientation="h", x=0.5, xanchor="center", y=1.15, yanchor="bottom"), margin=dict(t=110))
    fig_pareto.update_yaxes(title_text="Cumulative %", tickformat=".2%", range=[0, 1], secondary_y=True)
    fig_pareto.update_yaxes(title_text="Delay Hours", secondary_y=False)

    st.plotly_chart(fig_pareto, use_container_width=True)

    # -------------------------
    # CATEGORY FILTER (below Pareto) -> directly drives Drilldown
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
    # Drill-down table (directly showing rows for the selected category)
    # -------------------------
    st.subheader(f"Drill-down data (filtered by selected category) - Years: {', '.join(map(str, selected_years)) if selected_years else 'All'}")

    details_df = drill_df_base.copy()

    # -------- (UPDATED) show assigned month in drilldown: use PERIOD_MONTH when available --------
    if "PERIOD_MONTH" in details_df.columns:
        # overwrite MONTH for display so week -> latest month assignment is visible in the table
        details_df["MONTH"] = details_df["PERIOD_MONTH"]

    # Ensure required columns exist
    required_cols = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "EQ_DESC", "DELAY", "NOTE", "SUB_CATEGORY", "YEAR"]
    for c in required_cols:
        if c not in details_df.columns:
            details_df[c] = ""

    # Prepare output DataFrame
    details_out = details_df[["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "EQ_DESC", "DELAY", "NOTE", "SUB_CATEGORY", "YEAR"]].copy()

    # Rename EQ_DESC -> Equipment Description for display
    details_out = details_out.rename(columns={"EQ_DESC": "Equipment Description"})

    # Reorder columns depending on category selection
    if selected_category == "MAINTENANCE (ALL)":
        ordered = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "SUB_CATEGORY", "Equipment Description", "DELAY", "NOTE"]
    else:
        ordered = ["WEEK", "MONTH", "DATE", "START", "STOP", "EQUIPMENT", "Equipment Description", "DELAY", "NOTE"]

    # Some columns may not exist (safeguard)
    ordered = [c for c in ordered if c in details_out.columns]

    # Convert WEEK to numeric for stable sorting
    details_out["WEEK"] = pd.to_numeric(details_out["WEEK"], errors="coerce")

    # Sort by YEAR, WEEK, START if YEAR exists, else by WEEK, START
    if "YEAR" in details_out.columns and details_out["YEAR"].notna().any():
        details_out["YEAR"] = pd.to_numeric(details_out["YEAR"], errors="coerce")
        details_out = details_out.sort_values(by=["YEAR", "WEEK", "START"], ascending=[True, True, True]).reset_index(drop=True)
        details_out = details_out.drop(columns=["YEAR"], errors="ignore")
    else:
        details_out = details_out.sort_values(by=["WEEK", "START"], ascending=[True, True]).reset_index(drop=True)

    # Only keep ordered columns for display
    details_out = details_out[ordered].reset_index(drop=True)

    # ---------- ROUND DELAY column to 2 decimals for display ----------
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

    # Show using AgGrid (no pagination, scrollable)
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
# MTBF & MTTR TAB
# -------------------------
with tab_mtbf_mttr:
    st.subheader("MTBF & MTTR time series (weekly & monthly)")

    if df_op_proc is None:
        st.info("Operational sheet not found or could not be loaded. MTBF/MTTR computations are unavailable.")
    else:
        # Use the processed df_op_proc we loaded earlier; apply selected_years global filter
        df_op = df_op_proc.copy()
        if selected_years:
            df_op = df_op[df_op["YEAR"].isin(selected_years)]

        # Ensure YEAR/WEEK columns present and correct (defensive)
        # --- THIS IS THE CAREFUL FIX: ensure YEAR & WEEK are 1D columns and index reset
        df_op = df_op.reset_index(drop=True)
        if "DATE" in df_op.columns:
            # if DATE is a string column, parse it
            try:
                df_op["_date_parsed2"] = pd.to_datetime(df_op["DATE"], errors="coerce")
                iso_series = df_op["_date_parsed2"].apply(lambda dt: pd.NaT if pd.isna(dt) else dt.isocalendar())
                def _iso2(t):
                    try:
                        if t is pd.NaT:
                            return (pd.NA, pd.NA)
                        return (int(t.year), int(t.week))
                    except Exception:
                        return (pd.NA, pd.NA)
                iso_tuples = iso_series.apply(_iso2)
                df_op["YEAR"] = iso_tuples.apply(lambda t: t[0]).astype("Int64")
                df_op["WEEK"] = iso_tuples.apply(lambda t: t[1]).astype("Int64")
            except Exception:
                pass
        # Additional safety: convert YEAR/WEEK to numeric
        if "YEAR" in df_op.columns:
            df_op["YEAR"] = pd.to_numeric(df_op["YEAR"], errors="coerce").astype("Int64")
        if "WEEK" in df_op.columns:
            df_op["WEEK"] = pd.to_numeric(df_op["WEEK"], errors="coerce").astype("Int64")

        # Group weekly
        weekly_group = df_op.groupby(["YEAR","WEEK"], dropna=False).agg(
            count_maintenance = ("_op_hours_dec", "count"),
            total_operational_hours = ("_op_hours_dec", "sum"),
            total_maintenance_delay = ("_maint_delay_num", "sum")
        ).reset_index()

        # compute MTBF and MTTR per week (count as denominator)
        weekly_group["MTBF_hours"] = weekly_group.apply(lambda r: (r["total_operational_hours"] / r["count_maintenance"]) if (r["count_maintenance"] and r["count_maintenance"]>0) else pd.NA, axis=1)
        weekly_group["MTTR_hours"] = weekly_group.apply(lambda r: (r["total_maintenance_delay"] / r["count_maintenance"]) if (r["count_maintenance"] and r["count_maintenance"]>0) else pd.NA, axis=1)

        # create period label and week_start for sorting
        def _week_label(row):
            try:
                return f"{int(row['YEAR'])} W{int(row['WEEK'])}"
            except Exception:
                return ""
        def _week_start_calendar(row):
            try:
                return datetime.date.fromisocalendar(int(row["YEAR"]), int(row["WEEK"]), 1)
            except Exception:
                return pd.NaT
        weekly_group["period_label"] = weekly_group.apply(_week_label, axis=1)
        weekly_group["week_start"] = weekly_group.apply(_week_start_calendar, axis=1)
        weekly_group = weekly_group.sort_values("week_start", ascending=True).reset_index(drop=True)

        # Limit to latest 52 weeks globally (based on the latest week present in df_op)
        try:
            latest_week_start_global_op = weekly_group["week_start"].dropna().max()
            if pd.notna(latest_week_start_global_op):
                cutoff_op = latest_week_start_global_op - timedelta(weeks=51)
                weekly_group_limited = weekly_group[weekly_group["week_start"] >= cutoff_op].copy()
                if weekly_group_limited.empty:
                    weekly_group_limited = weekly_group.copy()
            else:
                weekly_group_limited = weekly_group.copy()
        except Exception:
            weekly_group_limited = weekly_group.copy()

        # Monthly group
        # Make PERIOD_MONTH using week end logic or date
        df_op["_period_month_calc"] = None
        try:
            if "_date_parsed" in df_op.columns and df_op["_date_parsed"].notna().any():
                # choose the month of the date
                df_op["_period_month_calc"] = pd.to_datetime(df_op["_date_parsed"]).dt.strftime("%b %Y")
            elif "YEAR" in df_op.columns and "WEEK" in df_op.columns:
                # fallback: use week_start + 6 days -> month that contains week end
                week_start_series = df_op.apply(lambda r: datetime.date.fromisocalendar(int(r["YEAR"]), int(r["WEEK"]), 1) if (pd.notna(r["YEAR"]) and pd.notna(r["WEEK"])) else pd.NaT, axis=1)
                week_end = pd.to_datetime(week_start_series) + pd.Timedelta(days=6)
                df_op["_period_month_calc"] = week_end.dt.strftime("%b %Y")
            else:
                df_op["_period_month_calc"] = pd.Series([""] * len(df_op))
        except Exception:
            df_op["_period_month_calc"] = pd.Series([""] * len(df_op))

        monthly_group = df_op.groupby("_period_month_calc", dropna=False).agg(
            count_maintenance = ("_op_hours_dec", "count"),
            total_operational_hours = ("_op_hours_dec", "sum"),
            total_maintenance_delay = ("_maint_delay_num", "sum")
        ).reset_index().rename(columns={"_period_month_calc": "PERIOD_MONTH"})
        monthly_group["MTBF_hours"] = monthly_group.apply(lambda r: (r["total_operational_hours"] / r["count_maintenance"]) if (r["count_maintenance"] and r["count_maintenance"]>0) else pd.NA, axis=1)
        monthly_group["MTTR_hours"] = monthly_group.apply(lambda r: (r["total_maintenance_delay"] / r["count_maintenance"]) if (r["count_maintenance"] and r["count_maintenance"]>0) else pd.NA, axis=1)
        # sort monthly by parsed month
        monthly_group["period_dt"] = pd.to_datetime(monthly_group["PERIOD_MONTH"], format="%b %Y", errors="coerce")
        monthly_group = monthly_group.sort_values("period_dt").reset_index(drop=True)

        # ----- Plotting: 4 charts (MTTR weekly, MTTR monthly, MTBF weekly, MTBF monthly) -----
        st.markdown("**MTTR (weekly)**")
        fig_mttr_w = go.Figure()
        fig_mttr_w.add_trace(go.Bar(x=weekly_group_limited["period_label"], y=weekly_group_limited["MTTR_hours"].round(2), name="MTTR (hrs)"))
        fig_mttr_w.add_trace(go.Scatter(x=weekly_group_limited["period_label"], y=weekly_group_limited["count_maintenance"], name="Maintenance Count", yaxis="y2", mode="lines+markers"))
        fig_mttr_w.update_layout(yaxis2=dict(overlaying="y", side="right", title="Count"), legend=dict(orientation="h"), margin=dict(t=40))
        st.plotly_chart(fig_mttr_w, use_container_width=True)

        st.markdown("**MTTR (monthly)**")
        fig_mttr_m = go.Figure()
        fig_mttr_m.add_trace(go.Bar(x=monthly_group["PERIOD_MONTH"], y=monthly_group["MTTR_hours"].round(2), name="MTTR (hrs)"))
        fig_mttr_m.add_trace(go.Scatter(x=monthly_group["PERIOD_MONTH"], y=monthly_group["count_maintenance"], name="Maintenance Count", yaxis="y2", mode="lines+markers"))
        fig_mttr_m.update_layout(yaxis2=dict(overlaying="y", side="right", title="Count"), legend=dict(orientation="h"), margin=dict(t=40))
        st.plotly_chart(fig_mttr_m, use_container_width=True)

        st.markdown("**MTBF (weekly)**")
        fig_mtbf_w = go.Figure()
        fig_mtbf_w.add_trace(go.Bar(x=weekly_group_limited["period_label"], y=weekly_group_limited["MTBF_hours"].round(2), name="MTBF (hrs)"))
        fig_mtbf_w.add_trace(go.Scatter(x=weekly_group_limited["period_label"], y=weekly_group_limited["count_maintenance"], name="Maintenance Count", yaxis="y2", mode="lines+markers"))
        fig_mtbf_w.update_layout(yaxis2=dict(overlaying="y", side="right", title="Count"), legend=dict(orientation="h"), margin=dict(t=40))
        st.plotly_chart(fig_mtbf_w, use_container_width=True)

        st.markdown("**MTBF (monthly)**")
        fig_mtbf_m = go.Figure()
        fig_mtbf_m.add_trace(go.Bar(x=monthly_group["PERIOD_MONTH"], y=monthly_group["MTBF_hours"].round(2), name="MTBF (hrs)"))
        fig_mtbf_m.add_trace(go.Scatter(x=monthly_group["PERIOD_MONTH"], y=monthly_group["count_maintenance"], name="Maintenance Count", yaxis="y2", mode="lines+markers"))
        fig_mtbf_m.update_layout(yaxis2=dict(overlaying="y", side="right", title="Count"), legend=dict(orientation="h"), margin=dict(t=40))
        st.plotly_chart(fig_mtbf_m, use_container_width=True)

# -------------------------
# RELIABILITY METRICS TAB (small matrix + table)
# -------------------------
with tab_reliability:
    st.subheader("Reliability Metrics (debug & grouping checks)")

    if df_op_proc is None:
        st.info("Operational sheet not found or could not be loaded. Reliability metrics unavailable.")
    else:
        # Defensive: ensure YEAR & WEEK are present and 1-D
        df_op_check = df_op_proc.copy()
        df_op_check = df_op_check.reset_index(drop=True)

        # parse DATE if present and ensure isocalendar YEAR/WEEK
        if "DATE" in df_op_check.columns:
            try:
                df_op_check["_date_parsed"] = pd.to_datetime(df_op_check["DATE"], errors="coerce")
                iso_series = df_op_check["_date_parsed"].apply(lambda dt: pd.NaT if pd.isna(dt) else dt.isocalendar())
                def _iso_to_tuple(x):
                    try:
                        if x is pd.NaT:
                            return (pd.NA, pd.NA)
                        return (int(x.year), int(x.week))
                    except Exception:
                        return (pd.NA, pd.NA)
                iso_t = iso_series.apply(_iso_to_tuple)
                df_op_check["YEAR"] = iso_t.apply(lambda t: t[0]).astype("Int64")
                df_op_check["WEEK"] = iso_t.apply(lambda t: t[1]).astype("Int64")
            except Exception:
                pass

        # ensure numeric
        if "YEAR" in df_op_check.columns:
            df_op_check["YEAR"] = pd.to_numeric(df_op_check["YEAR"], errors="coerce").astype("Int64")
        if "WEEK" in df_op_check.columns:
            df_op_check["WEEK"] = pd.to_numeric(df_op_check["WEEK"], errors="coerce").astype("Int64")

        # reset index one more time (core fix)
        df_op_check = df_op_check.reset_index(drop=True)

        # Now we can safely group by YEAR and WEEK without "not 1-dimensional" error
        try:
            weekly_group_check = df_op_check.groupby(["YEAR","WEEK"], dropna=False).agg(
                count_maintenance = ("_op_hours_dec", "count"),
                total_operational_hours = ("_op_hours_dec", "sum"),
                total_maintenance_delay = ("_maint_delay_num", "sum")
            ).reset_index()
            weekly_group_check["MTBF_hours"] = weekly_group_check.apply(lambda r: (r["total_operational_hours"] / r["count_maintenance"]) if (r["count_maintenance"] and r["count_maintenance"]>0) else pd.NA, axis=1)
            weekly_group_check["MTTR_hours"] = weekly_group_check.apply(lambda r: (r["total_maintenance_delay"] / r["count_maintenance"]) if (r["count_maintenance"] and r["count_maintenance"]>0) else pd.NA, axis=1)
        except Exception as e:
            st.error(f"Error while grouping operational data: {e}")
            weekly_group_check = None

        st.write("Operational sheet load status and a sample (first 5 rows):")
        st.write(_mtbf_mttr_res.get("df_op_processed_sample", "No sample available."))

        if weekly_group_check is not None:
            # sort descending by YEAR/WEEK
            weekly_group_check = weekly_group_check.sort_values(by=["YEAR","WEEK"], ascending=[False, False]).reset_index(drop=True)
            st.write("Weekly reliability summary (descending by YEAR,WEEK):")
            # show interactive table
            gob = GridOptionsBuilder.from_dataframe(weekly_group_check)
            gob.configure_default_column(editable=False, sortable=True, filter=True, resizable=True)
            AgGrid(weekly_group_check, gridOptions=gob.build(), height=300)
        else:
            st.info("Weekly grouping couldn't be performed; check the operational sheet format.")

# End of file
