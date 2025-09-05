import streamlit as st
import pandas as pd
import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# AgGrid
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode

st.set_page_config(page_title="Physical Availability - Data Delay Time", layout="wide")

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
    if "PERIOD_MONTH" not in df.columns or df["PERIOD_MONTH"].isnull().all() or (df["PERIOD_MONTH"].astype(str).str.strip()=="" ).all():
        if "MONTH" in df.columns and "YEAR" in df.columns:
            df["PERIOD_MONTH"] = df["MONTH"].astype(str).str.strip() + " " + df["YEAR"].astype(str)

    # trim
    if "PERIOD_MONTH" in df.columns:
        df["PERIOD_MONTH"] = df["PERIOD_MONTH"].astype(str).str.strip()

    # drop rows with no DELAY numeric
    df = df[df["DELAY"].notna()].copy()

    # remove years before 2024
    if "YEAR" in df.columns:
        df = df[df["YEAR"].notna() & (df["YEAR"] >= 2024)]

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

    return df

# -------------------------
# Load data
# -------------------------
df = load_data_from_url()
if df is None:
    st.stop()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters & Options")
granularity = st.sidebar.selectbox("Time granularity", options=["WEEK", "PERIOD_MONTH"], index=1)

# Build month list and sort chronologically using MONTH + YEAR (keep "JAN 2024" strings as-is)
months_available = []
if "MONTH" in df.columns and "YEAR" in df.columns:
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
    # sort chronologically and deduplicate while preserving order
    tmp_sorted = sorted(set(tmp), key=lambda x: (x[0], x[1]))
    months_available = [t[2] for t in tmp_sorted]

# If PERIOD_MONTH already present as strings (maybe different casing), ensure we include them too but keep canonical "MON YYYY".
# We'll use months_available above; if it's empty but PERIOD_MONTH exists, fallback to unique PERIOD_MONTH.
if not months_available and "PERIOD_MONTH" in df.columns:
    months_available = pd.Series(df["PERIOD_MONTH"].unique()).dropna().astype(str).str.strip().tolist()

# Default to latest available month if exists
if months_available:
    # latest by sorting the month tuples
    last = months_available[-1]
    try:
        default_idx = len(months_available) - 1
    except Exception:
        default_idx = 0
else:
    default_idx = 0
    
months = ["All"] + months_available
selected_month = st.sidebar.selectbox("MONTH", months, index=0)

filtered = df.copy()
if selected_month != "All" and selected_month != "":
    # match exact PERIOD_MONTH strings (they are in "JAN 2024" format)
    filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]

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
    st.metric("Physical Availability (PA)", f"{PA:.1%}" if PA is not None else "N/A", delta=f"Target {pa_target:.0%}")
    st.metric("Maintenance Availability (MA)", f"{MA:.1%}" if MA is not None else "N/A", delta=f"Target {ma_target:.0%}")
    st.metric("Total Delay Hours (selected)", f"{total_delay:.2f} hrs")
    st.metric("Total Available Time (selected)", f"{available_time:.2f} hrs" if available_time else "N/A")

    # YTD row
    if ytd_PA is not None:
        st.write("")  # spacing
        st.caption(f"YTD (up to selected): PA {ytd_PA:.1%} | MA {ytd_MA:.1%} | Delay {ytd_total_delay:.2f} hrs")
    else:
        st.write("")

with donut1_col:
    st.subheader("Delay by Category")
    if "CATEGORY" in filtered.columns:
        donut_data = filtered.groupby("CATEGORY", dropna=False)["DELAY"].sum().reset_index().sort_values("DELAY", ascending=False)
        if not donut_data.empty:
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

if group_field == "WEEK":
    trend = filtered.groupby(["YEAR","WEEK"], dropna=False).agg(
        total_delay_hours=("DELAY","sum"),
        available_time_month=("AVAILABLE_TIME_MONTH","max"),
        available_hours=("AVAILABLE_HOURS","max")
    ).reset_index()
    trend["period_label"] = trend["YEAR"].astype(str) + " W" + trend["WEEK"].astype("Int64").astype(str)
    trend = trend.sort_values(by=["YEAR","WEEK"])
    x_field = "period_label"
elif group_field == "PERIOD_MONTH":
    trend = filtered.groupby("PERIOD_MONTH", dropna=False).agg(
        total_delay_hours=("DELAY","sum"),
        available_time_month=("AVAILABLE_TIME_MONTH","max"),
        available_hours=("AVAILABLE_HOURS","max")
    ).reset_index()
    # parse PERIOD_MONTH strings (format "JAN 2024") for sorting, but keep labels as strings
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

fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(x=trend[x_field], y=trend["total_delay_hours"], name="Total Delay Hours"))
fig_trend.add_trace(go.Scatter(x=trend[x_field], y=trend["PA_pct"], name="PA%", yaxis="y2", mode="lines+markers"))

fig_trend.add_shape(type="line", x0=0, x1=1, xref="paper", y0=pa_target, y1=pa_target, yref="y2", line=dict(color="green", dash="dash"))
fig_trend.add_annotation(x=0, xref="paper", y=pa_target, yref="y2", showarrow=False, text=f"PA Target {pa_target:.0%}", font=dict(color="green"), align="left", xanchor="left", yanchor="bottom")

fig_trend.update_layout(
    xaxis_title="Period",
    yaxis_title="Delay Hours",
    yaxis2=dict(title="PA%", overlaying="y", side="right", tickformat="%", range=[0,1]),
    legend=dict(orientation="h"),
    margin=dict(t=30)
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
fig_pareto.add_trace(go.Bar(x=pareto_df[equipment_key], y=pareto_df["hours"], name="Hours"), secondary_y=False)
fig_pareto.add_trace(go.Scatter(x=pareto_df[equipment_key], y=pareto_df["cum_pct"], name="Cumulative %", mode="lines+markers"), secondary_y=True)
fig_pareto.update_layout(xaxis_tickangle=-45, yaxis_title="Hours", legend=dict(orientation="h"), margin=dict(t=30))
fig_pareto.update_yaxes(title_text="Cumulative %", tickformat="%", range=[0, 1], secondary_y=True)

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
st.subheader("Drill-down data (filtered by selected category)")

details_df = drill_df_base.copy()

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
