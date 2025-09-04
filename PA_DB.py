import streamlit as st
import pandas as pd
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
    # Use st.image to display the logo
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

    # Detect header row
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

    # Normalize column names if present
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
        "CATEGORY": "CATEGORY",
    }
    for orig, new in replacements.items():
        for col in df.columns:
            if str(col).strip().upper() == orig:
                df.rename(columns={col: new}, inplace=True)
                break

    # Ensure essential columns
    essential = ["WEEK", "MONTH", "YEAR", "DELAY"]
    for c in essential:
        if c not in df.columns:
            st.error(f"Expected column '{c}' not found after cleaning. Found columns: {df.columns.tolist()}")
            return None

    # Convert numeric
    df["DELAY"] = pd.to_numeric(df["DELAY"], errors="coerce")

    if "AVAILABLE_TIME_MONTH" in df.columns:
        df["AVAILABLE_TIME_MONTH"] = pd.to_numeric(df["AVAILABLE_TIME_MONTH"], errors="coerce")
    else:
        df["AVAILABLE_TIME_MONTH"] = None

    # Keep START/STOP as raw strings for display but compute durations
    if "START" in df.columns and "STOP" in df.columns:
        # Keep raw strings (clean)
        df["START"] = df["START"].astype(str).str.strip()
        df["STOP"] = df["STOP"].astype(str).str.strip()

        import datetime

        def parse_time_to_hours(start_str, stop_str):
            try:
                # Accept H:M or HH:MM formats; if seconds present, this still works
                start_t = datetime.datetime.strptime(start_str, "%H:%M")
                stop_t  = datetime.datetime.strptime(stop_str, "%H:%M")
                diff = (stop_t - start_t).total_seconds() / 3600
                if diff < 0:  # crossed midnight
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

    # Fill category-like columns (create if missing)
    for cat in ["MTN_DELAY_TYPE","SCH_MTN","UNSCH_MTN","MINING_DELAY","WEATHER_DELAY","OTHER_DELAY","MTN_NOTE","NOTE","EQUIPMENT","CATEGORY"]:
        if cat in df.columns:
            df[cat] = df[cat].fillna("").astype(str)
        else:
            df[cat] = ""

    # Create CATEGORY if absent or incomplete (prefer MTN / Mining / Weather / Other)
    def determine_category(r):
        if r.get("MTN_DELAY_TYPE") and str(r.get("MTN_DELAY_TYPE")).strip() != "":
            return "Maintenance"
        if r.get("MINING_DELAY") and str(r.get("MINING_DELAY")).strip() != "":
            return "Mining"
        if r.get("WEATHER_DELAY") and str(r.get("WEATHER_DELAY")).strip() != "":
            return "Weather"
        if r.get("OTHER_DELAY") and str(r.get("OTHER_DELAY")).strip() != "":
            return "Other"
        # keep existing CATEGORY if provided otherwise Unknown
        if r.get("CATEGORY") and str(r.get("CATEGORY")).strip() not in ("", "nan"):
            return str(r.get("CATEGORY")).strip()
        return "Unknown"

    df["CATEGORY"] = df.apply(determine_category, axis=1)

    # Compose a CAUSE column (useful for pareto when equipment not present)
    def compose_cause(r):
        parts = []
        for c in ["MTN_DELAY_TYPE","SCH_MTN","UNSCH_MTN","MINING_DELAY","WEATHER_DELAY","OTHER_DELAY","MTN_NOTE","NOTE"]:
            v = r.get(c)
            if v is not None:
                vs = str(v).strip()
                if vs and vs.lower() != "nan":
                    parts.append(vs)
        return " | ".join(parts)
    df["CAUSE"] = df.apply(compose_cause, axis=1)

    # YEAR/WEEK normalization (numeric)
    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce").astype("Int64")

    # Add PERIOD_MONTH if missing
    if "PERIOD_MONTH" not in df.columns and "MONTH" in df.columns and "YEAR" in df.columns:
        df["PERIOD_MONTH"] = df["MONTH"].astype(str).str.strip() + " " + df["YEAR"].astype(str)

    # Drop invalid DELAY rows
    df = df[df["DELAY"].notna()].copy()

    # Bruteforce: remove years before 2024
    if "YEAR" in df.columns:
        df = df[df["YEAR"].notna() & (df["YEAR"] >= 2024)]

    # Ensure PERIOD_MONTH values are trimmed
    if "PERIOD_MONTH" in df.columns:
        df["PERIOD_MONTH"] = df["PERIOD_MONTH"].astype(str).str.strip()

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
granularity = st.sidebar.selectbox("Time granularity", options=["WEEK","PERIOD_MONTH"], index=1)

# Build month list and sort chronologically using parsed dates
if "PERIOD_MONTH" in df.columns:
    period_dt = pd.to_datetime(df["PERIOD_MONTH"], format="%b %Y", errors="coerce")
    months_df = pd.DataFrame({"PERIOD_MONTH": df["PERIOD_MONTH"], "period_dt": period_dt})
    months_available = (
        months_df.dropna(subset=["period_dt"]).drop_duplicates("PERIOD_MONTH").sort_values("period_dt")["PERIOD_MONTH"].tolist()
    )
    # include any unparsable months at end (unique)
    unparsable = [x for x in df["PERIOD_MONTH"].unique().tolist() if x not in months_available and pd.isna(pd.to_datetime(x, format="%b %Y", errors="coerce"))]
    months_available += unparsable
else:
    months_available = []

selected_month = st.sidebar.selectbox("Month", options=["All"] + months_available, index=0)

filtered = df.copy()
if selected_month != "All" and selected_month != "":
    filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]

# -------------------------
# KPI Calculations
# -------------------------
total_delay = filtered["DELAY"].sum()

if "AVAILABLE_TIME_MONTH" in filtered.columns and filtered["AVAILABLE_TIME_MONTH"].notna().any():
    available_time = filtered.drop_duplicates("PERIOD_MONTH")["AVAILABLE_TIME_MONTH"].dropna().sum()
elif "AVAILABLE_HOURS" in filtered.columns and filtered["AVAILABLE_HOURS"].notna().any():
    per_period_avail = filtered.groupby("PERIOD_MONTH")["AVAILABLE_HOURS"].max().dropna()
    available_time = per_period_avail.sum()
else:
    available_time = None

PA = max(0, 1 - total_delay / available_time) if (available_time and available_time > 0) else None
maintenance_delay = filtered[filtered["CATEGORY"] == "Maintenance"]["DELAY"].sum() if "CATEGORY" in filtered.columns else 0
MA = max(0, 1 - maintenance_delay / available_time) if (available_time and available_time > 0) else None

pa_target = filtered["PA_TARGET"].dropna().unique().tolist() if "PA_TARGET" in filtered.columns else []
ma_target = filtered["MA_TARGET"].dropna().unique().tolist() if "MA_TARGET" in filtered.columns else []
pa_target = pa_target[0] if pa_target else 0.9
ma_target = ma_target[0] if ma_target else 0.85
if pa_target > 1: pa_target = pa_target / 100.0
if ma_target > 1: ma_target = ma_target / 100.0

# -------------------------
# Top Row: KPIs + Donuts (Delay by Category restored)
# -------------------------
kpi_col, donut1_col, donut2_col = st.columns([1,2,2])
with kpi_col:
    st.subheader("Key KPIs")
    # Build a readable date range for caption using PERIOD_MONTH if possible
    min_caption = None
    max_caption = None
    if "PERIOD_MONTH" in filtered.columns and not filtered["PERIOD_MONTH"].dropna().empty:
        parsed = pd.to_datetime(filtered["PERIOD_MONTH"].dropna().unique(), format="%b %Y", errors="coerce")
        if parsed.notna().any():
            min_dt = parsed.min()
            max_dt = parsed.max()
            if pd.notna(min_dt) and pd.notna(max_dt):
                min_caption = min_dt.strftime("%d/%m/%Y")
                # last day of max month:
                max_month_end = (max_dt + pd.offsets.MonthEnd(0)).strftime("%d/%m/%Y")
                max_caption = max_month_end
    # fallback to YEAR range if PERIOD_MONTH not parsable
    if min_caption is None and "YEAR" in filtered.columns and filtered["YEAR"].notna().any():
        min_y = int(filtered["YEAR"].min())
        max_y = int(filtered["YEAR"].max())
        min_caption = f"01/01/{min_y}"
        max_caption = f"31/12/{max_y}"

    if min_caption and max_caption:
        st.caption(f"Data obtained from {min_caption} to {max_caption}")
    else:
        st.caption("Data obtained from unknown date range")

    st.metric("Physical Availability (PA)", f"{PA:.1%}" if PA is not None else "N/A", delta=f"Target {pa_target:.0%}")
    st.metric("Maintenance Availability (MA)", f"{MA:.1%}" if MA is not None else "N/A", delta=f"Target {ma_target:.0%}")
    st.metric("Total Delay Hours", f"{total_delay:.2f} hrs")
    st.metric("Total Available Time", f"{available_time:.2f} hrs" if available_time else "N/A")

with donut1_col:
    st.subheader("Delay by Category")
    # Use CATEGORY created above
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
    st.subheader("By Maintenance")
    if "MTN_DELAY_TYPE" in filtered.columns:
        sched_data = filtered[filtered["MTN_DELAY_TYPE"].notna() & (filtered["MTN_DELAY_TYPE"] != "")].copy()
        # Remove blank/null mtn types
        sched_data = sched_data[sched_data["MTN_DELAY_TYPE"].str.strip() != ""]
        if not sched_data.empty:
            sched_donut = sched_data.groupby("MTN_DELAY_TYPE")["DELAY"].sum().reset_index().sort_values("DELAY", ascending=False)
            donut_fig2 = go.Figure(data=[go.Pie(labels=sched_donut["MTN_DELAY_TYPE"], values=sched_donut["DELAY"], hole=0.4, textinfo="label+percent", hovertemplate="%{label}: %{value:.2f} hrs<extra></extra>")])
            donut_fig2.update_layout(margin=dict(t=20,b=20,l=20,r=20))
            st.plotly_chart(donut_fig2, use_container_width=True)
        else:
            st.info("No maintenance breakdown available.")
    else:
        st.info("No MTN_DELAY_TYPE column available.")

st.markdown("---")

# -------------------------
# Trend Analysis
# -------------------------
st.subheader("Trend: Total Delay Hours vs PA%")

group_field = granularity

if group_field == "WEEK":
    # Group by YEAR+WEEK and ensure chronological sorting by YEAR then WEEK
    trend = filtered.groupby(["YEAR","WEEK"], dropna=False).agg(
        total_delay_hours=("DELAY","sum"),
        available_time_month=("AVAILABLE_TIME_MONTH","max"),
        available_hours=("AVAILABLE_HOURS","max")
    ).reset_index()
    # create readable label
    trend["period_label"] = trend["YEAR"].astype(str) + " W" + trend["WEEK"].astype("Int64").astype(str)
    trend = trend.sort_values(by=["YEAR","WEEK"])
    x_field = "period_label"
elif group_field == "PERIOD_MONTH":
    # Group by PERIOD_MONTH but sort using parsed month-year
    trend = filtered.groupby("PERIOD_MONTH", dropna=False).agg(
        total_delay_hours=("DELAY","sum"),
        available_time_month=("AVAILABLE_TIME_MONTH","max"),
        available_hours=("AVAILABLE_HOURS","max")
    ).reset_index()
    # parse PERIOD_MONTH to datetime for sorting (e.g. "Jan 2024")
    trend["period_dt"] = pd.to_datetime(trend["PERIOD_MONTH"], format="%b %Y", errors="coerce")
    # if parsing fails for some entries, they will sort last
    trend = trend.sort_values(by=["period_dt", "PERIOD_MONTH"])
    x_field = "PERIOD_MONTH"
else:
    # fallback
    trend = filtered.groupby(group_field).agg(total_delay_hours=("DELAY","sum"), available_time_month=("AVAILABLE_TIME_MONTH","max"), available_hours=("AVAILABLE_HOURS","max")).reset_index()
    x_field = group_field

# compute PA% per period
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

# Use EQUIPMENT if present; fallback to CAUSE
if "EQUIPMENT" in filtered.columns and filtered["EQUIPMENT"].notna().any():
    equipment_key = "EQUIPMENT"
    equipment_series = filtered["EQUIPMENT"].replace("", "(Unknown)")
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
# AgGrid: Top Equipment Table
# -------------------------
st.subheader("Details for Top Equipment")

equip_summary = pareto_df[[equipment_key, "hours"]].rename(columns={"hours": "Total Delay Hours"})
equip_summary = equip_summary.sort_values("Total Delay Hours", ascending=False).reset_index(drop=True)

gob = GridOptionsBuilder.from_dataframe(equip_summary)
gob.configure_selection(selection_mode="single", use_checkbox=True)
gob.configure_grid_options(pagination=False)  # no pagination
gob.configure_default_column(editable=False, sortable=True, filter=True, resizable=True)
grid_options = gob.build()

grid_resp = AgGrid(
    equip_summary,
    gridOptions=grid_options,
    update_mode=GridUpdateMode.SELECTION_CHANGED,
    allow_unsafe_jscode=False,
    height=400,
    fit_columns_on_grid_load=True,
    theme="balham"
)

selected_rows = grid_resp.get("selected_rows", [])

# Safe handling: AgGrid may return list-of-dicts
selected_equipment = None
if isinstance(selected_rows, list) and len(selected_rows) > 0:
    selected_equipment = selected_rows[0].get(equipment_key, None)
elif isinstance(selected_rows, pd.DataFrame) and not selected_rows.empty:
    selected_equipment = selected_rows.iloc[0].get(equipment_key, None)

# -------------------------
# Drill-down table (for chosen equipment/cause)
# -------------------------
if selected_equipment:
    st.markdown("---")
    st.subheader(f"Drill-down: {selected_equipment}")

    # match the same equipment_key used above
    if equipment_key == "EQUIPMENT":
        selector = filtered["EQUIPMENT"].replace("", "(Unknown)") == selected_equipment
    else:
        selector = filtered["CAUSE"].replace("", "(Unknown)") == selected_equipment

    details_df = filtered[selector].copy()

    columns_needed = ["WEEK", "MONTH", "DELAY", "CATEGORY", "START", "STOP", "MTN_NOTE", "NOTE"]
    for col in columns_needed:
        if col not in details_df.columns:
            details_df[col] = ""

    # sort by YEAR then WEEK, then START (start is string so it's stable)
    details_df["WEEK"] = pd.to_numeric(details_df["WEEK"], errors="coerce")
    if "YEAR" in details_df.columns and details_df["YEAR"].notna().any():
        details_df = details_df.sort_values(by=["YEAR", "WEEK", "START"], ascending=[True, True, True])
    else:
        details_df = details_df.sort_values(by=["WEEK", "START"], ascending=[True, True])

    # display START/STOP as raw strings (they were kept as strings)
    details_out = details_df[columns_needed].copy().reset_index(drop=True)

    # Show using AgGrid (no pagination)
    gob2 = GridOptionsBuilder.from_dataframe(details_out)
    gob2.configure_grid_options(pagination=False)
    gob2.configure_default_column(editable=False, sortable=True, filter=True, resizable=True)
    grid_options2 = gob2.build()

    AgGrid(
        details_out,
        gridOptions=grid_options2,
        update_mode=GridUpdateMode.NO_UPDATE,
        allow_unsafe_jscode=False,
        height=400,
        fit_columns_on_grid_load=True,
        theme="balham"
    )
