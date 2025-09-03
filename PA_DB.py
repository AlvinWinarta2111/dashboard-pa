import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Physical Availability - Data Delay Time", layout="wide")

st.title("Physical Availability Dashboard — Data Delay Time")
st.markdown(
    """
    Interactive Streamlit dashboard that reads the **'Data Delay Time'** sheet
    from GitHub and provides KPIs, breakdowns, trends, Pareto, and drill-down tables.
    """
)

# -------------------------
# Helper function
# -------------------------
def load_data_from_github():
    file_url = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/main/Draft_New%20Version_Weekly_Report_Maintenance_CHPP.xlsx"

    try:
        raw = pd.read_excel(file_url, sheet_name="Data Delay Time", header=None)
    except Exception as e:
        st.error(f"Unable to read sheet 'Data Delay Time' from GitHub: {e}")
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

    try:
        df = pd.read_excel(file_url, sheet_name="Data Delay Time", header=header_row)
    except Exception as e:
        st.error(f"Unable to re-read sheet with headers: {e}")
        return None

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
        "NOTE": "NOTE"
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

    for cat in ["MTN_DELAY_TYPE", "SCH_MTN", "UNSCH_MTN", "MINING_DELAY", "WEATHER_DELAY", "OTHER_DELAY", "MTN_NOTE", "NOTE"]:
        if cat in df.columns:
            df[cat] = df[cat].fillna("").astype(str)
        else:
            df[cat] = ""

    def determine_category(row):
        if row["MTN_DELAY_TYPE"]:
            return "Maintenance"
        if row["MINING_DELAY"]:
            return "Mining"
        if row["WEATHER_DELAY"]:
            return "Weather"
        if row["OTHER_DELAY"]:
            return "Other"
        return "Unknown"

    df["CATEGORY"] = df.apply(determine_category, axis=1)

    def compose_cause(row):
        parts = []
        if row["MTN_DELAY_TYPE"]: parts.append(row["MTN_DELAY_TYPE"])
        if row["SCH_MTN"]: parts.append(row["SCH_MTN"])
        if row["UNSCH_MTN"]: parts.append(row["UNSCH_MTN"])
        if row["MINING_DELAY"]: parts.append(row["MINING_DELAY"])
        if row["WEATHER_DELAY"]: parts.append(row["WEATHER_DELAY"])
        if row["OTHER_DELAY"]: parts.append(row["OTHER_DELAY"])
        if row["MTN_NOTE"]: parts.append(row["MTN_NOTE"])
        if row["NOTE"]: parts.append(row["NOTE"])
        return " | ".join([p for p in parts if p and str(p).strip() != "nan"])

    df["CAUSE"] = df.apply(compose_cause, axis=1)

    df = df[df["DELAY"].notna()].copy()

    try:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        df = df[df["YEAR"].notna()]
        df = df[(df["YEAR"] >= 2000) & (df["YEAR"] <= 2100)]
    except:
        pass

    try:
        df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce").astype("Int64")
    except:
        pass

    df["PERIOD_MONTH"] = df["MONTH"].astype(str) + " " + df["YEAR"].astype(str)
    df["PERIOD_YEAR"] = df["YEAR"].astype(str)

    return df

# -------------------------
# Load data
# -------------------------
df = load_data_from_github()
if df is None:
    st.stop()

# -------------------------
# Sidebar filters
# -------------------------
st.sidebar.header("Filters & Options")
granularity = st.sidebar.selectbox("Time granularity", options=["WEEK", "PERIOD_MONTH", "PERIOD_YEAR"], index=1)

months_available = df["PERIOD_MONTH"].dropna().unique().tolist()
months_available.sort(key=lambda x: (int(x.split()[-1]), pd.to_datetime(x.split()[0], format='%b').month))
selected_month = st.sidebar.selectbox("Month", options=["All"] + months_available, index=0)

category_filter = st.sidebar.multiselect("Delay categories", options=sorted(df["CATEGORY"].unique().tolist()), default=sorted(df["CATEGORY"].unique().tolist()))
show_notes = st.sidebar.checkbox("Show notes column in drill-down table", value=True)

filtered = df.copy()
if selected_month != "All":
    filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]
if category_filter:
    filtered = filtered[filtered["CATEGORY"].isin(category_filter)]

if granularity == "PERIOD_YEAR":
    selected_year = st.sidebar.selectbox("Year", options=["All"] + sorted(df["YEAR"].dropna().unique().tolist()), index=0)
    if selected_year != "All":
        filtered = filtered[filtered["YEAR"] == int(selected_year)]

if selected_month != "All":
    filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]

if category_filter:
    filtered = filtered[filtered["CATEGORY"].isin(category_filter)]

# -------------------------
# Aggregations
# -------------------------
group_field = granularity
agg = filtered.groupby(group_field).agg(
    total_delay_hours=pd.NamedAgg(column="DELAY", aggfunc="sum"),
    maintenance_delay_hours=pd.NamedAgg(column="DELAY", aggfunc=lambda x: x[df.loc[x.index, "CATEGORY"]=="Maintenance"].sum()),
    mining_delay_hours=pd.NamedAgg(column="DELAY", aggfunc=lambda x: x[df.loc[x.index, "CATEGORY"]=="Mining"].sum()),
    weather_delay_hours=pd.NamedAgg(column="DELAY", aggfunc=lambda x: x[df.loc[x.index, "CATEGORY"]=="Weather"].sum()),
    other_delay_hours=pd.NamedAgg(column="DELAY", aggfunc=lambda x: x[df.loc[x.index, "CATEGORY"]=="Other"].sum()),
    available_time_month=pd.NamedAgg(column="AVAILABLE_TIME_MONTH", aggfunc="first"),
).reset_index()

if group_field == "WEEK":
    agg = agg.sort_values("WEEK")
elif group_field == "PERIOD_MONTH":
    order = filtered["PERIOD_MONTH"].drop_duplicates().tolist()
    agg["__order"] = agg[group_field].apply(lambda v: order.index(v) if v in order else 999)
    agg = agg.sort_values("__order").drop(columns="__order")
elif group_field == "PERIOD_YEAR":
    agg = agg.sort_values("PERIOD_YEAR")

# -------------------------
# KPI calculations
# -------------------------
total_delay = filtered["DELAY"].sum()
available_time = filtered["AVAILABLE_TIME_MONTH"].dropna().sum() if filtered["AVAILABLE_TIME_MONTH"].notna().any() else None

if available_time and available_time > 0:
    PA = max(0, 1 - total_delay / available_time)
else:
    if "AVAILABLE_TIME_MONTH" in filtered.columns and filtered["AVAILABLE_TIME_MONTH"].notna().any():
        pa_vals = 1 - filtered["DELAY"] / filtered["AVAILABLE_TIME_MONTH"]
        PA = pa_vals.mean()
    else:
        PA = None

maintenance_delay = filtered[filtered["CATEGORY"]=="Maintenance"]["DELAY"].sum()
if available_time and available_time > 0:
    MA = max(0, 1 - maintenance_delay / available_time)
else:
    MA = None

pa_target = filtered["PA_TARGET"].dropna().unique().tolist()
ma_target = filtered["MA_TARGET"].dropna().unique().tolist()
pa_target = pa_target[0] if pa_target else 0.9
ma_target = ma_target[0] if ma_target else 0.85

if pa_target > 1:
    pa_target = pa_target / 100.0
if ma_target > 1:
    ma_target = ma_target / 100.0

# -------------------------
# KPI cards
# -------------------------
kpi1, kpi2, kpi3, kpi4 = st.columns([2,2,2,2])

with kpi1:
    if PA is not None:
        st.metric("Physical Availability (PA)", f"{PA:.1%}", delta=f"Target {pa_target:.0%}")
    else:
        st.metric("Physical Availability (PA)", "N/A", delta=f"Target {pa_target:.0%}")

with kpi2:
    if MA is not None:
        st.metric("Maintenance Availability (MA)", f"{MA:.1%}", delta=f"Target {ma_target:.0%}")
    else:
        st.metric("Maintenance Availability (MA)", "N/A", delta=f"Target {ma_target:.0%}")

with kpi3:
    st.metric("Total Delay Hours (filtered)", f"{total_delay:.2f} hrs")

with kpi4:
    st.metric("Available Time (sum)", f"{available_time:.2f} hrs" if available_time else "N/A")

st.markdown("---")

# -------------------------
# Trend Analysis
# -------------------------
st.subheader("Trend: Total Delay Hours vs PA%")

agg["PA_pct"] = None
for idx, row in agg.iterrows():
    avail = row.get("available_time_month", None)
    if avail and avail > 0:
        agg.at[idx, "PA_pct"] = max(0, 1 - row["total_delay_hours"] / avail)

fig_trend = go.Figure()
fig_trend.add_trace(go.Bar(x=agg[group_field], y=agg["total_delay_hours"], name="Total Delay Hours"))
fig_trend.add_trace(go.Scatter(x=agg[group_field], y=agg["PA_pct"], name="PA%", yaxis="y2", mode="lines+markers"))

fig_trend.add_shape(
    type="line",
    x0=0, x1=1,
    xref="paper",
    y0=pa_target, y1=pa_target,
    yref="y2",
    line=dict(color="green", dash="dash")
)

fig_trend.add_annotation(
    x=0,
    xref="paper",
    y=pa_target,
    yref="y2",
    showarrow=False,
    text=f"PA Target {pa_target:.0%}",
    font=dict(color="green"),
    align="left",
    xanchor="left",
    yanchor="bottom"
)

fig_trend.update_layout(
    xaxis_title="Period",
    yaxis_title="Delay Hours",
    yaxis2=dict(title="PA%", overlaying="y", side="right", tickformat="%", range=[0,1]),
    legend=dict(orientation="h")
)

st.plotly_chart(fig_trend, use_container_width=True)

# -------------------------
# Pareto
# -------------------------
st.subheader("Top Delay Causes (Pareto)")

cause_agg = filtered.groupby("CAUSE").agg(hours=("DELAY","sum")).reset_index().sort_values("hours", ascending=False)
cause_agg["cum_hours"] = cause_agg["hours"].cumsum()
cause_agg["cum_pct"] = cause_agg["cum_hours"] / cause_agg["hours"].sum()

top_n = st.slider("Top N causes to show", min_value=5, max_value=50, value=15)
pareto_df = cause_agg.head(top_n)

fig_pareto = go.Figure()
fig_pareto.add_trace(go.Bar(x=pareto_df["CAUSE"], y=pareto_df["hours"], name="Hours"))
fig_pareto.add_trace(go.Line(x=pareto_df["CAUSE"], y=pareto_df["cum_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers"))
fig_pareto.update_layout(
    xaxis_tickangle=-45,
    yaxis_title="Hours",
    yaxis2=dict(title="Cumulative %", overlaying="y", side="right", tickformat="%", range=[0,1]),
    legend=dict(orientation="h")
)
st.plotly_chart(fig_pareto, use_container_width=True)

st.markdown("---")

# -------------------------
# Drill-down table
# -------------------------
st.subheader("Drill-down: Delay Records")
display_cols = ["WEEK","MONTH","YEAR","DELAY","CATEGORY","CAUSE","MTN_NOTE","NOTE"]
if not show_notes:
    display_cols = [c for c in display_cols if c not in ("MTN_NOTE","NOTE")]

table_df = filtered[display_cols].copy()

table_df["WEEK"] = pd.to_numeric(table_df["WEEK"], errors="coerce")

if table_df["YEAR"].nunique() > 1:
    table_df = table_df.sort_values(["YEAR", "WEEK"], ascending=[True, True])
else:
    table_df = table_df.sort_values("WEEK", ascending=True)

st.dataframe(table_df, use_container_width=True, height=400)
