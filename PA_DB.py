import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Physical Availability - Data Delay Time", layout="wide")

# Logo URL
LOGO_URL = "https://raw.githubusercontent.com/AlvinWinarta2111/dashboard-pa/refs/heads/main/images/alamtri_logo.jpeg"

# Layout for title and logo
logo_col, title_col = st.columns([1, 6])

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
@st.cache_data
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

    replacements = {
        "WEEK": "WEEK", "MONTH": "MONTH", "YEAR": "YEAR", "DELAY": "DELAY",
        "MTN DELAY TYPE": "MTN_DELAY_TYPE", "SCH MTN": "SCH_MTN", "UNSCH MTN": "UNSCH_MTN",
        "MINING DELAY": "MINING_DELAY", "WEATHER DELAY": "WEATHER_DELAY", "OTHER DELAY": "OTHER_DELAY",
        "AVAILABLE TIME (MONTH)": "AVAILABLE_TIME_MONTH", "MA TARGET": "MA_TARGET", "PA TARGET": "PA_TARGET",
        "MTN NOTE": "MTN_NOTE", "NOTE": "NOTE", "START": "START", "STOP": "STOP"
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

    # Convert START/STOP to datetime first
    if "START" in df.columns and "STOP" in df.columns:
        df["START"] = pd.to_datetime(df["START"], errors="coerce")
        df["STOP"] = pd.to_datetime(df["STOP"], errors="coerce")
        df["AVAILABLE_HOURS"] = (df["STOP"] - df["START"]).dt.total_seconds() / 3600
    else:
        df["AVAILABLE_HOURS"] = None

    # Calculate ISO Week Period from the START date
    if "START" in df.columns and pd.api.types.is_datetime64_any_dtype(df["START"]):
        iso_dates = df['START'].dt.isocalendar()
        df['ISO_WEEK_PERIOD'] = iso_dates['year'].astype(str) + '-W' + iso_dates['week'].astype(str).str.zfill(2)
    else:
        df['ISO_WEEK_PERIOD'] = None

    # numeric conversions
    df["DELAY"] = pd.to_numeric(df["DELAY"], errors="coerce")
    if "AVAILABLE_TIME_MONTH" in df.columns:
        df["AVAILABLE_TIME_MONTH"] = pd.to_numeric(df["AVAILABLE_TIME_MONTH"], errors="coerce")
    else:
        df["AVAILABLE_TIME_MONTH"] = None

    # fill category-like columns
    for cat in ["MTN_DELAY_TYPE","SCH_MTN","UNSCH_MTN","MINING_DELAY","WEATHER_DELAY","OTHER_DELAY","MTN_NOTE","NOTE"]:
        if cat in df.columns:
            df[cat] = df[cat].fillna("").astype(str)
        else:
            df[cat] = ""

    # category and cause
    def determine_category(row):
        if row["MTN_DELAY_TYPE"] or row["SCH_MTN"] or row["UNSCH_MTN"]: return "Maintenance"
        if row["MINING_DELAY"]: return "Mining"
        if row["WEATHER_DELAY"]: return "Weather"
        if row["OTHER_DELAY"]: return "Other"
        return "Unknown"
    df["CATEGORY"] = df.apply(determine_category, axis=1)

    def compose_cause(row):
        parts = []
        for cat in ["MTN_DELAY_TYPE","SCH_MTN","UNSCH_MTN","MINING_DELAY","WEATHER_DELAY","OTHER_DELAY","MTN_NOTE","NOTE"]:
            if row.get(cat) and str(row.get(cat)).strip().lower() != "nan":
                parts.append(str(row.get(cat)).strip())
        return " | ".join(parts)
    df["CAUSE"] = df.apply(compose_cause, axis=1)

    # drop rows without numeric delay
    df = df[df["DELAY"].notna()].copy()

    # YEAR / WEEK normalization
    try:
        df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
        df = df[(df["YEAR"].notna()) & (df["YEAR"] >= 2000) & (df["YEAR"] <= 2100)]
    except: pass
    try:
        df["WEEK"] = pd.to_numeric(df["WEEK"], errors="coerce").astype("Int64")
    except: pass

    df["PERIOD_MONTH"] = df["MONTH"].astype(str) + " " + df["YEAR"].astype(str)

    # Remove rows where all column values are identical to another row.
    df = df.drop_duplicates()
    
    return df

# -------------------------
# Load data
# -------------------------
df = load_data_from_url()
if df is None: st.stop()

# -------------------------
# Sidebar Filters
# -------------------------
st.sidebar.header("Filters & Options")
granularity = st.sidebar.selectbox(
    "Time granularity",
    options=["ISO_WEEK_PERIOD", "PERIOD_MONTH"],
    format_func=lambda x: "Week (ISO)" if x == "ISO_WEEK_PERIOD" else "Month",
    index=1
)
months_available = df["PERIOD_MONTH"].dropna().unique().tolist()
months_available.sort(key=lambda x: (int(x.split()[-1]), pd.to_datetime(x.split()[0], format='%b').month))
selected_month = st.sidebar.selectbox("Month", options=["All"]+months_available, index=0)
st.sidebar.markdown("**Delay Categories**")
all_categories = sorted(df["CATEGORY"].unique().tolist())
selected_categories = []
for category in all_categories:
    if st.sidebar.checkbox(category, value=True, key=f"cb_{category}"):
        selected_categories.append(category)
show_notes = st.sidebar.checkbox("Show notes column in drill-down table", value=True)

# Apply filters
filtered = df.copy()
if selected_month != "All":
    filtered = filtered[filtered["PERIOD_MONTH"] == selected_month]
if selected_categories:
    filtered = filtered[filtered["CATEGORY"].isin(selected_categories)]
else:
    st.warning("No delay categories selected.")
    filtered = filtered.iloc[0:0]

# -------------------------
# Aggregations
# -------------------------
if not filtered.empty:
    group_field = granularity
    agg = filtered.groupby(group_field).agg(
        total_delay_hours=("DELAY","sum"),
        available_time_month=("AVAILABLE_TIME_MONTH","max"),
        available_hours=("AVAILABLE_HOURS","sum")
    ).reset_index()

    if group_field == "ISO_WEEK_PERIOD":
        agg = agg.sort_values("ISO_WEEK_PERIOD")
    elif group_field == "PERIOD_MONTH":
        agg[group_field] = pd.Categorical(agg[group_field], categories=months_available, ordered=True)
        agg = agg.sort_values(group_field)
else:
    agg = pd.DataFrame()

# -------------------------
# KPI cards + Donut chart
# -------------------------
total_delay = filtered["DELAY"].sum()
if "AVAILABLE_TIME_MONTH" in filtered.columns and filtered["AVAILABLE_TIME_MONTH"].notna().any():
    available_time = filtered.drop_duplicates("PERIOD_MONTH")["AVAILABLE_TIME_MONTH"].dropna().sum()
elif "AVAILABLE_HOURS" in filtered.columns and filtered["AVAILABLE_HOURS"].notna().any():
    per_period_avail = filtered.groupby("PERIOD_MONTH")["AVAILABLE_HOURS"].max().dropna()
    available_time = per_period_avail.sum()
else: available_time = None
PA = max(0, 1 - total_delay / available_time) if available_time and available_time > 0 else None
maintenance_delay = filtered[filtered["CATEGORY"]=="Maintenance"]["DELAY"].sum()
MA = max(0, 1 - maintenance_delay / available_time) if available_time and available_time > 0 else None
pa_target_series = df["PA_TARGET"].dropna().unique()
ma_target_series = df["MA_TARGET"].dropna().unique()
pa_target = float(pa_target_series[0]) if len(pa_target_series) > 0 else 0.90
ma_target = float(ma_target_series[0]) if len(ma_target_series) > 0 else 0.85
if pa_target > 1: pa_target /= 100.0
if ma_target > 1: ma_target /= 100.0

donut_data = filtered.groupby("CATEGORY")["DELAY"].sum().reset_index().sort_values("DELAY",ascending=False)
donut_fig = go.Figure(data=[go.Pie(labels=donut_data["CATEGORY"], values=donut_data["DELAY"], hole=0.4, textinfo="label+percent", hovertemplate="%{label}: %{value:.2f} hrs<extra></extra>")])
donut_fig.update_layout(margin=dict(t=20,b=20,l=20,r=20), legend_orientation="h")

left_col, right_col = st.columns([2,3])
with left_col:
    st.subheader("Key KPIs")
    st.metric("Physical Availability (PA)", f"{PA:.1%}" if PA is not None else "N/A", delta=f"Target {pa_target:.0%}", delta_color="off")
    st.metric("Maintenance Availability (MA)", f"{MA:.1%}" if MA is not None else "N/A", delta=f"Target {ma_target:.0%}", delta_color="off")
    st.metric("Total Delay Hours (filtered)", f"{total_delay:.2f} hrs")
    st.metric("Available Time (sum)", f"{available_time:.2f} hrs" if available_time is not None else "N/A")
with right_col:
    st.subheader("Delay Breakdown (Overall)")
    if not donut_data.empty:
        st.plotly_chart(donut_fig, use_container_width=True)
    else:
        st.info("No delay data available for the selected filters.")

st.markdown("---")

# -------------------------
# Category Drill-Down Donut
# -------------------------
st.subheader("Category Drill-Down")
drill_down_categories = filtered['CATEGORY'].unique().tolist()
if not drill_down_categories:
    st.info("No data available to drill down.")
else:
    selected_category_drilldown = st.selectbox("Select a category to see its breakdown:", options=drill_down_categories)
    drilldown_df = filtered[filtered['CATEGORY'] == selected_category_drilldown].copy()
    total_drilldown_hours = drilldown_df['DELAY'].sum()

    labels, values = [], []
    if selected_category_drilldown == "Maintenance":
        drilldown_df['MAINTENANCE_TYPE'] = np.where(drilldown_df['SCH_MTN'] != '', 'Scheduled', 'Unscheduled')
        breakdown_data = drilldown_df.groupby('MAINTENANCE_TYPE')['DELAY'].sum().reset_index()
        labels = breakdown_data['MAINTENANCE_TYPE']
        values = breakdown_data['DELAY']
    else:
        breakdown_data = drilldown_df.groupby('CAUSE')['DELAY'].sum().reset_index()
        labels = breakdown_data['CAUSE']
        values = breakdown_data['DELAY']

    left_drill_col, right_drill_col = st.columns([1, 2])
    # This is inside the "Category Drill-Down" section
with left_drill_col:
    st.metric(
        label=f"Total Hours for {selected_category_drilldown}",
        value=f"{total_drilldown_hours:.2f} hrs"
    )
    
    # Add breakdown if the category is Maintenance
    if selected_category_drilldown == "Maintenance":
        # Create a dictionary for easy lookup, e.g., {'Scheduled': 10.5, 'Unscheduled': 5.2}
        hours_by_type = pd.Series(values.values, index=labels.values).to_dict()
        scheduled_hours = hours_by_type.get('Scheduled', 0)
        unscheduled_hours = hours_by_type.get('Unscheduled', 0)
        
        st.markdown("---") # Visual separator
        st.markdown(f"**Scheduled:** `{scheduled_hours:.2f} hrs`")
        st.markdown(f"**Unscheduled:** `{unscheduled_hours:.2f} hrs`")
    # --- END OF ADDED CODE ---
    with right_drill_col:
        if not breakdown_data.empty:
            fig_drilldown = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4, textinfo="label+percent", hovertemplate="%{label}: %{value:.2f} hrs (%{percent})<extra></extra>")])
            fig_drilldown.update_layout(margin=dict(t=20, b=20, l=20, r=20), legend_orientation="h")
            st.plotly_chart(fig_drilldown, use_container_width=True)
        else:
            st.info(f"No breakdown data available for {selected_category_drilldown}.")

st.markdown("---")

# -------------------------
# Trend Analysis
# -------------------------
st.subheader("Trend: Total Delay Hours vs PA%")
if not agg.empty:
    agg["PA_pct"] = None
    agg["available_for_pa"] = None
    for idx,row in agg.iterrows():
        avail_month = row.get("available_time_month", None)
        avail_hours = row.get("available_hours", None)
        avail = avail_month if pd.notna(avail_month) and avail_month > 0 else (avail_hours if pd.notna(avail_hours) and avail_hours > 0 else None)
        agg.at[idx,"available_for_pa"] = avail
        if avail and avail > 0:
            agg.at[idx,"PA_pct"] = max(0, 1 - row["total_delay_hours"] / avail)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Bar(x=agg[group_field], y=agg["total_delay_hours"], name="Total Delay Hours"))
    fig_trend.add_trace(go.Scatter(x=agg[group_field], y=agg["PA_pct"], name="PA%", yaxis="y2", mode="lines+markers"))
    fig_trend.add_shape(type="line", x0=0, x1=1, xref="paper", y0=pa_target, y1=pa_target, yref="y2", line=dict(color="green", dash="dash"))
    fig_trend.add_annotation(x=0.01, xref="paper", y=pa_target, yref="y2", showarrow=False, text=f"PA Target {pa_target:.0%}", font=dict(color="green"), align="left", xanchor="left", yanchor="bottom")
    fig_trend.update_layout(xaxis_title="Period", yaxis_title="Delay Hours", yaxis2=dict(title="PA%", overlaying="y", side="right", tickformat=".0%", range=[0,1.05]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_trend, use_container_width=True)
else:
    st.info("No data to display trend chart for the current selection.")

# -------------------------
# Pareto Chart
# -------------------------
st.subheader("Top Delay Causes (Pareto)")
if not filtered.empty:
    cause_agg = filtered.groupby("CAUSE").agg(hours=("DELAY","sum")).reset_index().sort_values("hours",ascending=False)
    if not cause_agg.empty and cause_agg['hours'].sum() > 0:
        cause_agg["cum_hours"] = cause_agg["hours"].cumsum()
        cause_agg["cum_pct"] = cause_agg["cum_hours"] / cause_agg["hours"].sum()
        max_causes = len(cause_agg)
        if max_causes > 5:
            top_n = st.slider("Top N causes to show", min_value=5, max_value=min(max_causes, 50), value=min(max_causes,15))
            pareto_df = cause_agg.head(top_n)
            fig_pareto = go.Figure()
            fig_pareto.add_trace(go.Bar(x=pareto_df["CAUSE"], y=pareto_df["hours"], name="Hours"))
            fig_pareto.add_trace(go.Scatter(x=pareto_df["CAUSE"], y=pareto_df["cum_pct"], name="Cumulative %", yaxis="y2", mode="lines+markers"))
            fig_pareto.update_layout(xaxis_tickangle=-45, yaxis_title="Hours", yaxis2=dict(title="Cumulative %", overlaying="y", side="right", tickformat=".0%", range=[0,1.05]), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_pareto, use_container_width=True)
        else:
            st.info("Not enough different causes to generate a meaningful Pareto chart.")
    else:
        st.info("No delay causes to display for the current selection.")
else:
    st.info("No data to display Pareto chart for the current selection.")

st.markdown("---")

# -------------------------
# Drill-down table
# -------------------------
st.subheader("Drill-down: Delay Records")
display_cols=["WEEK","MONTH","YEAR","DELAY","CATEGORY","CAUSE"]
if show_notes:
    if "MTN_NOTE" in filtered.columns: display_cols.append("MTN_NOTE")
    if "NOTE" in filtered.columns: display_cols.append("NOTE")

final_display_cols = [col for col in display_cols if col in filtered.columns]
if not filtered.empty:
    table_df = filtered[final_display_cols].copy()
    table_df["DELAY"] = table_df["DELAY"].round(2)
    sort_by = ["YEAR", "WEEK"] if "YEAR" in table_df.columns and table_df["YEAR"].nunique() > 1 else ["WEEK"]
    sort_by = [col for col in sort_by if col in table_df.columns]
    if sort_by:
        table_df = table_df.sort_values(by=sort_by, ascending=True)
    st.dataframe(table_df, use_container_width=True, height=400)
else:
    st.info("No drill-down records to show for the current selection.")
