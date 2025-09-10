import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
from io import BytesIO
import datetime

st.set_page_config(page_title="Physical Availability Dashboard", layout="wide")

# -----------------------------
# Load Data
# -----------------------------
# Example: df = pd.read_excel("data.xlsx", sheet_name="Scorecard")
# Replace with your data loading logic
df = pd.read_excel("data.xlsx", sheet_name="Scorecard")

# Ensure correct types
df["DELAY"] = pd.to_numeric(df["DELAY"], errors='coerce')
df["AVAILABLE_HOURS"] = pd.to_numeric(df.get("AVAILABLE_HOURS", 0), errors='coerce')
df["PERIOD_MONTH"] = df["PERIOD_MONTH"].astype(str)
df["PERIOD_YEAR"] = pd.to_numeric(df["PERIOD_YEAR"], errors='coerce')

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.title("Filters")
selected_month = st.sidebar.selectbox("Select Month", ["All"] + sorted(df["PERIOD_MONTH"].unique()))
selected_years = st.sidebar.multiselect(
    "Select Years",
    sorted(df["PERIOD_YEAR"].dropna().unique()),
    default=list(df["PERIOD_YEAR"].dropna().unique())
)

# Filter data based on sidebar
df_filtered = df.copy()
if selected_month != "All":
    df_filtered = df_filtered[df_filtered["PERIOD_MONTH"] == selected_month]
if selected_years:
    df_filtered = df_filtered[df_filtered["PERIOD_YEAR"].isin(selected_years)]

# -----------------------------
# Tabs: Main Dashboard & Reliability
# -----------------------------
tabs = st.tabs(["Main Dashboard", "Reliability"])

with tabs[0]:
    st.header("Physical Availability Dashboard")

    # -----------------------------
    # KPI Metrics
    # -----------------------------
    total_delay = df_filtered["DELAY"].sum() if "DELAY" in df_filtered.columns else 0
    available_time = df_filtered["AVAILABLE_HOURS"].sum() if "AVAILABLE_HOURS" in df_filtered.columns else 0
    PA = max(0, 1 - total_delay / available_time) if available_time > 0 else None

    maintenance_delay = df_filtered[df_filtered.get("CATEGORY") == "Maintenance"]["DELAY"].sum() if "CATEGORY" in df_filtered.columns else 0
    MA = max(0, 1 - maintenance_delay / available_time) if available_time > 0 else None

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Delay (hrs)", f"{total_delay:.2f}")
    col2.metric("Available Time (hrs)", f"{available_time:.2f}")
    col3.metric("PA (%)", f"{PA:.2%}" if PA is not None else "N/A")
    col4.metric("MA (%)", f"{MA:.2%}" if MA is not None else "N/A")

    # -----------------------------
    # Trend Chart: Delay Over Time
    # -----------------------------
    if "PERIOD_MONTH" in df_filtered.columns:
        df_trend = df_filtered.groupby("PERIOD_MONTH")["DELAY"].sum().reset_index()
        fig_trend = px.line(df_trend, x="PERIOD_MONTH", y="DELAY", title="Delay Trend Over Months")
        st.plotly_chart(fig_trend, use_container_width=True)

    # -----------------------------
    # Pareto Chart: Delay by Category
    # -----------------------------
    if "CATEGORY" in df_filtered.columns and "DELAY" in df_filtered.columns:
        df_pareto = df_filtered.groupby("CATEGORY")["DELAY"].sum().reset_index()
        df_pareto = df_pareto.sort_values("DELAY", ascending=False)
        df_pareto["cum_percent"] = df_pareto["DELAY"].cumsum() / df_pareto["DELAY"].sum() * 100
        fig_pareto = make_subplots(specs=[[{"secondary_y": True}]])
        fig_pareto.add_trace(go.Bar(x=df_pareto["CATEGORY"], y=df_pareto["DELAY"], name="Delay (hrs)"))
        fig_pareto.add_trace(go.Scatter(x=df_pareto["CATEGORY"], y=df_pareto["cum_percent"], name="Cumulative %", marker=dict(color="red")), secondary_y=True)
        fig_pareto.update_yaxes(title_text="Delay (hrs)", secondary_y=False)
        fig_pareto.update_yaxes(title_text="Cumulative %", secondary_y=True)
        fig_pareto.update_layout(title="Pareto Chart by Category")
        st.plotly_chart(fig_pareto, use_container_width=True)

    # -----------------------------
    # Donut Charts
    # -----------------------------
    if "CATEGORY" in df_filtered.columns:
        df_cat = df_filtered.groupby("CATEGORY")["DELAY"].sum().reset_index()
        fig_donut_cat = px.pie(df_cat, names="CATEGORY", values="DELAY", hole=0.4, title="Delay by Category")
        st.plotly_chart(fig_donut_cat, use_container_width=True)

    if "SCHEDULED" in df_filtered.columns:
        df_sched = df_filtered.groupby("SCHEDULED")["DELAY"].sum().reset_index()
        fig_donut_sched = px.pie(df_sched, names="SCHEDULED", values="DELAY", hole=0.4, title="Scheduled vs Unscheduled")
        st.plotly_chart(fig_donut_sched, use_container_width=True)

    # -----------------------------
    # Drilldown Table
    # -----------------------------
    st.subheader("Equipment Drilldown Table")
    if not df_filtered.empty:
        gb = GridOptionsBuilder.from_dataframe(df_filtered)
        gb.configure_pagination(paginationAutoPageSize=True)
        gb.configure_side_bar()
        gb.configure_selection(selection_mode="single")
        grid_options = gb.build()
        AgGrid(df_filtered, gridOptions=grid_options, update_mode=GridUpdateMode.SELECTION_CHANGED, theme="alpine")

with tabs[1]:
    st.header("Reliability Dashboard")

    # -----------------------------
    # MTTR and MTBF Metrics
    # -----------------------------
    if "MTTR" in df_filtered.columns:
        avg_mttr = df_filtered["MTTR"].mean()
        st.metric("Average MTTR (hrs)", f"{avg_mttr:.2f}")

    if "MTBF" in df_filtered.columns:
        avg_mtbf = df_filtered["MTBF"].mean()
        st.metric("Average MTBF (hrs)", f"{avg_mtbf:.2f}")

    # -----------------------------
    # MTTR and MTBF Trend Charts
    # -----------------------------
    if "PERIOD_MONTH" in df_filtered.columns and "MTTR" in df_filtered.columns:
        df_mttr_trend = df_filtered.groupby("PERIOD_MONTH")["MTTR"].mean().reset_index()
        fig_mttr = px.line(df_mttr_trend, x="PERIOD_MONTH", y="MTTR", title="MTTR Trend")
        st.plotly_chart(fig_mttr, use_container_width=True)

    if "PERIOD_MONTH" in df_filtered.columns and "MTBF" in df_filtered.columns:
        df_mtbf_trend = df_filtered.groupby("PERIOD_MONTH")["MTBF"].mean().reset_index()
        fig_mtbf = px.line(df_mtbf_trend, x="PERIOD_MONTH", y="MTBF", title="MTBF Trend")
        st.plotly_chart(fig_mtbf, use_container_width=True)

    # -----------------------------
    # Reliability Pareto Chart
    # -----------------------------
    if "CATEGORY" in df_filtered.columns and "MTTR" in df_filtered.columns:
        df_reliability_pareto = df_filtered.groupby("CATEGORY")["MTTR"].sum().reset_index()
        df_reliability_pareto = df_reliability_pareto.sort_values("MTTR", ascending=False)
        df_reliability_pareto["cum_percent"] = df_reliability_pareto["MTTR"].cumsum() / df_reliability_pareto["MTTR"].sum() * 100
        fig_reliability = make_subplots(specs=[[{"secondary_y": True}]])
        fig_reliability.add_trace(go.Bar(x=df_reliability_pareto["CATEGORY"], y=df_reliability_pareto["MTTR"], name="MTTR (hrs)"))
        fig_reliability.add_trace(go.Scatter(x=df_reliability_pareto["CATEGORY"], y=df_reliability_pareto["cum_percent"], name="Cumulative %", marker=dict(color="red")), secondary_y=True)
        fig_reliability.update_yaxes(title_text="MTTR (hrs)", secondary_y=False)
        fig_reliability.update_yaxes(title_text="Cumulative %", secondary_y=True)
        fig_reliability.update_layout(title="Reliability Pareto by Category")
        st.plotly_chart(fig_reliability, use_container_width=True)

    # -----------------------------
    # Reliability Drilldown Table
    # -----------------------------
    st.subheader("Reliability Drilldown Table")
    reliability_cols = ["EQUIPMENT", "CATEGORY", "MTTR", "MTBF", "FAILURE_COUNT"]
    df_reliability_table = df_filtered[reliability_cols].copy() if all(col in df_filtered.columns for col in reliability_cols) else pd.DataFrame()
    if not df_reliability_table.empty:
        gb_rel = GridOptionsBuilder.from_dataframe(df_reliability_table)
        gb_rel.configure_pagination(paginationAutoPageSize=True)
        gb_rel.configure_side_bar()
        gb_rel.configure_selection(selection_mode="single")
        grid_options_rel = gb_rel.build()
        AgGrid(df_reliability_table, gridOptions=grid_options_rel, update_mode=GridUpdateMode.SELECTION_CHANGED, theme="alpine")

# -----------------------------
# Export to Excel
# -----------------------------
st.sidebar.subheader("Export Data")
export_button = st.sidebar.button("Export Filtered Data to Excel")
if export_button:
    to_export = BytesIO()
    df_filtered.to_excel(to_export, index=False)
    st.sidebar.download_button(
        label="Download Excel",
        data=to_export.getvalue(),
        file_name=f"Filtered_Data_{datetime.date.today()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# -----------------------------
# Footer / Notes
# -----------------------------
st.markdown(
    """
    ---
    *Dashboard generated using Streamlit and Plotly.*
    """
)
