# streamlit_app/components.py
from typing import Any
import streamlit as st
import pandas as pd


def render_metric_card(title: str, value: Any, delta: Any = None, icon: str = None):
    """Render a metric card with icon"""
    col1, col2 = st.columns([1, 3])
    with col1:
        if icon:
            st.markdown(f"<h1 style='text-align: center;'>{icon}</h1>", unsafe_allow_html=True)
    with col2:
        st.metric(title, value, delta)


def render_chart_selector():
    """Render chart type selector"""
    chart_types = {
        "bar": "Bar Chart",
        "line": "Line Chart",
        "pie": "Pie Chart",
        "scatter": "Scatter Plot",
        "histogram": "Histogram",
        "box": "Box Plot"
    }
    return st.selectbox("Chart Type", options=list(chart_types.values()))


def render_data_quality_report(df: pd.DataFrame):
    """Render data quality report"""
    st.markdown("### 📊 Data Quality Report")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        missing = df.isnull().sum().sum()
        st.metric("Missing Values", missing)

    # Column-wise quality
    quality_data = []
    for col in df.columns:
        quality_data.append({
            "Column": col,
            "Type": str(df[col].dtype),
            "Nulls": df[col].isnull().sum(),
            "Null %": f"{(df[col].isnull().sum() / len(df)) * 100:.1f}%",
            "Unique": df[col].nunique()
        })

    st.dataframe(pd.DataFrame(quality_data))


def render_sql_editor(default_sql: str = ""):
    """Render SQL editor with syntax highlighting"""
    return st.text_area(
        "SQL Query",
        value=default_sql,
        height=200,
        key="sql_editor"
    )


def render_database_connections():
    """Render database connection status"""
    st.markdown("### 🔗 Database Connections")

    connections = {
        "MySQL": {"status": "Connected", "color": "green"},
        "MongoDB": {"status": "Connected", "color": "green"},
        "SQLite": {"status": "Connected", "color": "green"}
    }

    for db_name, info in connections.items():
        st.markdown(f"""
        <div style='display: flex; align-items: center; margin: 5px 0;'>
            <span style='display: inline-block; width: 10px; height: 10px; 
                         border-radius: 50%; background-color: {info["color"]}; 
                         margin-right: 10px;'></span>
            <span style='font-weight: 500;'>{db_name}</span>
            <span style='margin-left: auto; color: #666;'>{info["status"]}</span>
        </div>
        """, unsafe_allow_html=True)