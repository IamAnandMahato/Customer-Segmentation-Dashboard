import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from model import load_and_train_model
from utils import assign_persona, get_cluster_summary

st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Load Model
df, kmeans, scaler, le = load_and_train_model()

# Assign Persona
df["Persona"] = df["Cluster"].apply(assign_persona)

# -----------------------------
# UI Header
# -----------------------------
st.title("🛍️ Customer Segmentation Dashboard (Advanced)")
st.markdown("Interactive ML Dashboard for Business Insights")

# -----------------------------
# Sidebar Input (Real-Time)
# -----------------------------
st.sidebar.header("🎯 Predict New Customer")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 18, 70, 30)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 60)
score = st.sidebar.slider("Spending Score", 1, 100, 50)

gender_val = 1 if gender == "Male" else 0

input_data = np.array([[gender_val, age, income, score]])
input_scaled = scaler.transform(input_data)

cluster = kmeans.predict(input_scaled)[0]
persona = assign_persona(cluster)

st.sidebar.success(f"Cluster: {cluster}")
st.sidebar.success(f"Persona: {persona}")

# -----------------------------
# KPI Section
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Avg Income", round(df["Annual Income (k$)"].mean(), 2))
col3.metric("Avg Spending Score", round(df["Spending Score (1-100)"].mean(), 2))

# -----------------------------
# Plotly Scatter (Interactive)
# -----------------------------
st.subheader("📊 Customer Segmentation (Interactive)")

fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Persona",
    size="Age",
    hover_data=["Age", "Gender"],
    title="Customer Segments"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3D Visualization (Advanced)
# -----------------------------
st.subheader("🌐 3D Customer View")

fig3d = px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Persona",
    title="3D Segmentation"
)

st.plotly_chart(fig3d, use_container_width=True)

# -----------------------------
# Cluster Distribution
# -----------------------------
st.subheader("📈 Cluster Distribution")

fig_bar = px.bar(
    df["Cluster"].value_counts().reset_index(),
    x="index",
    y="Cluster",
    labels={"index": "Cluster", "Cluster": "Count"},
    title="Customer Distribution per Cluster"
)

st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Cluster Summary Table
# -----------------------------
st.subheader("📋 Cluster Summary")

summary = get_cluster_summary(df)
st.dataframe(summary)

# -----------------------------
# Raw Data
# -----------------------------
with st.expander("🔍 View Raw Data"):
    st.dataframe(df)

# -----------------------------
# Download Option
# -----------------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Segmented Data", csv, "segmented_data.csv", "text/csv")
