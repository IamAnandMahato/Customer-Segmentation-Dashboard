import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

from model import load_and_train_model

# -----------------------------
# Helper Functions
# -----------------------------
def assign_persona(cluster):
    personas = {
        0: "💰 Budget Customer",
        1: "🙂 Standard Customer",
        2: "💎 Premium Customer",
        3: "🔥 High Spender",
        4: "🧠 Careful Saver"
    }
    return personas.get(cluster, "Unknown")

def get_cluster_summary(df):
    return df.groupby("Cluster").mean(numeric_only=True)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# -----------------------------
# Load Data & Model
# -----------------------------
df, kmeans, scaler = load_and_train_model()

# -----------------------------
# ➕ Add New Customer (OPTION 2)
# -----------------------------
st.subheader("➕ Add New Customer")

with st.form("new_customer_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender_input = st.selectbox("Gender", ["Male", "Female"])
        age_input = st.number_input("Age", 18, 70, 25)

    with col2:
        income_input = st.number_input("Annual Income (k$)", 10, 150, 50)
        score_input = st.number_input("Spending Score", 1, 100, 50)

    submitted = st.form_submit_button("Add Customer")

    if submitted:
        new_row = {
            "Gender": 1 if gender_input == "Male" else 0,
            "Age": age_input,
            "Annual Income (k$)": income_input,
            "Spending Score (1-100)": score_input
        }

        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.success("✅ New customer added!")

# -----------------------------
# Retrain Model After Update
# -----------------------------
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

X = df[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

df["Persona"] = df["Cluster"].apply(assign_persona)

# -----------------------------
# Title
# -----------------------------
st.title("🛍️ Customer Segmentation Dashboard")
st.markdown("### Advanced ML + Real-Time Data Input")

# -----------------------------
# Sidebar Prediction
# -----------------------------
st.sidebar.header("🎯 Predict Customer")

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
col3.metric("Avg Spending", round(df["Spending Score (1-100)"].mean(), 2))

# -----------------------------
# Interactive Plot
# -----------------------------
st.subheader("📊 Customer Segments")

fig = px.scatter(
    df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    color="Persona",
    size="Age",
    hover_data=["Age"],
    title="Customer Segmentation"
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 3D Plot
# -----------------------------
st.subheader("🌐 3D Visualization")

fig3d = px.scatter_3d(
    df,
    x="Age",
    y="Annual Income (k$)",
    z="Spending Score (1-100)",
    color="Persona"
)

st.plotly_chart(fig3d, use_container_width=True)

# -----------------------------
# Cluster Distribution
# -----------------------------
st.subheader("📈 Cluster Distribution")

cluster_counts = df["Cluster"].value_counts().reset_index()
cluster_counts.columns = ["Cluster", "Count"]

fig_bar = px.bar(cluster_counts, x="Cluster", y="Count")

st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Summary Table
# -----------------------------
st.subheader("📋 Cluster Summary")
st.dataframe(get_cluster_summary(df))

# -----------------------------
# Download Data
# -----------------------------
csv = df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Data", csv, "segmented_data.csv", "text/csv")
