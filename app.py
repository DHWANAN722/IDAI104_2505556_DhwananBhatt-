import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

st.set_page_config(page_title="Rocket Launch Dashboard", layout="wide")

st.title("🚀 Rocket Launch Path Visualization Dashboard")

# =========================
# LOAD DATA
# =========================
@st.cache_data
def load_data():
    df = pd.read_csv("rocket_data.csv")

    # Convert Launch Date if exists
    if "Launch Date" in df.columns:
        df["Launch Date"] = pd.to_datetime(df["Launch Date"], errors="coerce")

    # Basic cleaning
    df = df.drop_duplicates()
    df = df.dropna()

    return df

try:
    df = load_data()
except:
    st.error("Make sure rocket_data.csv is uploaded to the repository.")
    st.stop()

st.sidebar.header("Navigation")
section = st.sidebar.selectbox(
    "Choose Section",
    ["Dataset Analysis", "Rocket Simulation"]
)

# =========================
# DATASET ANALYSIS
# =========================
if section == "Dataset Analysis":

    st.header("📊 Mission Data Analysis")

    col1, col2 = st.columns(2)

    # Scatter Plot: Payload vs Fuel
    with col1:
        st.subheader("Payload Weight vs Fuel Consumption")
        fig1 = plt.figure()
        sns.scatterplot(
            data=df,
            x="Payload Weight",
            y="Fuel Consumption",
            hue="Mission Success"
        )
        plt.xlabel("Payload Weight")
        plt.ylabel("Fuel Consumption")
        st.pyplot(fig1)

    # Bar Chart: Cost Success vs Failure
    with col2:
        st.subheader("Mission Cost: Success vs Failure")
        cost_summary = df.groupby("Mission Success")["Mission Cost"].mean()

        fig2 = plt.figure()
        cost_summary.plot(kind="bar")
        plt.ylabel("Average Mission Cost")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    # Line Chart: Duration vs Distance
    with col3:
        st.subheader("Mission Duration vs Distance from Earth")
        fig3 = plt.figure()
        sns.lineplot(
            data=df,
            x="Distance from Earth",
            y="Mission Duration"
        )
        st.pyplot(fig3)

    # Box Plot: Crew Size vs Success
    with col4:
        st.subheader("Crew Size vs Mission Success")
        fig4 = plt.figure()
        sns.boxplot(
            data=df,
            x="Mission Success",
            y="Crew Size"
        )
        st.pyplot(fig4)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    numeric_df = df.select_dtypes(include=np.number)
    corr = numeric_df.corr()

    fig5 = plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    st.pyplot(fig5)

# =========================
# ROCKET SIMULATION
# =========================
if section == "Rocket Simulation":

    st.header("🧠 Rocket Launch Simulation (Newton's Second Law)")

    thrust = st.slider("Thrust (N)", 5_000_000, 12_000_000, 8_000_000)
    payload = st.slider("Payload Weight (kg)", 10_000, 100_000, 20_000)
    fuel = st.slider("Fuel Mass (kg)", 100_000, 500_000, 300_000)

    structure_mass = 100_000
    mass = structure_mass + payload + fuel

    g = 9.81
    drag_coeff = 0.00005
    velocity = 0
    altitude = 0
    dt = 1

    altitudes = []
    velocities = []

    for t in range(200):

        drag = drag_coeff * velocity**2
        acceleration = (thrust - (mass * g) - drag) / mass

        velocity += acceleration * dt
        altitude += velocity * dt

        fuel_burn = 1500
        if fuel > 0:
            fuel -= fuel_burn
            mass -= fuel_burn

        altitudes.append(altitude)
        velocities.append(velocity)

    # Altitude Plot
    fig_alt = go.Figure()
    fig_alt.add_trace(go.Scatter(
        x=list(range(200)),
        y=altitudes,
        mode="lines",
        name="Altitude"
    ))

    fig_alt.update_layout(
        title="Rocket Altitude Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Altitude (meters)"
    )

    st.plotly_chart(fig_alt)

    # Velocity Plot
    fig_vel = go.Figure()
    fig_vel.add_trace(go.Scatter(
        x=list(range(200)),
        y=velocities,
        mode="lines",
        name="Velocity"
    ))

    fig_vel.update_layout(
        title="Rocket Velocity Over Time",
        xaxis_title="Time (seconds)",
        yaxis_title="Velocity (m/s)"
    )

    st.plotly_chart(fig_vel)
