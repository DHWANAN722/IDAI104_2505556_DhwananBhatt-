import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.integrate import solve_ivp

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🚀 Space Mission Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main {background-color: #0d0d1a;}
    h1, h2, h3 {color: #e0e0ff;}
    .metric-card {
        background: linear-gradient(135deg, #1a1a3e 0%, #0d0d2b 100%);
        border: 1px solid #3a3a6e;
        border-radius: 12px;
        padding: 18px;
        text-align: center;
    }
    .stSelectbox label, .stMultiSelect label, .stSlider label {color: #a0a0d0 !important;}
</style>
""", unsafe_allow_html=True)

# ── Load & Clean Data ─────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("space_missions_dataset.csv")

    # Rename columns for readability
    df.columns = [c.strip() for c in df.columns]

    # Date conversion
    df["Launch Date"] = pd.to_datetime(df["Launch Date"], errors="coerce")
    df["Launch Year"] = df["Launch Date"].dt.year

    # Numeric coercion for key columns
    numeric_cols = [
        "Distance from Earth (light-years)",
        "Mission Duration (years)",
        "Mission Cost (billion USD)",
        "Scientific Yield (points)",
        "Crew Size",
        "Mission Success (%)",
        "Fuel Consumption (tons)",
        "Payload Weight (tons)",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing critical values
    df.dropna(subset=numeric_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


df = load_data()

# ── Sidebar Filters ───────────────────────────────────────────────────────────
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg",
    width=120,
)
st.sidebar.title("🔭 Mission Filters")

mission_types = st.sidebar.multiselect(
    "Mission Type",
    options=sorted(df["Mission Type"].unique()),
    default=sorted(df["Mission Type"].unique()),
)

launch_vehicles = st.sidebar.multiselect(
    "Launch Vehicle",
    options=sorted(df["Launch Vehicle"].unique()),
    default=sorted(df["Launch Vehicle"].unique()),
)

target_types = st.sidebar.multiselect(
    "Target Type",
    options=sorted(df["Target Type"].unique()),
    default=sorted(df["Target Type"].unique()),
)

min_success, max_success = st.sidebar.slider(
    "Mission Success (%)",
    min_value=0,
    max_value=100,
    value=(0, 100),
)

# Apply filters
filtered = df[
    df["Mission Type"].isin(mission_types)
    & df["Launch Vehicle"].isin(launch_vehicles)
    & df["Target Type"].isin(target_types)
    & df["Mission Success (%)"].between(min_success, max_success)
]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚀 Space Mission Analytics Dashboard")
st.markdown(
    "Exploring 500 simulated space missions — fuel, payload, cost, and success insights."
)
st.markdown("---")

# ── KPI Row ───────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Missions", len(filtered))
k2.metric("Avg Success Rate", f"{filtered['Mission Success (%)'].mean():.1f}%")
k3.metric("Avg Mission Cost", f"${filtered['Mission Cost (billion USD)'].mean():.1f}B")
k4.metric("Avg Fuel (tons)", f"{filtered['Fuel Consumption (tons)'].mean():,.0f}")
k5.metric("Avg Payload (tons)", f"{filtered['Payload Weight (tons)'].mean():.1f}")

st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📊 Mission Overview",
        "⚗️ Fuel & Payload",
        "💰 Cost & Success",
        "🌌 Rocket Trajectory Sim",
        "📋 Raw Data",
    ]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 – Mission Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            filtered,
            x="Mission Type",
            color="Mission Type",
            title="Mission Count by Type",
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        vehicle_success = (
            filtered.groupby("Launch Vehicle")["Mission Success (%)"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            vehicle_success,
            x="Launch Vehicle",
            y="Mission Success (%)",
            color="Mission Success (%)",
            color_continuous_scale="Viridis",
            title="Avg Success Rate by Launch Vehicle",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.pie(
            filtered,
            names="Target Type",
            title="Missions by Target Type",
            template="plotly_dark",
            hole=0.4,
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        yearly = (
            filtered.groupby("Launch Year")
            .agg(missions=("Mission ID", "count"), success=("Mission Success (%)", "mean"))
            .reset_index()
        )
        fig = px.line(
            yearly,
            x="Launch Year",
            y="success",
            markers=True,
            title="Avg Success Rate Over Time",
            template="plotly_dark",
            labels={"success": "Avg Success (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 – Fuel & Payload
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            filtered,
            x="Payload Weight (tons)",
            y="Fuel Consumption (tons)",
            color="Mission Type",
            size="Mission Cost (billion USD)",
            hover_name="Mission Name",
            title="Payload Weight vs Fuel Consumption",
            template="plotly_dark",
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.box(
            filtered,
            x="Launch Vehicle",
            y="Fuel Consumption (tons)",
            color="Launch Vehicle",
            title="Fuel Consumption Distribution by Launch Vehicle",
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        fig = px.scatter(
            filtered,
            x="Payload Weight (tons)",
            y="Mission Success (%)",
            color="Mission Type",
            hover_name="Mission Name",
            title="Payload Weight vs Mission Success",
            template="plotly_dark",
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        fig = px.histogram(
            filtered,
            x="Fuel Consumption (tons)",
            nbins=30,
            color="Mission Type",
            title="Fuel Consumption Distribution",
            template="plotly_dark",
            barmode="overlay",
            opacity=0.7,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 – Cost & Success
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        fig = px.scatter(
            filtered,
            x="Mission Cost (billion USD)",
            y="Mission Success (%)",
            color="Mission Type",
            size="Crew Size",
            hover_name="Mission Name",
            title="Mission Cost vs Success Rate",
            template="plotly_dark",
            trendline="ols",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.scatter(
            filtered,
            x="Mission Duration (years)",
            y="Scientific Yield (points)",
            color="Target Type",
            size="Mission Cost (billion USD)",
            hover_name="Mission Name",
            title="Mission Duration vs Scientific Yield",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        cost_type = (
            filtered.groupby("Mission Type")["Mission Cost (billion USD)"]
            .mean()
            .reset_index()
        )
        fig = px.bar(
            cost_type,
            x="Mission Type",
            y="Mission Cost (billion USD)",
            color="Mission Type",
            title="Avg Cost by Mission Type",
            template="plotly_dark",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        # Correlation heatmap
        num_cols = [
            "Distance from Earth (light-years)",
            "Mission Duration (years)",
            "Mission Cost (billion USD)",
            "Scientific Yield (points)",
            "Crew Size",
            "Mission Success (%)",
            "Fuel Consumption (tons)",
            "Payload Weight (tons)",
        ]
        corr = filtered[num_cols].corr().round(2)
        short_labels = [
            "Distance", "Duration", "Cost", "Sci Yield",
            "Crew", "Success", "Fuel", "Payload"
        ]
        fig = px.imshow(
            corr,
            x=short_labels,
            y=short_labels,
            color_continuous_scale="RdBu",
            zmin=-1, zmax=1,
            title="Feature Correlation Heatmap",
            template="plotly_dark",
            text_auto=True,
        )
        st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 – Rocket Trajectory Simulation (ODE)
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("🛸 Rocket Trajectory Simulator")
    st.markdown(
        "Simulate rocket ascent using Newton's 2nd Law with thrust, gravity, and drag. "
        "Adjust parameters to see how they affect altitude and velocity."
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        thrust = st.slider("Thrust (kN)", 1000, 10000, 3500, step=100)
        payload = st.slider("Payload Mass (kg)", 500, 20000, 5000, step=500)
    with sc2:
        fuel_mass = st.slider("Initial Fuel Mass (kg)", 5000, 100000, 30000, step=1000)
        burn_rate = st.slider("Fuel Burn Rate (kg/s)", 50, 500, 150, step=10)
    with sc3:
        drag_coeff = st.slider("Drag Coefficient (Cd)", 0.1, 1.0, 0.3, step=0.05)
        rocket_area = st.slider("Rocket Cross-section Area (m²)", 1.0, 20.0, 5.0, step=0.5)

    # Constants
    g0 = 9.81          # m/s²
    rho0 = 1.225       # kg/m³ sea-level air density
    H = 8500           # scale height (m)
    thrust_N = thrust * 1000  # convert kN to N

    def rocket_odes(t, y):
        altitude, velocity = y
        mass_dry = payload
        fuel_remaining = max(fuel_mass - burn_rate * t, 0)
        total_mass = mass_dry + fuel_remaining

        rho = rho0 * np.exp(-altitude / H)
        drag = 0.5 * rho * drag_coeff * rocket_area * velocity**2 * np.sign(velocity)

        current_thrust = thrust_N if fuel_remaining > 0 else 0
        accel = (current_thrust - drag) / total_mass - g0

        return [velocity, accel]

    burn_time = fuel_mass / burn_rate
    t_end = burn_time + 60   # simulate 60s after burnout
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, 1000)
    y0 = [0.0, 0.0]

    sol = solve_ivp(rocket_odes, t_span, y0, t_eval=t_eval, method="RK45", max_step=1.0)

    alt_km = sol.y[0] / 1000
    vel_ms = sol.y[1]
    time_s = sol.t

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_s, y=alt_km, mode="lines", name="Altitude",
                                  line=dict(color="#00d4ff", width=2)))
        fig.add_vline(x=burn_time, line_dash="dash", line_color="orange",
                      annotation_text="Burnout", annotation_position="top right")
        fig.update_layout(
            title="Altitude vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Altitude (km)",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time_s, y=vel_ms, mode="lines", name="Velocity",
                                  line=dict(color="#ff6b6b", width=2)))
        fig.add_vline(x=burn_time, line_dash="dash", line_color="orange",
                      annotation_text="Burnout", annotation_position="top right")
        fig.update_layout(
            title="Velocity vs Time",
            xaxis_title="Time (s)",
            yaxis_title="Velocity (m/s)",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)

    max_alt = alt_km.max()
    max_vel = vel_ms.max()
    m1, m2, m3 = st.columns(3)
    m1.metric("Max Altitude", f"{max_alt:.2f} km")
    m2.metric("Max Velocity", f"{max_vel:.1f} m/s")
    m3.metric("Burn Time", f"{burn_time:.0f} s")

    st.info(
        "**Physics model:** Newton's 2nd Law — net force = Thrust − Drag − Weight. "
        "Air density decreases exponentially with altitude (isothermal atmosphere). "
        "Fuel burns at a constant rate until exhausted."
    )


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 – Raw Data
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.subheader(f"Dataset — {len(filtered)} missions (filtered)")
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True, height=500)
    csv = filtered.to_csv(index=False).encode()
    st.download_button("⬇️ Download Filtered CSV", csv, "filtered_missions.csv", "text/csv")

st.markdown("---")
st.caption("Built with Streamlit · Plotly · Pandas · SciPy | Space Mission Analytics Dashboard")
