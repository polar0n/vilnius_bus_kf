import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
from filterpy.kalman import KalmanFilter, IMMEstimator


ICON_SVG = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxMDAiIGhlaWdodD0iMTAwIiB2aWV3Qm94PSIwIDAgMTAwIDEwMCI+PHBvbHlnb24gcG9pbnRzPSI1MCwwIDk1LDk1IDUwLDcwIDUsOTUgNTAsMCIgZmlsbD0icmVkIiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjIiLz48L3N2Zz4="

def get_live_buses():
    url = "http://m.stops.lt/vilnius/gps.txt"
    try:
        # Load ID (index 6) for unique identification
        df = pd.read_csv(url, header=None, encoding="utf-8", index_col=False)
        df["lon"] = df[2] / 1000000
        df["lat"] = df[3] / 1000000
        df["angle"] = 360 - pd.to_numeric(df[5], errors="coerce").fillna(0)
        df["route_name"] = df[1].astype(str)
        df["bus_id"] = df[6].astype(str)
        df["speed"] = df[5].astype(float)

        df["icon_data"] = None
        for i in df.index:
            df.at[i, "icon_data"] = {"url": ICON_SVG, "width": 100, "height": 100}

        return df[["lat", "lon", "angle", "route_name", "bus_id", "speed", "icon_data"]].dropna()
    except:
        return pd.DataFrame()


def init_imm(x_init):
    # Measurement Noise
    R = np.eye(2) * 1e-8

    # Transition Probability Matrix
    # Row 0: Stationary, Row 1: CV, Row 2: CA
    # Most of the probability is on the diagonal (staying in current mode)
    M = np.array([[0.94, 0.01, 0.01],  # If stopped, 95% stay stopped, 5% start moving
                  [0.05, 0.85, 0.10],  # If cruising, 90% stay cruising, 5% stop/brake
                  [0.1, 0.10, 0.80]]) # If maneuvering, 90% stay maneuvering, 10% cruise

    # Model 1: Stationary
    kf1 = KalmanFilter(dim_x=6, dim_z=2)
    # F is Identity: position, velocity, and accel don't change
    kf1.F = np.eye(6)
    kf1.H = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]])
    kf1.R = R.copy()
    # Q is extremely low: we don't expect the "stationary" state to drift
    kf1.Q = np.eye(6) * 1e-14

    # Model 2: Constant Velocity (CV)
    kf2 = KalmanFilter(dim_x=6, dim_z=2)
    dt = 10.0
    # State Transition Matrix (Physics: pos = pos + v*dt)
    kf2.F = np.array([[1, 0, dt, 0, 0, 0],
                    [0, 1, 0, dt, 0, 0],
                    [0, 0, 1,  0, 0, 0],
                    [0, 0, 0,  1, 0, 0],
                    [0, 0, 0,  0, 1, 0], # Accel stays 0
                    [0, 0, 0,  0, 0, 1]])
    kf2.H = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]])
    kf2.P *= .1 # Initial uncertainty
    kf2.R = R.copy() # Measurement noise (GPS jitter)
    kf2.Q = np.eye(6) * 1e-10  # Process noise (Bus acceleration/braking)

    # Model 3: Constant Acceleration (CA)
    kf3 = KalmanFilter(dim_x=6, dim_z=2)
    dt2 = 0.5 * (dt**2)
    kf3.F = np.array([[1, 0, dt, 0, dt2, 0],   # p = p + vt + 0.5at^2
                      [0, 1, 0, dt, 0, dt2],
                      [0, 0, 1, 0, dt, 0],     # v = v + at
                      [0, 0, 0, 1, 0, dt],
                      [0, 0, 0, 0, 1, 0],      # a = a
                      [0, 0, 0, 0, 0, 1]])
    kf3.H = np.array([[1, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0]])
    kf3.R = R.copy()
    # Q is high: allows the filter to rapidly adjust acceleration
    # to "match" the braking or speeding up of the bus.
    kf3.Q = np.eye(6) * 1e-8

    imm = IMMEstimator([kf1, kf2, kf3], np.ones(3)/3, M)
    imm.x = x_init
    return imm


def get_x(bus_data: dict) -> np.array:
    angle = np.radians(bus_data["angle"])
    speed = bus_data["speed"] / (3600.0 * 111320.0)
    return np.array([
        [bus_data["lon"]],
        [bus_data["lat"]],
        [np.sin(angle) * speed],
        [np.cos(angle) * speed],
        [0.0],
        [0.0],
    ])


def get_z(bus_data: dict) -> np.array:
    return np.array([
        [bus_data["lon"]],
        [bus_data["lat"]],
    ])


st.title("Live Kalman Filter Vilnius Bus Tracker")
target_bus_id = st.sidebar.text_input("Enter Bus ID to Track", help="Hover over a bus on the map to see its ID")

# --- PLACEHOLDERS FOR SIDEBAR CONTENT ---
sidebar_metrics = st.sidebar.empty()
sidebar_errors = st.sidebar.container()

if "map_view" not in st.session_state:
    st.session_state.map_view = pdk.ViewState(latitude=54.6872, longitude=25.2797, zoom=12)
if "imm" not in st.session_state:
    st.session_state.imm = None
if "tracking" not in st.session_state:
    st.session_state.tracking = False
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "error_history" not in st.session_state:
    st.session_state.error_history = []
if "current_target" not in st.session_state:
    st.session_state.current_target = None


@st.fragment(run_every=10)
def bus_map_fragment(selected_id, metric_container, error_container):
    full_df = get_live_buses()

    if full_df.empty:
        st.warning("No data available.")
        return

    layers = []

    # Reset tracking and history if the user changes the target bus
    if selected_id != st.session_state.current_target:
        st.session_state.current_target = selected_id
        st.session_state.error_history = []
        st.session_state.tracking = False
        st.session_state.imm = None
        st.session_state.last_prediction = None

    if selected_id:
        display_df = full_df[full_df["bus_id"] == selected_id]
        if display_df.empty:
            error_container.error(f"Bus ID {selected_id} not found.")
            display_df = full_df
        else:
            bus = display_df.iloc[0].to_dict()
            if not st.session_state.tracking:
                st.session_state.tracking = True
                st.session_state.imm = init_imm(get_x(bus))
                st.session_state.imm.predict()
            else:
                # Calculate Error using the last prediction and current GPS
                if st.session_state.last_prediction:
                    last_lon, last_lat, _ = st.session_state.last_prediction
                    meas_lon, meas_lat = bus["lon"], bus["lat"]

                    error_deg = np.sqrt((last_lon - meas_lon)**2 + (last_lat - meas_lat)**2)
                    error_meters = error_deg * 111320

                    # Store latest error at the start of the list
                    st.session_state.error_history.insert(0, error_meters)
                    # Keep only last 10 entries
                    st.session_state.error_history = st.session_state.error_history[:10]

                # Display error history in sidebar container
                with metric_container.container(): # Using .container() on an .empty() slot clears it
                    if st.session_state.error_history:
                        st.subheader("Error History")
                        for i, err in enumerate(st.session_state.error_history[:5]):
                            label = "Latest" if i == 0 else f"T -{i*10}s"
                            st.metric(label, f"{err:.2f} m")

                var_x = st.session_state.imm.P[0, 0]
                var_y = st.session_state.imm.P[1, 1]
                std_dev_meters = np.sqrt(max(var_x, var_y)) * 111320

                curr_lon = float(st.session_state.imm.x[0][0])
                curr_lat = float(st.session_state.imm.x[1][0])

                st.session_state.last_prediction = (
                    curr_lon, curr_lat, std_dev_meters
                )

                st.session_state.imm.update(get_z(bus))
                st.session_state.imm.predict()

            # Prediction visualization layers
            var_x = st.session_state.imm.P[0, 0]
            var_y = st.session_state.imm.P[1, 1]
            std_dev_meters = np.sqrt(max(var_x, var_y)) * 111320
            curr_lon = float(st.session_state.imm.x[0][0])
            curr_lat = float(st.session_state.imm.x[1][0])
            curr_pred_df = pd.DataFrame([{"lon": curr_lon, "lat": curr_lat, "std": std_dev_meters}])

            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=curr_pred_df,
                get_position="[lon, lat]",
                get_color=[0, 255, 0, 50],
                get_radius="std",
                radius_units="'meters'",
                stroked=True,
                get_line_color=[0, 255, 0, 255],
                line_width_min_pixels=0.5,
                pickable=False
            ))

            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=curr_pred_df,
                get_position="[lon, lat]",
                get_color=[0, 255, 0, 255],
                get_radius=5,
                radius_units="'pixels'",
                pickable=False
            ))

            if st.session_state.last_prediction:
                last_lon, last_lat, last_std = st.session_state.last_prediction
                last_pred_df = pd.DataFrame([{"lon": last_lon, "lat": last_lat, "std": last_std}])
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=last_pred_df,
                    get_position="[lon, lat]",
                    get_color=[255, 255, 0, 50],
                    get_radius="std",
                    radius_units="'meters'",
                    stroked=True,
                    get_line_color=[255, 255, 0, 150],
                    line_width_min_pixels=0.5,
                    pickable=False
                ))
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=last_pred_df,
                    get_position="[lon, lat]",
                    get_color=[255, 255, 0, 150],
                    get_radius=5,
                    radius_units="'pixels'",
                    pickable=False
                ))
    else:
        display_df = full_df
        st.session_state.tracking = False

    icon_layer = pdk.Layer(
        "IconLayer",
        data=display_df,
        get_icon="icon_data",
        get_position="[lon, lat]",
        get_angle="angle",
        get_size=25,
        size_units="'pixels'",
        pickable=True
    )
    layers.append(icon_layer)

    text_layer = pdk.Layer(
        "TextLayer",
        data=display_df,
        get_position="[lon, lat]",
        get_text="route_name",
        get_size=12,
        get_color=[255, 255, 255],
        pickable=False
    )
    layers.append(text_layer)

    st.pydeck_chart(
        pdk.Deck(
            layers=layers,
            initial_view_state=st.session_state.map_view,
            map_style=None,
            tooltip={
                "html": "<b>Bus ID:</b> {bus_id}<br/><b>Route:</b> {route_name}",
                "style": {"color": "white"}
            }
        ),
        key="vilnius_transit_map"
    )

bus_map_fragment(target_bus_id, sidebar_metrics, sidebar_errors)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cascadia+Code:ital,wght@0,200..700;1,200..700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Cascadia Code', sans-serif;
        font-size: 14px;
        font-weight: 500;
    }
    </style>
""", unsafe_allow_html=True)

st.divider()

# 1. Operation Instructions
st.header("Operation Instructions")
st.markdown(r"""
To begin tracking a specific bus:
1. **Locate a Target:** Hover the cursor over any bus to find its **Bus ID**.
2. **Input ID:** Insert the ID into the sidebar.
3. **Wait for Convergence:** Because the IMM tracks position, velocity, and acceleration across three models, it may take **1 minute** for the filters to accurately understand the current movement of the bus.
""")

st.divider()

# 2. Visual Legend and Methodology
st.header("Visual Legend and Methodology")
st.markdown("""
This system uses an **IMM Estimator**, which runs three specialized Kalman Filters simultaneously to handle different movement behaviors.
""")

legend_df = pd.DataFrame([
    {"Element": "Current Position", "Visual": "Red Arrow", "Meaning": "Raw GPS telemetry", "Basis": "Direct stream from stops.lt"},
    {"Element": "Prediction", "Visual": "Green Circle", "Meaning": "Weighted trajectory projection", "Basis": "Blended output of Stationary, CV, and CA models"},
    {"Element": "Previous Target", "Visual": "Yellow Circle", "Meaning": "Last step's prediction", "Basis": "Retained from $x_{t-1}$"},
    {"Element": "Uncertainty", "Visual": "Circular Radius", "Meaning": "Confidence Interval ($1\\sigma$)", "Basis": "Weighted Covariance ($P$) of all active models"}
])
st.table(legend_df)

st.divider()

# 3. History Section: From the Moon to the Streets
st.header("A Brief History: Why One Filter Isn't Enough")
st.image("aoc.webp")
st.markdown(r"""
The Kalman Filter is named after **Rudolf E. Kálmán**, who published his seminal paper in 1960. Its most famous early application was the **Apollo Program**.

NASA engineers used this recursive algorithm to solve the navigation problem for the moon landing. The onboard computers had limited memory and processed noisy sensor data from the inertial guidance system. The Kalman Filter allowed the **Apollo 11 Guidance Computer** to provide smooth, accurate estimates of the Lunar Module's position and velocity in real-time. Without this algorithm, landing on the lunar surface with such precision would have been significantly more difficult.

In the vacuum of space, objects move with predictable physics. However, driving in a city is more unpredictable, it breaks at stops, accelerates on highways, turns on roundabouts.

A single filter is always a compromise: tuning it for smooth predictions during movement, leads to lags during stops; tuning it for responsiveness leads to jumpiness when the vehicle is not moving. The **IMM approach** solves this by running three filters for each movement mode of a vehicle at once using probability to decide which one describes the bus at this exact moment. One filter for when the vehicle is stationary, one when it is moving at a constant speed, and one when it has a constant acceleration (whether positive or negative).
""")

st.divider()

# 4. IMM Architecture
st.header("The Mathematics of the IMM")

st.subheader("1. The 6D State Vector")
st.markdown(r"""
To account for complex maneuvers, the state vector $x$ has **six dimensions**, which tracks position, velocity, and acceleration for both longitude and latitude:

$$x = \begin{bmatrix} lon \\ lat \\ v_{lon} \\ v_{lat} \\ a_{lon} \\ a_{lat} \end{bmatrix}$$
""")

st.subheader("2. The Three-Model Hypothesis")
st.markdown(r"""
The IMM maintains three separate filters, each representing a different "Mode of Flight":

1. **Stationary Model ($F=I$):** Assumes the bus is at a stop. $Q$ is extremely low ($10^{-14}$) to lock the position and prevent the prediction from "drifting" due to minor GPS noise.
	* This low $Q$ can be noticed on the map while tracking as the small/precise circle when the bus is not moving.
2. **Constant Velocity (CV):** Assumes the bus is "cruising" (moving at a constant speed). It accounts for velocity but assumes acceleration is zero.
3. **Constant Acceleration (CA):** Assumes the bus is actively changing speed. $Q$ is high ($10^{-8}$) to allow the filter to rapidly adjust the acceleration $a$ to match the bus's physical behavior.
""")

st.subheader("3. Transition Probability Matrix ($M$)")
st.markdown(r"""
The "Interaction" in IMM comes from the matrix $M$, which defines the probability of the bus switching modes between 10-second pings:

$$M = \begin{bmatrix} 0.94 & 0.01 & 0.01 \\ 0.05 & 0.85 & 0.10 \\ 0.10 & 0.10 & 0.80 \end{bmatrix}$$

* **Diagonal Elements (0.94, 0.85, 0.80):** These represent the high probability that a bus stays in its current movement mode.
* **Off-Diagonal Elements:** These represent the probability of switching. For example, there is a $10\%$ probability that a bus moving at a constant speed (CV) will suddenly begin accelerating or decelerating (CA).
""")

st.divider()

# 5. Why Uncertainty Grows During Maneuvers
st.header("Why the $1\\sigma$ Circle Expands")
st.markdown(r"""
You will notice the green circle **growing significantly** when the bus brakes for a light or speeds up. This is a critical feature of the IMM logic.

### Maneuver Detection
During acceleration or deceleration (known as a maneuvers), the bus's movements no longer match the **Stationary** or **CV** models which gets detected by the IMM and it shifts its "Model Probability" (weight) toward the **Constant Acceleration (CA)** model.

Because the CA model has a much higher **Process Noise ($Q$)**, the blended Covariance Matrix ($P$) expands. Visually, the circle grows because the filter is admitting: *"The bus is changing its behavior, so I am temporarily less certain about exactly where it will be in 10 seconds."* Once the bus settles back into a constant speed, the CV model regains weight, and the circle shrinks again.
""")

st.divider()

# 6. Tuning the Parameters
st.header("Hardcoded Parameters and Blending")
st.markdown(r"""
### Measurement Noise ($R$)
**Variable:** $1e-8$
This reflects the precision of the Vilnius Bus Live GPS feed. We treat this as constant across all models.

### Blended Prediction
The final position seen on the map is a **weighted sum** of all positions predicted by the filters:


$$\hat{x}_{IMM} = \sum_{i=1}^{3} \mu_i \hat{x}_i$$

Where $\mu_i$ is the probability that model $i$ is the correct description of the bus's current state.
""")