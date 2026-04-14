import numpy as np
import streamlit as st
import pandas as pd
import pydeck as pdk
from filterpy.kalman import KalmanFilter


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


def init_kf(x_init):
    kf = KalmanFilter(dim_x=4, dim_z=2)
    dt = 10.0 # seconds between pings
    # State Transition Matrix (Physics: pos = pos + v*dt)
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1,  0],
                     [0, 0, 0,  1]])
    # Measurement Matrix (We only observe Lat/Lon, not velocity)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.P *= .1 # Initial uncertainty
    kf.R = np.eye(2) * 1e-8 # Measurement noise (GPS jitter)
    kf.Q = np.eye(4) * 1e-9  # Process noise (Bus acceleration/braking)
    kf.x = x_init
    return kf


def get_x(bus_data: dict) -> np.array:
    angle = np.radians(bus_data["angle"])
    speed = bus_data["speed"] / (3600.0 * 111320.0)
    return [
        [bus_data["lon"]],
        [bus_data["lat"]],
        [np.sin(angle) * speed],
        [np.cos(angle) * speed],
    ]


def get_z(bus_data: dict) -> np.array:
    return [
        [bus_data["lon"]],
        [bus_data["lat"]],
    ]


st.title("Live Kalman Filter Vilnius Bus Tracker")
target_bus_id = st.sidebar.text_input("Enter Bus ID to Track", help="Hover over a bus on the map to see its ID")

# --- PLACEHOLDERS FOR SIDEBAR CONTENT ---
sidebar_metrics = st.sidebar.empty()
sidebar_errors = st.sidebar.container()

if "map_view" not in st.session_state:
    st.session_state.map_view = pdk.ViewState(latitude=54.6872, longitude=25.2797, zoom=12)
if "kalman" not in st.session_state:
    st.session_state.kalman = None
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
        st.session_state.kalman = None
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
                st.session_state.kalman = init_kf(get_x(bus))
                st.session_state.kalman.predict()
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

                var_x = st.session_state.kalman.P[0, 0]
                var_y = st.session_state.kalman.P[1, 1]
                std_dev_meters = np.sqrt(max(var_x, var_y)) * 111320

                curr_lon = float(st.session_state.kalman.x[0][0])
                curr_lat = float(st.session_state.kalman.x[1][0])

                st.session_state.last_prediction = (
                    curr_lon, curr_lat, std_dev_meters
                )

                st.session_state.kalman.update(get_z(bus))
                st.session_state.kalman.predict()

            # Prediction visualization layers
            var_x = st.session_state.kalman.P[0, 0]
            var_y = st.session_state.kalman.P[1, 1]
            std_dev_meters = np.sqrt(max(var_x, var_y)) * 111320
            curr_lon = float(st.session_state.kalman.x[0][0])
            curr_lat = float(st.session_state.kalman.x[1][0])
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

# --- APPEND TO THE END OF YOUR SCRIPT ---

st.divider()

# 1. Operation Instructions
st.header("Instructions")
st.markdown(r"""
To begin tracking a specific vehicle with the Kalman Filter:
1. **Locate a Target:** Hover your cursor over any red bus icon on the map to reveal its unique **Bus ID** in the tooltip.
2. **Input ID:** Type the numerical ID into the sidebar text field.
3. **Wait for Initialization:** The map will clear all other vehicles. Please wait **10–20 seconds** for the filter to collect its first two pings and begin projecting the trajectory.
""")

st.divider()

# 2. Visual Legend
st.header("Visual Legend and Methodology")
st.markdown("""
The tracking system uses a recursive mathematical algorithm to estimate the "true" state of a bus, accounting for GPS noise and transmission delays.
""")

legend_df = pd.DataFrame([
    {"Element": "Current Position", "Visual": "Red Arrow", "Meaning": "Raw GPS telemetry", "Basis": "Direct stream from stops.lt"},
    {"Element": "Prediction", "Visual": "Green Circle", "Meaning": "Expected position in 10s", "Basis": "Extrapolated via State Transition Matrix ($F$)"},
    {"Element": "Previous Target", "Visual": "Yellow Circle", "Meaning": "Last step's prediction", "Basis": "Retained from $x_{t-1}$"},
    {"Element": "Uncertainty", "Visual": "Circular Radius", "Meaning": "1-Std Confidence Interval", "Basis": "Diagonal elements of Covariance Matrix ($P$)"}
])
st.table(legend_df)

st.divider()

# 3. History Section
st.header("A Brief History: From the Moon to Vilnius")
st.markdown(r"""
The Kalman Filter is named after **Rudolf E. Kálmán**, who published his seminal paper in 1960. Its most famous early application was the **Apollo Program**.

NASA engineers used this recursive algorithm to solve the navigation problem for the moon landing. The onboard computers had limited memory and processed noisy sensor data from the inertial guidance system. The Kalman Filter allowed the **Apollo 11 Guidance Computer** to provide smooth, accurate estimates of the Lunar Module's position and velocity in real-time. Without this algorithm, landing on the lunar surface with such precision would have been significantly more difficult.
""")

st.divider()

# 4. Mathematical Foundation
st.header("The Mathematics of the State")

st.subheader("1. The State Vector")
st.markdown(r"""
We track the bus using a four-dimensional state vector $x$, representing 2D position and 2D velocity:
$$x = \begin{bmatrix} x_{lon} \\ x_{lat} \\ v_{lon} \\ v_{lat} \end{bmatrix}$$
""")

st.subheader("2. Prediction (Extrapolation)")
st.markdown(r"""
The filter predicts the next state using the transition matrix $F$, which assumes constant velocity over the time interval $\Delta t = 10s$:

$$x_{k|k-1} = F x_{k-1|k-1}$$

Where $F$ is defined in your code as:
$$F = \begin{bmatrix} 1 & 0 & 10 & 0 \\ 0 & 1 & 0 & 10 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$
""")

st.subheader("3. Covariance and Uncertainty")
st.markdown(r"""
The "size" of the circles on the map is determined by the covariance matrix $P$. This tracks the uncertainty in our estimate:
$$P_{k|k-1} = F P_{k-1|k-1} F^T + Q$$

Here, $Q$ is the **Process Noise** (`1e-9`), representing the physical reality that buses can accelerate or brake unexpectedly between GPS pings.
""")

st.divider()

# 5. Tuning and Parameters
st.header("Tuning the Filter: Hardcoded Parameters")
st.markdown(r"""
A Kalman Filter is a "tug-of-war" between the **mathematical model** and the **sensor measurements**.

### Measurement Matrix ($H$)
Since we only "see" longitude and latitude from the GPS, but track four variables, $H$ filters out the velocity components:
$$H = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

### Measurement Noise ($R$)
**Variable:** `kf.R = np.eye(2) * 1e-8`
This represents **GPS Jitter**. A small value like $10^{-8}$ tells the filter that our GPS is very accurate. If this were larger, the filter would ignore sudden jumps in the data, assuming they were just noise.

### The Update Step
Every 10 seconds, the filter performs a weighted average using the **Kalman Gain ($K$)**:
$$\text{New Estimate} = \text{Prediction} + K \times (\text{Measurement} - \text{Prediction})$$

If the GPS is noisy ($R$ is large), $K$ becomes small, and the filter trusts the **Prediction** more. If the bus moves unexpectedly ($Q$ is large), $K$ becomes large, and the filter trusts the **Measurement** more.
""")