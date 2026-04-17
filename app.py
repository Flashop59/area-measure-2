import streamlit as st
import pandas as pd
import numpy as np
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN
from scipy.spatial import ConvexHull
import folium
from folium import plugins
from geopy.distance import geodesic
from streamlit_folium import folium_static
from math import cos, radians

# ---------------------------
# Config
# ---------------------------
MAPBOX_TOKEN = "pk.eyJ1IjoiZmxhc2hvcDAwNyIsImEiOiJjbW44a2s5MzcwYm5vMnFzZGloMGpodDI2In0.HO3qwCL8N4YSH3PmwVc3mw"

# ---------------------------
# Helpers
# ---------------------------
def latlon_to_local_xy(points):
    """
    Convert lat/lng points to local X/Y meters using equirectangular approximation.
    Good enough for field-size area calculations.
    points: Nx2 array -> [lat, lng]
    """
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.empty((0, 2))

    lat0 = np.mean(points[:, 0])
    lon0 = np.mean(points[:, 1])

    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * cos(radians(lat0))

    x = (points[:, 1] - lon0) * meters_per_deg_lon
    y = (points[:, 0] - lat0) * meters_per_deg_lat

    return np.column_stack((x, y))


def calculate_convex_hull_area_m2(points):
    """
    Calculate convex hull area in square meters from lat/lng points.
    """
    points = np.asarray(points, dtype=float)

    if len(points) < 3:
        return 0.0

    try:
        xy = latlon_to_local_xy(points)

        # ConvexHull needs unique points
        xy_unique = np.unique(xy, axis=0)
        if len(xy_unique) < 3:
            return 0.0

        hull = ConvexHull(xy_unique)
        hull_points = xy_unique[hull.vertices]
        poly = Polygon(hull_points)

        return float(poly.area)
    except Exception:
        return 0.0


def calculate_centroid(points):
    points = np.asarray(points, dtype=float)
    if len(points) == 0:
        return np.array([np.nan, np.nan])
    return np.mean(points, axis=0)


def process_data(gps_data):
    """
    gps_data must contain:
    lat, lng, Timestamp
    """
    gps_data = gps_data.copy()

    gps_data = gps_data.dropna(subset=["lat", "lng", "Timestamp"])
    gps_data = gps_data.sort_values("Timestamp").reset_index(drop=True)

    if gps_data.empty:
        raise ValueError("No valid GPS data found after cleaning.")

    coords = gps_data[["lat", "lng"]].values

    # DBSCAN on raw lat/lng degrees
    # eps=0.00008 ~ around 8-10 meters depending on latitude
    db = DBSCAN(eps=0.00008, min_samples=11).fit(coords)
    gps_data["field_id"] = db.labels_

    fields = gps_data[gps_data["field_id"] != -1].copy()

    if fields.empty:
        raise ValueError("No valid field clusters detected.")

    # Area
    field_areas_m2 = fields.groupby("field_id").apply(
        lambda df: calculate_convex_hull_area_m2(df[["lat", "lng"]].values)
    )
    field_areas_gunthas = field_areas_m2 / 101.17

    # Time
    field_times = fields.groupby("field_id").apply(
        lambda df: (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 60.0
    )

    # Start/end dates
    field_dates = fields.groupby("field_id").agg(
        start_date=("Timestamp", "min"),
        end_date=("Timestamp", "max")
    )

    # Keep only fields >= 5 gunthas
    valid_fields = field_areas_gunthas[field_areas_gunthas >= 5].index

    if len(valid_fields) == 0:
        raise ValueError("No fields above 5 gunthas found.")

    field_areas_gunthas = field_areas_gunthas.loc[valid_fields]
    field_times = field_times.loc[valid_fields]
    field_dates = field_dates.loc[valid_fields]

    # Sort fields by actual visit order
    ordered_field_ids = field_dates.sort_values("start_date").index.tolist()

    # Centroids
    centroids = fields.groupby("field_id").apply(
        lambda df: calculate_centroid(df[["lat", "lng"]].values)
    )

    # Travel data: one value per field, last field gets NaN
    travel_distances = []
    travel_times = []

    for i, field_id in enumerate(ordered_field_ids):
        if i < len(ordered_field_ids) - 1:
            next_field_id = ordered_field_ids[i + 1]

            end_point = fields[fields["field_id"] == field_id][["lat", "lng"]].iloc[-1].values
            start_point = fields[fields["field_id"] == next_field_id][["lat", "lng"]].iloc[0].values

            distance_km = geodesic(
                (float(end_point[0]), float(end_point[1])),
                (float(start_point[0]), float(start_point[1]))
            ).kilometers

            time_min = (
                field_dates.loc[next_field_id, "start_date"] -
                field_dates.loc[field_id, "end_date"]
            ).total_seconds() / 60.0

            # prevent negative gaps due to overlap/noise
            if time_min < 0:
                time_min = 0.0

            travel_distances.append(distance_km)
            travel_times.append(time_min)
        else:
            travel_distances.append(np.nan)
            travel_times.append(np.nan)

    combined_df = pd.DataFrame({
        "Field ID": ordered_field_ids,
        "Area (Gunthas)": [field_areas_gunthas.loc[fid] for fid in ordered_field_ids],
        "Time (Minutes)": [field_times.loc[fid] for fid in ordered_field_ids],
        "Start Date": [field_dates.loc[fid, "start_date"] for fid in ordered_field_ids],
        "End Date": [field_dates.loc[fid, "end_date"] for fid in ordered_field_ids],
        "Travel Distance to Next Field (km)": travel_distances,
        "Travel Time to Next Field (minutes)": travel_times
    })

    total_area = combined_df["Area (Gunthas)"].sum()
    total_time = combined_df["Time (Minutes)"].sum()
    total_travel_distance = np.nansum(combined_df["Travel Distance to Next Field (km)"])
    total_travel_time = np.nansum(combined_df["Travel Time to Next Field (minutes)"])

    # Map
    map_center = [gps_data["lat"].mean(), gps_data["lng"].mean()]
    m = folium.Map(location=map_center, zoom_start=16, tiles=None)

    folium.TileLayer(
        tiles=f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/tiles/256/{{z}}/{{x}}/{{y}}?access_token={MAPBOX_TOKEN}",
        attr="Mapbox Satellite Imagery",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(m)

    plugins.Fullscreen(position="topright").add_to(m)
    folium.LayerControl().add_to(m)

    valid_field_set = set(ordered_field_ids)

    # Plot points
    for _, row in gps_data.iterrows():
        color = "blue" if row["field_id"] in valid_field_set else "red"
        folium.CircleMarker(
            location=(row["lat"], row["lng"]),
            radius=2,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.8
        ).add_to(m)

    # Plot centroid markers for valid fields
    for fid in ordered_field_ids:
        centroid = centroids.loc[fid]
        folium.Marker(
            location=(float(centroid[0]), float(centroid[1])),
            popup=f"Field {fid}<br>Area: {field_areas_gunthas.loc[fid]:.2f} gunthas",
            tooltip=f"Field {fid}",
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m)

    return m, combined_df, total_area, total_time, total_travel_distance, total_travel_time


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Field Area and Time Calculation", layout="wide")

st.title("Field Area and Time Calculation from GPS CSV")

st.write(
    "Upload your CSV file containing GPS points. "
    "The file should have an 'Ignition' column with values in 'lat,lon' format, "
    "and preferably a timestamp column."
)

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        if "Ignition" not in df.columns:
            st.error("CSV must contain an 'Ignition' column with 'lat,lon' values.")
            st.stop()

        # Split Ignition into lat/lon
        split_cols = df["Ignition"].astype(str).str.split(",", expand=True)
        if split_cols.shape[1] < 2:
            st.error("Ignition column does not contain valid 'lat,lon' data.")
            st.stop()

        df["lat"] = pd.to_numeric(split_cols[0].str.strip(), errors="coerce")
        df["lng"] = pd.to_numeric(split_cols[1].str.strip(), errors="coerce")

        # Detect timestamp column
        possible_time_cols = [
            "time", "Time", "timestamp", "Timestamp", "date", "Date", "datetime", "Datetime"
        ]

        time_col = None
        for col in possible_time_cols:
            if col in df.columns:
                time_col = col
                break

        if time_col is not None:
            df["Timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
        else:
            # fallback: use row order as pseudo-time
            df["Timestamp"] = pd.date_range(
                start=pd.Timestamp.now().floor("s"),
                periods=len(df),
                freq="s"
            )

        df = df.dropna(subset=["lat", "lng", "Timestamp"]).reset_index(drop=True)

        if df.empty:
            st.error("No valid rows found after parsing lat, lon, and timestamp.")
            st.stop()

        map_obj, combined_df, total_area, total_time, total_travel_distance, total_travel_time = process_data(
            df[["lat", "lng", "Timestamp"]]
        )

        st.subheader("Field Map")
        folium_static(map_obj, width=1400, height=700)

        st.subheader("Field Area and Time Data")
        st.dataframe(combined_df, use_container_width=True)

        st.subheader("Total Metrics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Area", f"{total_area:.2f} Gunthas")
        col2.metric("Total Time", f"{total_time:.2f} Minutes")
        col3.metric("Total Travel Distance", f"{total_travel_distance:.2f} km")
        col4.metric("Total Travel Time", f"{total_travel_time:.2f} Minutes")

        csv = combined_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV",
            data=csv,
            file_name="field_data.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
