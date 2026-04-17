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
    points = np.asarray(points, dtype=float)

    if len(points) < 3:
        return 0.0

    try:
        xy = latlon_to_local_xy(points)
        xy_unique = np.unique(xy, axis=0)

        if len(xy_unique) < 3:
            return 0.0

        hull = ConvexHull(xy_unique)
        hull_points = xy_unique[hull.vertices]
        poly = Polygon(hull_points)

        return float(poly.area)
    except:
        return 0.0


def calculate_centroid(points):
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.array([np.nan, np.nan])

    return np.mean(points, axis=0)


# ---------------------------
# Core Processing
# ---------------------------
def process_data(gps_data):
    gps_data = gps_data.copy()
    gps_data = gps_data.dropna(subset=["lat", "lng", "Timestamp"])
    gps_data = gps_data.sort_values("Timestamp").reset_index(drop=True)

    if gps_data.empty:
        raise ValueError("No valid GPS data found.")

    coords = gps_data[["lat", "lng"]].values

    # DBSCAN clustering
    db = DBSCAN(eps=0.00008, min_samples=10).fit(coords)
    gps_data["field_id"] = db.labels_

    fields = gps_data[gps_data["field_id"] != -1].copy()

    if fields.empty:
        raise ValueError("No field clusters detected.")

    # Area
    field_areas_m2 = fields.groupby("field_id").apply(
        lambda df: calculate_convex_hull_area_m2(df[["lat", "lng"]].values)
    )
    field_areas_gunthas = field_areas_m2 / 101.17

    # Time
    field_times = fields.groupby("field_id").apply(
        lambda df: (df["Timestamp"].max() - df["Timestamp"].min()).total_seconds() / 60
    )

    # Dates
    field_dates = fields.groupby("field_id").agg(
        start_date=("Timestamp", "min"),
        end_date=("Timestamp", "max")
    )

    # Filter small fields
    valid_fields = field_areas_gunthas[field_areas_gunthas >= 5].index

    if len(valid_fields) == 0:
        raise ValueError("No fields above 5 gunthas.")

    field_areas_gunthas = field_areas_gunthas.loc[valid_fields]
    field_times = field_times.loc[valid_fields]
    field_dates = field_dates.loc[valid_fields]

    ordered_field_ids = field_dates.sort_values("start_date").index.tolist()

    # Centroids
    centroids = fields.groupby("field_id").apply(
        lambda df: calculate_centroid(df[["lat", "lng"]].values)
    )

    # Travel calc
    travel_distances = []
    travel_times = []

    for i, fid in enumerate(ordered_field_ids):
        if i < len(ordered_field_ids) - 1:
            next_fid = ordered_field_ids[i + 1]

            end_pt = fields[fields["field_id"] == fid][["lat", "lng"]].iloc[-1]
            start_pt = fields[fields["field_id"] == next_fid][["lat", "lng"]].iloc[0]

            dist = geodesic(
                (end_pt["lat"], end_pt["lng"]),
                (start_pt["lat"], start_pt["lng"])
            ).kilometers

            time_gap = (
                field_dates.loc[next_fid, "start_date"] -
                field_dates.loc[fid, "end_date"]
            ).total_seconds() / 60

            if time_gap < 0:
                time_gap = 0

            travel_distances.append(dist)
            travel_times.append(time_gap)
        else:
            travel_distances.append(np.nan)
            travel_times.append(np.nan)

    combined_df = pd.DataFrame({
        "Field ID": ordered_field_ids,
        "Area (Gunthas)": [field_areas_gunthas.loc[f] for f in ordered_field_ids],
        "Time (Minutes)": [field_times.loc[f] for f in ordered_field_ids],
        "Start Date": [field_dates.loc[f, "start_date"] for f in ordered_field_ids],
        "End Date": [field_dates.loc[f, "end_date"] for f in ordered_field_ids],
        "Travel Distance (km)": travel_distances,
        "Travel Time (min)": travel_times
    })

    # Totals
    total_area = combined_df["Area (Gunthas)"].sum()
    total_time = combined_df["Time (Minutes)"].sum()
    total_dist = np.nansum(combined_df["Travel Distance (km)"])
    total_travel_time = np.nansum(combined_df["Travel Time (min)"])

    # Map
    center = [gps_data["lat"].mean(), gps_data["lng"].mean()]
    m = folium.Map(location=center, zoom_start=16)

    plugins.Fullscreen().add_to(m)

    for _, row in gps_data.iterrows():
        color = "blue" if row["field_id"] in ordered_field_ids else "red"
        folium.CircleMarker(
            location=(row["lat"], row["lng"]),
            radius=2,
            color=color,
            fill=True
        ).add_to(m)

    for fid in ordered_field_ids:
        c = centroids.loc[fid]
        folium.Marker(
            location=(c[0], c[1]),
            tooltip=f"Field {fid}"
        ).add_to(m)

    return m, combined_df, total_area, total_time, total_dist, total_travel_time


# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Field Area Calculator", layout="wide")

st.title("Field Area from GPS Excel")

uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)

        # normalize column names
        df.columns = df.columns.str.strip().str.lower()

        if "lat" not in df.columns or "lng" not in df.columns:
            st.error("Excel must contain 'lat' and 'lng'")
            st.stop()

        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lng"] = pd.to_numeric(df["lng"], errors="coerce")

        # timestamp detection
        time_col = None
        for col in ["timestamp", "time", "date", "datetime"]:
            if col in df.columns:
                time_col = col
                break

        if time_col:
            df["Timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
        else:
            df["Timestamp"] = pd.date_range(
                start=pd.Timestamp.now(),
                periods=len(df),
                freq="s"
            )

        df = df.dropna(subset=["lat", "lng", "Timestamp"])

        map_obj, table, ta, tt, td, ttt = process_data(df)

        st.subheader("Map")
        folium_static(map_obj, width=1400, height=600)

        st.subheader("Data")
        st.dataframe(table)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Area", f"{ta:.2f} Gunthas")
        c2.metric("Total Time", f"{tt:.2f} min")
        c3.metric("Travel Distance", f"{td:.2f} km")
        c4.metric("Travel Time", f"{ttt:.2f} min")

        st.download_button(
            "Download CSV",
            table.to_csv(index=False),
            "output.csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")
