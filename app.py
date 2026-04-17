import streamlit as st
import pandas as pd
import numpy as np
import folium

from math import cos, radians, sin, asin, sqrt
from shapely.geometry import LineString, Point, mapping
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from streamlit_folium import folium_static
from folium import plugins

# =========================================================
# CONFIG
# =========================================================
GUNTHA_M2 = 101.17

# =========================================================
# HELPERS
# =========================================================
def haversine_m(lat1, lon1, lat2, lon2):
    """
    Great-circle distance between two GPS points in meters.
    """
    r = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = (
        sin(dlat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    )
    c = 2 * asin(sqrt(a))
    return r * c


def latlon_to_local_xy(points, lat0=None, lon0=None):
    """
    Convert lat/lon to local XY meters using equirectangular approximation.
    points: Nx2 array [[lat, lon], ...]
    """
    points = np.asarray(points, dtype=float)

    if len(points) == 0:
        return np.empty((0, 2)), np.nan, np.nan

    if lat0 is None:
        lat0 = float(np.mean(points[:, 0]))
    if lon0 is None:
        lon0 = float(np.mean(points[:, 1]))

    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * cos(radians(lat0))

    x = (points[:, 1] - lon0) * meters_per_deg_lon
    y = (points[:, 0] - lat0) * meters_per_deg_lat

    return np.column_stack((x, y)), lat0, lon0


def local_xy_to_latlon(x, y, lat0, lon0):
    """
    Convert local XY meters back to lat/lon.
    """
    meters_per_deg_lat = 111320.0
    meters_per_deg_lon = 111320.0 * cos(radians(lat0))

    lat = lat0 + (y / meters_per_deg_lat)
    lon = lon0 + (x / meters_per_deg_lon)
    return lat, lon


def normalize_columns(df):
    """
    Auto-detect lat/lon/time columns.
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]

    lat_candidates = ["lat", "latitude"]
    lon_candidates = ["lng", "lon", "longitude", "long"]
    time_candidates = ["timestamp", "time", "date", "datetime", "created_at"]

    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    time_col = next((c for c in time_candidates if c in df.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not detect latitude/longitude columns. Use lat/lng or latitude/longitude."
        )

    out = pd.DataFrame()
    out["lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    out["lng"] = pd.to_numeric(df[lon_col], errors="coerce")

    synthetic_time = False
    if time_col is not None:
        out["Timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        synthetic_time = True
        out["Timestamp"] = pd.date_range(
            start=pd.Timestamp.now().floor("s"),
            periods=len(df),
            freq="s"
        )

    out = out.dropna(subset=["lat", "lng", "Timestamp"]).reset_index(drop=True)
    return out, synthetic_time, lat_col, lon_col, time_col


def remove_basic_gps_noise(df):
    """
    Remove impossible coordinates and duplicates.
    """
    df = df.copy()
    df = df[
        (df["lat"].between(-90, 90)) &
        (df["lng"].between(-180, 180))
    ].copy()

    df = df.drop_duplicates(subset=["lat", "lng", "Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    return df


def remove_speed_outliers(df, max_speed_kmph=15.0):
    """
    Remove points that imply impossible speed jumps.
    For farm machine use, default 15 kmph is already generous.
    """
    df = df.copy().sort_values("Timestamp").reset_index(drop=True)

    if len(df) < 3:
        return df

    keep = [True]

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        dt = (curr["Timestamp"] - prev["Timestamp"]).total_seconds()
        if dt <= 0:
            keep.append(False)
            continue

        dist_m = haversine_m(prev["lat"], prev["lng"], curr["lat"], curr["lng"])
        speed_kmph = (dist_m / dt) * 3.6

        keep.append(speed_kmph <= max_speed_kmph)

    df = df[np.array(keep, dtype=bool)].reset_index(drop=True)
    return df


def assign_sessions(df, time_gap_minutes=10):
    """
    Split data into sessions when time gap is large.
    """
    df = df.copy().sort_values("Timestamp").reset_index(drop=True)

    gaps = df["Timestamp"].diff().dt.total_seconds().fillna(0)
    new_session = (gaps > time_gap_minutes * 60).astype(int)
    df["session_id"] = new_session.cumsum()
    return df


def cluster_fields(df, dbscan_eps_m=10.0, min_samples=8):
    """
    Cluster points inside each session using XY meters.
    """
    df = df.copy()
    df["field_id"] = -1

    next_field_id = 0

    for session_id, g in df.groupby("session_id", sort=True):
        coords = g[["lat", "lng"]].to_numpy()
        xy, _, _ = latlon_to_local_xy(coords)

        if len(xy) < min_samples:
            continue

        labels = DBSCAN(eps=dbscan_eps_m, min_samples=min_samples).fit_predict(xy)

        session_idx = g.index.to_list()
        local_to_global = {}

        unique_labels = sorted(set(labels))
        for lbl in unique_labels:
            if lbl == -1:
                continue
            local_to_global[lbl] = next_field_id
            next_field_id += 1

        for row_i, lbl in zip(session_idx, labels):
            if lbl != -1:
                df.at[row_i, "field_id"] = local_to_global[lbl]

    return df


def make_coverage_geometry_area(points_latlon, implement_width_m):
    """
    Estimate covered area by buffering the machine path with implement width.
    This is much better than convex hull for field coverage.
    """
    pts = np.asarray(points_latlon, dtype=float)

    if len(pts) == 0:
        return None, 0.0, np.nan, np.nan

    xy, lat0, lon0 = latlon_to_local_xy(pts)

    # Remove repeated XY points
    xy_unique_ordered = np.array(pd.DataFrame(xy).drop_duplicates().values, dtype=float)

    if len(xy_unique_ordered) == 0:
        return None, 0.0, lat0, lon0

    radius = max(implement_width_m / 2.0, 0.05)

    if len(xy_unique_ordered) == 1:
        geom = Point(xy_unique_ordered[0]).buffer(radius, cap_style=1)
    else:
        line = LineString(xy_unique_ordered)
        geom = line.buffer(radius, cap_style=1, join_style=1)

    area_m2 = float(geom.area)
    return geom, area_m2, lat0, lon0


def representative_center(points_latlon):
    """
    Safe center marker point.
    """
    pts = np.asarray(points_latlon, dtype=float)
    if len(pts) == 0:
        return np.nan, np.nan
    return float(np.median(pts[:, 0])), float(np.median(pts[:, 1]))


def geom_to_folium_geojson(geom, lat0, lon0):
    """
    Convert local XY shapely geometry back into lat/lon GeoJSON for folium.
    """
    if geom is None or geom.is_empty:
        return None

    def convert_coords(coords):
        converted = []
        for x, y in coords:
            lat, lon = local_xy_to_latlon(x, y, lat0, lon0)
            converted.append((lon, lat))  # GeoJSON uses lon, lat
        return converted

    if geom.geom_type == "Polygon":
        exterior = convert_coords(list(geom.exterior.coords))
        interiors = [convert_coords(list(r.coords)) for r in geom.interiors]
        return {
            "type": "Polygon",
            "coordinates": [exterior] + interiors
        }

    if geom.geom_type == "MultiPolygon":
        polys = []
        for poly in geom.geoms:
            exterior = convert_coords(list(poly.exterior.coords))
            interiors = [convert_coords(list(r.coords)) for r in poly.interiors]
            polys.append([exterior] + interiors)
        return {
            "type": "MultiPolygon",
            "coordinates": polys
        }

    return None


def build_field_summary(df_fields, implement_width_m, min_field_gunthas):
    """
    Compute per-field metrics.
    """
    rows = []
    field_geom_info = {}

    for field_id, g in df_fields.groupby("field_id", sort=True):
        g = g.sort_values("Timestamp").reset_index(drop=True)
        pts = g[["lat", "lng"]].to_numpy()

        geom, area_m2, lat0, lon0 = make_coverage_geometry_area(pts, implement_width_m)
        area_gunthas = area_m2 / GUNTHA_M2

        if area_gunthas < min_field_gunthas:
            continue

        start_dt = g["Timestamp"].min()
        end_dt = g["Timestamp"].max()
        work_time_min = (end_dt - start_dt).total_seconds() / 60.0

        center_lat, center_lng = representative_center(pts)

        rows.append({
            "Field ID": int(field_id),
            "Area (Gunthas)": area_gunthas,
            "Area (m²)": area_m2,
            "Work Time (min)": work_time_min,
            "Start Date": start_dt,
            "End Date": end_dt,
            "Point Count": int(len(g)),
            "Center Lat": center_lat,
            "Center Lng": center_lng
        })

        field_geom_info[int(field_id)] = {
            "geom": geom,
            "lat0": lat0,
            "lon0": lon0,
            "points": g.copy()
        }

    if not rows:
        return pd.DataFrame(), {}

    summary = pd.DataFrame(rows).sort_values("Start Date").reset_index(drop=True)

    # Inter-field gap stats
    gap_distances_km = []
    gap_times_min = []

    for i in range(len(summary)):
        if i == len(summary) - 1:
            gap_distances_km.append(np.nan)
            gap_times_min.append(np.nan)
            continue

        fid_now = int(summary.loc[i, "Field ID"])
        fid_next = int(summary.loc[i + 1, "Field ID"])

        g1 = field_geom_info[fid_now]["points"]
        g2 = field_geom_info[fid_next]["points"]

        p1 = g1[["lat", "lng"]].iloc[-1]
        p2 = g2[["lat", "lng"]].iloc[0]

        dist_m = haversine_m(p1["lat"], p1["lng"], p2["lat"], p2["lng"])
        gap_min = (
            summary.loc[i + 1, "Start Date"] - summary.loc[i, "End Date"]
        ).total_seconds() / 60.0

        gap_times_min.append(max(gap_min, 0.0))
        gap_distances_km.append(dist_m / 1000.0)

    summary["Next Field Gap Distance (km)"] = gap_distances_km
    summary["Next Field Gap Time (min)"] = gap_times_min

    return summary, field_geom_info


def build_map(df_all, summary, field_geom_info):
    """
    Create folium map with:
    - raw points
    - field path
    - coverage polygons
    - center markers
    """
    center = [df_all["lat"].mean(), df_all["lng"].mean()]
    m = folium.Map(location=center, zoom_start=16, control_scale=True)
    plugins.Fullscreen().add_to(m)

    # Raw points
    raw_layer = folium.FeatureGroup(name="Raw GPS Points", show=False)
    for _, row in df_all.iterrows():
        color = "blue" if row["field_id"] != -1 else "red"
        folium.CircleMarker(
            location=(row["lat"], row["lng"]),
            radius=2,
            color=color,
            fill=True,
            fill_opacity=0.7,
            opacity=0.8
        ).add_to(raw_layer)
    raw_layer.add_to(m)

    # Fields
    palette = [
        "blue", "green", "purple", "orange", "darkred",
        "cadetblue", "darkgreen", "darkblue", "pink", "gray"
    ]

    for i, row in summary.iterrows():
        fid = int(row["Field ID"])
        color = palette[i % len(palette)]

        info = field_geom_info[fid]
        g = info["points"].sort_values("Timestamp").reset_index(drop=True)

        # Path line
        folium.PolyLine(
            locations=g[["lat", "lng"]].values.tolist(),
            color=color,
            weight=3,
            opacity=0.9,
            tooltip=(
                f"Field {fid} | "
                f"Area: {row['Area (Gunthas)']:.2f} gunthas | "
                f"Time: {row['Work Time (min)']:.1f} min"
            )
        ).add_to(m)

        # Coverage polygon
        gj = geom_to_folium_geojson(info["geom"], info["lat0"], info["lon0"])
        if gj is not None:
            folium.GeoJson(
                gj,
                style_function=lambda _, c=color: {
                    "fillColor": c,
                    "color": c,
                    "weight": 2,
                    "fillOpacity": 0.18
                },
                tooltip=(
                    f"Field {fid} | "
                    f"Area: {row['Area (Gunthas)']:.2f} gunthas"
                )
            ).add_to(m)

        # Center marker
        folium.Marker(
            location=(row["Center Lat"], row["Center Lng"]),
            tooltip=(
                f"Field {fid}\n"
                f"Area: {row['Area (Gunthas)']:.2f} gunthas\n"
                f"Points: {int(row['Point Count'])}"
            ),
            icon=folium.Icon(color="green", icon="info-sign")
        ).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def load_input_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload CSV or XLSX.")


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Field Coverage Area Calculator", layout="wide")
st.title("Field Coverage Area Calculator from GPS")

st.markdown(
    """
This version estimates **covered area** from the machine path using **implement width**,
instead of using a convex hull.
"""
)

with st.sidebar:
    st.header("Settings")

    implement_width_m = st.number_input(
        "Implement / Working Width (meters)",
        min_value=0.2,
        max_value=10.0,
        value=1.20,
        step=0.05
    )

    dbscan_eps_m = st.number_input(
        "DBSCAN eps (meters)",
        min_value=1.0,
        max_value=50.0,
        value=10.0,
        step=1.0
    )

    min_samples = st.number_input(
        "DBSCAN min samples",
        min_value=2,
        max_value=100,
        value=8,
        step=1
    )

    time_gap_minutes = st.number_input(
        "New session gap (minutes)",
        min_value=1,
        max_value=240,
        value=10,
        step=1
    )

    max_speed_kmph = st.number_input(
        "Max allowed speed for GPS outlier removal (kmph)",
        min_value=1.0,
        max_value=60.0,
        value=15.0,
        step=1.0
    )

    min_field_gunthas = st.number_input(
        "Minimum field area to keep (Gunthas)",
        min_value=0.0,
        max_value=1000.0,
        value=5.0,
        step=0.5
    )

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    try:
        raw_df = load_input_file(uploaded_file)

        df, synthetic_time, lat_col, lon_col, time_col = normalize_columns(raw_df)
        df = remove_basic_gps_noise(df)
        df = remove_speed_outliers(df, max_speed_kmph=max_speed_kmph)

        if df.empty:
            st.error("No valid GPS data left after cleaning.")
            st.stop()

        df = assign_sessions(df, time_gap_minutes=time_gap_minutes)
        df = cluster_fields(df, dbscan_eps_m=dbscan_eps_m, min_samples=int(min_samples))

        clustered = df[df["field_id"] != -1].copy()
        if clustered.empty:
            st.error("No valid field clusters detected. Try increasing DBSCAN eps or lowering min_samples.")
            st.stop()

        summary, field_geom_info = build_field_summary(
            clustered,
            implement_width_m=implement_width_m,
            min_field_gunthas=min_field_gunthas
        )

        if summary.empty:
            st.error("No fields remain after minimum area filter. Reduce the minimum field area or adjust clustering.")
            st.stop()

        map_obj = build_map(df, summary, field_geom_info)

        total_area_gunthas = summary["Area (Gunthas)"].sum()
        total_area_m2 = summary["Area (m²)"].sum()
        total_work_time = summary["Work Time (min)"].sum()
        total_gap_dist = np.nansum(summary["Next Field Gap Distance (km)"])
        total_gap_time = np.nansum(summary["Next Field Gap Time (min)"])

        # Header info
        cinfo1, cinfo2, cinfo3, cinfo4 = st.columns(4)
        cinfo1.info(f"Detected lat column: **{lat_col}**")
        cinfo2.info(f"Detected lon column: **{lon_col}**")
        cinfo3.info(f"Detected time column: **{time_col if time_col else 'Synthetic'}**")
        cinfo4.info(f"Rows after cleaning: **{len(df)}**")

        if synthetic_time:
            st.warning(
                "No timestamp column was found, so synthetic timestamps were created. "
                "Area is still usable, but time-based metrics are artificial."
            )

        st.subheader("Map")
        folium_static(map_obj, width=1500, height=700)

        st.subheader("Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Area", f"{total_area_gunthas:.2f} Gunthas")
        c2.metric("Total Area", f"{total_area_m2:.2f} m²")
        c3.metric("Total Work Time", f"{total_work_time:.2f} min")
        c4.metric("Inter-field Gap Distance", f"{total_gap_dist:.2f} km")
        c5.metric("Inter-field Gap Time", f"{total_gap_time:.2f} min")

        display_df = summary.copy()
        for col in ["Area (Gunthas)", "Area (m²)", "Work Time (min)", "Next Field Gap Distance (km)", "Next Field Gap Time (min)"]:
            display_df[col] = display_df[col].round(2)

        st.dataframe(display_df, use_container_width=True)

        csv_bytes = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Summary CSV",
            data=csv_bytes,
            file_name="field_coverage_summary.csv",
            mime="text/csv"
        )

        with st.expander("Show cleaned input data"):
            st.dataframe(df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
