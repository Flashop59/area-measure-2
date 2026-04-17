import streamlit as st
import pandas as pd
import numpy as np
import folium

from math import radians, cos, sin, asin, sqrt
from sklearn.neighbors import NearestNeighbors
from streamlit_folium import folium_static
from folium import plugins

# =========================================================
# CONSTANTS
# =========================================================
GUNTHA_M2 = 101.17
FT_TO_M = 0.3048

# =========================================================
# GEO HELPERS
# =========================================================
def haversine_m(lat1, lon1, lat2, lon2):
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


# =========================================================
# INPUT / CLEANING
# =========================================================
def load_input_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file. Upload CSV or XLSX.")


def normalize_columns(df):
    df = df.copy()
    original_cols = list(df.columns)
    df.columns = [str(c).strip().lower() for c in df.columns]

    lat_candidates = ["lat", "latitude"]
    lon_candidates = ["lng", "lon", "longitude", "long"]
    time_candidates = ["timestamp", "time", "date", "datetime", "created_at"]

    lat_col = next((c for c in lat_candidates if c in df.columns), None)
    lon_col = next((c for c in lon_candidates if c in df.columns), None)
    time_col = next((c for c in time_candidates if c in df.columns), None)

    if lat_col is None or lon_col is None:
        raise ValueError("Could not detect lat/lng columns. Use lat/lng or latitude/longitude.")

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

    out = out.dropna(subset=["lat", "lng", "Timestamp"]).copy()
    out = out[
        out["lat"].between(-90, 90) &
        out["lng"].between(-180, 180)
    ].copy()

    out = out.sort_values("Timestamp").reset_index(drop=True)

    return out, {
        "original_columns": original_cols,
        "lat_col": lat_col,
        "lon_col": lon_col,
        "time_col": time_col,
        "synthetic_time": synthetic_time
    }


def remove_duplicate_points(df):
    df = df.copy()
    df = df.drop_duplicates(subset=["lat", "lng", "Timestamp"]).reset_index(drop=True)
    return df


def remove_far_jump_points(df, max_jump_m=20.0):
    """
    Remove points that jump too far from the previous kept point.
    Rule:
    - first point is kept
    - if current point is more than max_jump_m away from previous kept point,
      current point is removed
    """
    df = df.copy().sort_values("Timestamp").reset_index(drop=True)

    if len(df) <= 1:
        return df

    keep_indices = [0]
    last_kept_idx = 0

    for i in range(1, len(df)):
        dist_m = haversine_m(
            df.loc[last_kept_idx, "lat"], df.loc[last_kept_idx, "lng"],
            df.loc[i, "lat"], df.loc[i, "lng"]
        )

        if dist_m <= max_jump_m:
            keep_indices.append(i)
            last_kept_idx = i

    return df.loc[keep_indices].reset_index(drop=True)


# =========================================================
# LABELING: MOVEMENT VS OPERATION
# =========================================================
def count_neighbors_within_radius(xy, radius_m):
    if len(xy) == 0:
        return np.array([], dtype=int)

    nn = NearestNeighbors(radius=radius_m, metric="euclidean")
    nn.fit(xy)
    neighbors = nn.radius_neighbors(xy, return_distance=False)

    counts = np.array([max(len(n) - 1, 0) for n in neighbors], dtype=int)
    return counts


def label_operation_by_density(df, radius_m=8.0, min_neighbors=8):
    df = df.copy()
    xy, _, _ = latlon_to_local_xy(df[["lat", "lng"]].to_numpy())

    neighbor_count = count_neighbors_within_radius(xy, radius_m=radius_m)
    df["neighbor_count"] = neighbor_count

    df["raw_point_type"] = np.where(
        df["neighbor_count"] >= min_neighbors,
        "operation",
        "movement"
    )

    return df


def make_segments_from_labels(labels):
    if len(labels) == 0:
        return []

    segments = []
    start = 0
    current = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append({
                "start_idx": start,
                "end_idx": i - 1,
                "label": current
            })
            start = i
            current = labels[i]

    segments.append({
        "start_idx": start,
        "end_idx": len(labels) - 1,
        "label": current
    })
    return segments


def segment_point_distance(df, start_idx, end_idx):
    if end_idx <= start_idx:
        return 0.0

    dist = 0.0
    for i in range(start_idx + 1, end_idx + 1):
        dist += haversine_m(
            df.loc[i - 1, "lat"], df.loc[i - 1, "lng"],
            df.loc[i, "lat"], df.loc[i, "lng"]
        )
    return dist


def smooth_point_types(
    df,
    min_operation_run_points=15,
    min_movement_run_points=5,
    min_movement_break_distance_m=20.0
):
    df = df.copy()
    labels = df["raw_point_type"].tolist()

    changed = True
    while changed:
        changed = False
        segments = make_segments_from_labels(labels)

        for seg in segments:
            seg_len = seg["end_idx"] - seg["start_idx"] + 1

            if seg["label"] == "operation" and seg_len < min_operation_run_points:
                for i in range(seg["start_idx"], seg["end_idx"] + 1):
                    labels[i] = "movement"
                changed = True

            elif seg["label"] == "movement" and seg_len < min_movement_run_points:
                for i in range(seg["start_idx"], seg["end_idx"] + 1):
                    labels[i] = "operation"
                changed = True

        if changed:
            continue

        segments = make_segments_from_labels(labels)
        for j in range(1, len(segments) - 1):
            prev_seg = segments[j - 1]
            mid_seg = segments[j]
            next_seg = segments[j + 1]

            if (
                prev_seg["label"] == "operation"
                and mid_seg["label"] == "movement"
                and next_seg["label"] == "operation"
            ):
                move_dist = segment_point_distance(df, mid_seg["start_idx"], mid_seg["end_idx"])
                if move_dist < min_movement_break_distance_m:
                    for i in range(mid_seg["start_idx"], mid_seg["end_idx"] + 1):
                        labels[i] = "operation"
                    changed = True
                    break

    df["point_type"] = labels
    return df


def assign_field_ids(df, min_operation_segment_points=20):
    df = df.copy()
    df["field_id"] = np.nan

    labels = df["point_type"].tolist()
    segments = make_segments_from_labels(labels)

    next_fid = 1
    for seg in segments:
        if seg["label"] != "operation":
            continue

        seg_len = seg["end_idx"] - seg["start_idx"] + 1
        if seg_len < min_operation_segment_points:
            continue

        df.loc[seg["start_idx"]:seg["end_idx"], "field_id"] = next_fid
        next_fid += 1

    return df


# =========================================================
# FIELD METRICS
# =========================================================
def compute_operation_path_distance_m(g):
    if len(g) < 2:
        return 0.0

    dist = 0.0
    g = g.sort_values("Timestamp").reset_index(drop=True)

    for i in range(1, len(g)):
        dist += haversine_m(
            g.loc[i - 1, "lat"], g.loc[i - 1, "lng"],
            g.loc[i, "lat"], g.loc[i, "lng"]
        )

    return dist


def compute_summary(df, working_width_m):
    op = df[df["field_id"].notna()].copy()
    if op.empty:
        return pd.DataFrame()

    rows = []
    for fid, g in op.groupby("field_id", sort=True):
        g = g.sort_values("Timestamp").reset_index(drop=True)

        run_distance_m = compute_operation_path_distance_m(g)
        area_m2 = run_distance_m * working_width_m
        area_gunthas = area_m2 / GUNTHA_M2

        start_dt = g["Timestamp"].min()
        end_dt = g["Timestamp"].max()
        duration_min = (end_dt - start_dt).total_seconds() / 60.0

        center_lat = g["lat"].median()
        center_lng = g["lng"].median()

        rows.append({
            "Field ID": int(fid),
            "Operation Points": int(len(g)),
            "Run Distance (m)": run_distance_m,
            "Working Width (m)": working_width_m,
            "Area (m²)": area_m2,
            "Area (Gunthas)": area_gunthas,
            "Start Date": start_dt,
            "End Date": end_dt,
            "Operation Time (min)": duration_min,
            "Center Lat": center_lat,
            "Center Lng": center_lng
        })

    summary = pd.DataFrame(rows).sort_values("Start Date").reset_index(drop=True)

    movement_rows = []
    for i in range(len(summary)):
        if i == len(summary) - 1:
            movement_rows.append((np.nan, np.nan))
            continue

        fid_now = summary.loc[i, "Field ID"]
        fid_next = summary.loc[i + 1, "Field ID"]

        g_now = op[op["field_id"] == fid_now].sort_values("Timestamp").reset_index(drop=True)
        g_next = op[op["field_id"] == fid_next].sort_values("Timestamp").reset_index(drop=True)

        d = haversine_m(
            g_now.loc[len(g_now) - 1, "lat"], g_now.loc[len(g_now) - 1, "lng"],
            g_next.loc[0, "lat"], g_next.loc[0, "lng"]
        ) / 1000.0

        t = (
            summary.loc[i + 1, "Start Date"] - summary.loc[i, "End Date"]
        ).total_seconds() / 60.0
        movement_rows.append((d, max(t, 0.0)))

    summary["Gap to Next Field (km)"] = [x[0] for x in movement_rows]
    summary["Gap to Next Field (min)"] = [x[1] for x in movement_rows]

    return summary


# =========================================================
# MAP
# =========================================================
def add_esri_satellite(base_map):
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True
    ).add_to(base_map)


def build_map(df, summary, show_raw_track=True, show_movement=True, show_operation=True, show_field_lines=True, show_centers=True):
    center = [df["lat"].mean(), df["lng"].mean()]
    m = folium.Map(location=center, zoom_start=17, control_scale=True, tiles=None)

    add_esri_satellite(m)
    plugins.Fullscreen().add_to(m)

    if show_raw_track:
        fg_track = folium.FeatureGroup(name="Raw Track", show=True)
        folium.PolyLine(
            locations=df[["lat", "lng"]].values.tolist(),
            color="white",
            weight=2,
            opacity=0.7
        ).add_to(fg_track)
        fg_track.add_to(m)

    if show_movement:
        fg_move = folium.FeatureGroup(name="Movement Points", show=True)
        move = df[df["point_type"] == "movement"]
        for _, row in move.iterrows():
            folium.CircleMarker(
                location=(row["lat"], row["lng"]),
                radius=3,
                color="red",
                fill=True,
                fill_color="red",
                fill_opacity=0.85,
                opacity=0.9,
                tooltip=f"Movement | neighbors={int(row['neighbor_count'])}"
            ).add_to(fg_move)
        fg_move.add_to(m)

    if show_operation:
        fg_op = folium.FeatureGroup(name="Operation Points", show=True)
        op = df[df["point_type"] == "operation"]
        for _, row in op.iterrows():
            fid_txt = f" | field={int(row['field_id'])}" if pd.notna(row["field_id"]) else ""
            folium.CircleMarker(
                location=(row["lat"], row["lng"]),
                radius=3,
                color="lime",
                fill=True,
                fill_color="lime",
                fill_opacity=0.85,
                opacity=0.9,
                tooltip=f"Operation | neighbors={int(row['neighbor_count'])}{fid_txt}"
            ).add_to(fg_op)
        fg_op.add_to(m)

    if show_field_lines and not summary.empty:
        fg_fields = folium.FeatureGroup(name="Field Operation Runs", show=True)
        palette = ["cyan", "yellow", "magenta", "orange", "deepskyblue", "lawngreen", "violet", "gold"]

        for i, row in summary.iterrows():
            fid = row["Field ID"]
            color = palette[i % len(palette)]
            g = df[df["field_id"] == fid].sort_values("Timestamp")

            folium.PolyLine(
                locations=g[["lat", "lng"]].values.tolist(),
                color=color,
                weight=4,
                opacity=0.9,
                tooltip=(
                    f"Field {int(fid)} | "
                    f"Area={row['Area (Gunthas)']:.2f} gunthas | "
                    f"Run={row['Run Distance (m)']:.1f} m"
                )
            ).add_to(fg_fields)

        fg_fields.add_to(m)

    if show_centers and not summary.empty:
        fg_centers = folium.FeatureGroup(name="Field Centers", show=True)
        for _, row in summary.iterrows():
            folium.Marker(
                location=(row["Center Lat"], row["Center Lng"]),
                tooltip=(
                    f"Field {int(row['Field ID'])}\n"
                    f"Area: {row['Area (Gunthas)']:.2f} gunthas\n"
                    f"Run Distance: {row['Run Distance (m)']:.1f} m\n"
                    f"Operation Points: {int(row['Operation Points'])}"
                ),
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(fg_centers)
        fg_centers.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# =========================================================
# MAIN PIPELINE
# =========================================================
def process_pipeline(
    raw_df,
    working_width_m,
    density_radius_m,
    min_neighbors,
    min_operation_run_points,
    min_movement_run_points,
    min_movement_break_distance_m,
    min_operation_segment_points,
    max_jump_m
):
    df, meta = normalize_columns(raw_df)
    df = remove_duplicate_points(df)
    df = remove_far_jump_points(df, max_jump_m=max_jump_m)

    if df.empty:
        raise ValueError("No valid data left after cleaning.")

    df = label_operation_by_density(
        df,
        radius_m=density_radius_m,
        min_neighbors=min_neighbors
    )

    df = smooth_point_types(
        df,
        min_operation_run_points=min_operation_run_points,
        min_movement_run_points=min_movement_run_points,
        min_movement_break_distance_m=min_movement_break_distance_m
    )

    df = assign_field_ids(
        df,
        min_operation_segment_points=min_operation_segment_points
    )

    summary = compute_summary(df, working_width_m=working_width_m)

    return df, summary, meta


# =========================================================
# STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Guntha Area Calculator - Movement vs Operation", layout="wide")
st.title("GPS Guntha Area Calculator")
st.caption("Logic: classify movement vs operation points first, then compute area as operation run distance × working width.")

with st.sidebar:
    st.header("Core Settings")

    width_mode = st.radio("Working Width Input", ["Feet", "Meters"], index=0)
    if width_mode == "Feet":
        working_width_ft = st.number_input("Working Width (ft)", min_value=0.5, max_value=30.0, value=4.0, step=0.1)
        working_width_m = working_width_ft * FT_TO_M
    else:
        working_width_m = st.number_input("Working Width (m)", min_value=0.1, max_value=20.0, value=1.2192, step=0.05)

    density_radius_m = st.number_input("Nearby Radius Y (meters)", min_value=1.0, max_value=50.0, value=8.0, step=1.0)
    min_neighbors = st.number_input("Min Nearby Points N", min_value=1, max_value=100, value=8, step=1)

    st.header("Smoothing / Sequence Rules")
    min_operation_run_points = st.number_input("Min operation run points", min_value=1, max_value=1000, value=15, step=1)
    min_movement_run_points = st.number_input("Min movement run points", min_value=1, max_value=1000, value=5, step=1)
    min_movement_break_distance_m = st.number_input("Min movement distance to split fields (m)", min_value=1.0, max_value=1000.0, value=20.0, step=1.0)
    min_operation_segment_points = st.number_input("Min final field points", min_value=1, max_value=5000, value=20, step=1)

    st.header("Jump Outlier Removal")
    max_jump_m = st.number_input(
        "Remove point if jump from previous kept point is more than (m)",
        min_value=1.0,
        max_value=200.0,
        value=20.0,
        step=1.0
    )

    st.header("Map Overlays")
    show_raw_track = st.checkbox("Show raw track", value=True)
    show_movement = st.checkbox("Show movement points", value=True)
    show_operation = st.checkbox("Show operation points", value=True)
    show_field_lines = st.checkbox("Show field operation runs", value=True)
    show_centers = st.checkbox("Show field centers", value=True)

uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

if uploaded_file:
    try:
        raw_df = load_input_file(uploaded_file)

        df, summary, meta = process_pipeline(
            raw_df=raw_df,
            working_width_m=working_width_m,
            density_radius_m=density_radius_m,
            min_neighbors=int(min_neighbors),
            min_operation_run_points=int(min_operation_run_points),
            min_movement_run_points=int(min_movement_run_points),
            min_movement_break_distance_m=min_movement_break_distance_m,
            min_operation_segment_points=int(min_operation_segment_points),
            max_jump_m=max_jump_m
        )

        c1, c2, c3, c4 = st.columns(4)
        c1.info(f"Lat col: **{meta['lat_col']}**")
        c2.info(f"Lng col: **{meta['lon_col']}**")
        c3.info(f"Time col: **{meta['time_col'] if meta['time_col'] else 'Synthetic'}**")
        c4.info(f"Rows used: **{len(df)}**")

        if meta["synthetic_time"]:
            st.warning("No time column found. Synthetic timestamps were created. Area logic is still okay, but time metrics are artificial.")

        map_obj = build_map(
            df=df,
            summary=summary,
            show_raw_track=show_raw_track,
            show_movement=show_movement,
            show_operation=show_operation,
            show_field_lines=show_field_lines,
            show_centers=show_centers
        )

        st.subheader("Satellite Map")
        folium_static(map_obj, width=1500, height=750)

        movement_count = int((df["point_type"] == "movement").sum())
        operation_count = int((df["point_type"] == "operation").sum())
        total_area_guntha = summary["Area (Gunthas)"].sum() if not summary.empty else 0.0
        total_area_m2 = summary["Area (m²)"].sum() if not summary.empty else 0.0
        total_run_m = summary["Run Distance (m)"].sum() if not summary.empty else 0.0

        st.subheader("Totals")
        t1, t2, t3, t4, t5 = st.columns(5)
        t1.metric("Movement Points", movement_count)
        t2.metric("Operation Points", operation_count)
        t3.metric("Total Run Distance", f"{total_run_m:.2f} m")
        t4.metric("Total Area", f"{total_area_m2:.2f} m²")
        t5.metric("Total Area", f"{total_area_guntha:.2f} Gunthas")

        st.subheader("Field Summary")
        if summary.empty:
            st.warning("No final operation field clusters detected. Tune radius / min neighbors / smoothing values.")
        else:
            display_df = summary.copy()
            numeric_cols = [
                "Run Distance (m)", "Working Width (m)", "Area (m²)",
                "Area (Gunthas)", "Operation Time (min)",
                "Gap to Next Field (km)", "Gap to Next Field (min)"
            ]
            for col in numeric_cols:
                if col in display_df.columns:
                    display_df[col] = display_df[col].round(2)

            st.dataframe(display_df, use_container_width=True)

            st.download_button(
                "Download Summary CSV",
                data=display_df.to_csv(index=False).encode("utf-8"),
                file_name="guntha_area_summary.csv",
                mime="text/csv"
            )

        with st.expander("Show classified point data"):
            view_df = df.copy()
            if "field_id" in view_df.columns:
                view_df["field_id"] = view_df["field_id"].astype("Int64")
            st.dataframe(view_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
