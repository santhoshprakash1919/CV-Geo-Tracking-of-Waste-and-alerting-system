"""
Streamlit front-end for the Computer Vision Based Geo Tracking of Waste
and Alerting System.

This reuses your existing ai_model/predict.py and backend/database.py
unchanged -- only the UI layer (previously Flask + Leaflet HTML) is
replaced with native Streamlit widgets + st.map / pydeck.

Run with:
    streamlit run streamlit_app.py
"""

import os
import sqlite3
import datetime

import streamlit as st
import pandas as pd
import pydeck as pdk

from ai_model.predict import predict
from backend.database import init_db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DB_PATH = os.path.join(BASE_DIR, "database", "garbage.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
init_db()

SEVERITY_COLOR = {
    "full": [193, 67, 46],       # safety red
    "partial": [217, 142, 43],   # hazard amber
    "empty": [62, 125, 82],      # clear green
}
SEVERITY_RANK = {"full": 0, "partial": 1, "empty": 2}


# --------------------------------------------------------------------------
# Database helpers (same schema as backend/database.py -- reused directly)
# --------------------------------------------------------------------------
def insert_detection(image_path, prediction, confidence, lat, lng):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO detections (image_path, prediction, confidence, latitude, longitude, timestamp) "
        "VALUES (?,?,?,?,?,?)",
        (image_path, prediction, confidence, lat, lng, str(datetime.datetime.now())),
    )
    conn.commit()
    conn.close()


def fetch_history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, prediction, confidence, latitude, longitude, timestamp "
        "FROM detections ORDER BY id DESC"
    ).fetchall()
    conn.close()
    return pd.DataFrame(
        rows,
        columns=["id", "prediction", "confidence", "latitude", "longitude", "timestamp"],
    )


def fetch_priority_order():
    df = fetch_history()
    if df.empty:
        return df
    df = df.dropna(subset=["latitude", "longitude"])
    df["rank"] = df["prediction"].map(SEVERITY_RANK).fillna(99)
    return df.sort_values("rank").drop(columns="rank")


# --------------------------------------------------------------------------
# Page config + styling (kept close to the deck/dashboard visual language)
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Garbage Watch — Bin Monitoring",
    page_icon="🗑️",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { background-color: #EFEBE3; }
    div[data-testid="stMetric"] {
        background: #FAF8F3;
        border: 1px solid #D9D2C3;
        padding: 12px;
        border-radius: 4px;
    }
    .badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 3px;
        color: white;
        font-weight: 600;
        font-size: 12.5px;
        text-transform: uppercase;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🗑️ Garbage Watch — Bin Monitoring Dashboard")
st.caption("Municipal Sanitation · Computer Vision Unit")

# --------------------------------------------------------------------------
# Layout: upload panel (left) + map (right)
# --------------------------------------------------------------------------
col_upload, col_map = st.columns([1, 1.6], gap="large")

with col_upload:
    st.subheader("Submit Photo")

    uploaded_file = st.file_uploader(
        "Choose a bin photo", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    st.markdown("**Location** (fill in manually, or leave 0.0 if unknown)")
    c1, c2 = st.columns(2)
    with c1:
        lat = st.number_input("Latitude", value=11.341, format="%.6f")
    with c2:
        lng = st.number_input("Longitude", value=78.145, format="%.6f")

    predict_clicked = st.button("Upload & Predict", type="primary", use_container_width=True)

    if predict_clicked:
        if uploaded_file is None:
            st.error("Choose an image first.")
        else:
            save_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Predicting…"):
                result = predict(save_path)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                insert_detection(save_path, result["prediction"], result["confidence"], lat, lng)

                color = {
                    "full": "#C1432E", "partial": "#D98E2B", "empty": "#3E7D52"
                }.get(result["prediction"], "#888")

                st.markdown(
                    f'<span class="badge" style="background:{color}">{result["prediction"]}</span>',
                    unsafe_allow_html=True,
                )
                st.write(f"**Confidence:** {result['confidence']*100:.1f}%")
                if "detection_confidence" in result:
                    st.write(f"**Bin detection confidence:** {result['detection_confidence']*100:.1f}%")
                if result.get("low_confidence_detection"):
                    st.warning("Low-confidence detection — verify visually.")

                st.success("Prediction saved.")
                st.rerun()

with col_map:
    st.subheader("Bin Locations")

    history_df = fetch_history()
    map_df = history_df.dropna(subset=["latitude", "longitude"]).copy()

    if map_df.empty:
        st.info("No bins logged yet — upload a photo to begin.")
    else:
        map_df["color"] = map_df["prediction"].map(SEVERITY_COLOR).apply(
            lambda c: c if isinstance(c, list) else [136, 136, 136]
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_fill_color="color",
            get_radius=100,
            radius_min_pixels=6,
            radius_max_pixels=50,
            pickable=True,
        )
        view_state = pdk.ViewState(
            latitude=map_df["latitude"].iloc[0],
            longitude=map_df["longitude"].iloc[0],
            zoom=14,
        )
        map_key = f"deck_{map_df['id'].iloc[0]}" if "id" in map_df.columns and not map_df.empty else "deck_default"
        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                map_style="road",
                tooltip={"text": "Bin #{id}: {prediction} ({confidence})"},
            ),
            use_container_width=True,
            key=map_key
        )

        legend_cols = st.columns(3)
        for col, (label, hexcolor) in zip(
            legend_cols,
            [("Full", "#C1432E"), ("Partial", "#D98E2B"), ("Empty", "#3E7D52")],
        ):
            col.markdown(
                f'<span class="badge" style="background:{hexcolor}">{label}</span>',
                unsafe_allow_html=True,
            )

# --------------------------------------------------------------------------
# Collection manifest (history table) + priority order
# --------------------------------------------------------------------------
st.markdown("---")

tab_history, tab_priority = st.tabs(["📋 Collection Manifest", "🚛 Priority Route (Full → Partial → Empty)"])

with tab_history:
    history_df = fetch_history()
    if history_df.empty:
        st.info("No bins logged yet.")
    else:
        st.dataframe(history_df, use_container_width=True, hide_index=True)

with tab_priority:
    priority_df = fetch_priority_order()
    if priority_df.empty:
        st.info("No bins with location data yet.")
    else:
        st.dataframe(
            priority_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Sanitary workers should visit bins in this order: "
            "full bins first, then partial, then empty."
        )

st.markdown("---")
st.caption("Full → Partial → Empty collection priority · Data stored locally in SQLite")
