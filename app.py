import streamlit as st
from roboflow import Roboflow
import cv2
import numpy as np
from PIL import Image
import tempfile
import pandas as pd
import altair as alt
from collections import Counter
from utils import draw_bboxes, process_video  # Your helper functions
import os

# --- Page setup ---
st.set_page_config(page_title="⚽ Football Ball Tracker", layout="wide")
st.title("⚽ Football Ball Possession Tracker")
st.markdown("Upload an **image or video** of a match to detect which player possesses the ball.")

# --- Optional CSS ---
css_path = os.path.join("assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar controls ---
st.sidebar.header("Detection Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
start = st.sidebar.button("▶️ Start Prediction")
stop  = st.sidebar.button("❌ Stop Prediction")

# Initialize running flag for video
if "running" not in st.session_state:
    st.session_state.running = False
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# --- Robust Roboflow model loader ---
@st.cache_resource
def load_model():
    try:
        api_key = st.secrets.get("ROBOFLOW_API_KEY")
        workspace = st.secrets.get("WORKSPACE")
        project_name = st.secrets.get("PROJECT")
        version_number = st.secrets.get("VERSION")

        if not all([api_key, workspace, project_name, version_number]):
            st.error("❌ Missing Roboflow secrets!")
            return None

        rf = Roboflow(api_key=api_key)
        ws = rf.workspace(workspace)
        proj = ws.project(project_name)
        model = proj.version(version_number).model

        if model is None:
            st.error(f"❌ Model version {version_number} not ready. Check Roboflow dashboard.")
            return None

        st.success(f"✅ Roboflow model loaded: {project_name} v{version_number}")
        return model

    except Exception as e:
        st.error(f"❌ Failed to load Roboflow model: {e}")
        return None

model = load_model()

# --- File upload ---
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","jpeg","png","mp4","avi"])
image_placeholder = st.empty()
results_sidebar = st.sidebar.empty()

# --- Main prediction logic ---
if uploaded_file is not None:
    if model is None:
        st.warning("Model not loaded. Cannot perform prediction.")
    else:
        # --- IMAGE ---
        if uploaded_file.type.startswith("image"):
            image = Image.open(uploaded_file).convert("RGB")
            image_placeholder.image(image, caption="Uploaded Image", use_column_width=True)

            # Save temp file for Roboflow API
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                image.save(tmp.name)
                temp_path = tmp.name

            col1, col2 = st.columns([1,1])
            with col1:
                predict_button = st.button("▶️ Predict")
            with col2:
                cancel_button = st.button("❌ Cancel")

            if cancel_button:
                results_sidebar.info("Prediction cancelled.")

            if predict_button:
                with st.spinner("🔍 Detecting ball possession..."):
                    try:
                        # Roboflow prediction
                        preds = model.predict(temp_path, confidence=confidence_threshold).json()
                        detections = preds.get("predictions", [])

                        image_cv = np.array(image)

                        if len(detections) == 0:
                            results_sidebar.warning("No players detected with the ball.")
                        else:
                            # Draw bounding boxes
                            player_list = []
                            for det in detections:
                                cls = det["class"]
                                player_list.append(cls)
                                x, y, w, h = det["x"], det["y"], det["width"], det["height"]
                                conf_det = det["confidence"]
                                x1, y1 = int(x - w / 2), int(y - h / 2)
                                x2, y2 = int(x + w / 2), int(y + h / 2)
                                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (46, 134, 193), 2)
                                cv2.putText(image_cv, f"{cls} ({conf_det:.2f})", (x1, y1 - 10),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (46, 134, 193), 2)

                            image_placeholder.image(image_cv, caption="Detection Result", use_column_width=True)

                            # Sidebar summary
                            player_count = Counter(player_list)
                            results_sidebar.markdown("### 🎯 Detection Summary")
                            results_sidebar.markdown(f"✅ Total players detected: **{len(detections)}**")
                            bullet_points = "\n".join([f"- {cls}: {count}" for cls, count in player_count.items()])
                            results_sidebar.markdown(bullet_points)

                    except Exception as e:
                        results_sidebar.error(f"Prediction failed: {e}")

        # --- VIDEO ---
        elif uploaded_file.type.startswith("video"):
            st.info("Video prediction requires pressing 'Start Prediction' in the sidebar.")
            if st.session_state.running:
                st_progress = st.progress(0)

                # Save video temporarily
                with open("temp_video.mp4", "wb") as f:
                    f.write(uploaded_file.read())

                def stop_flag():
                    return st.session_state.running

                try:
                    processed_video_path, timeline_data = process_video(
                        "temp_video.mp4", model, confidence_threshold, st_progress, stop_flag
                    )
                    st.video(processed_video_path)

                    # Timeline chart
                    st.subheader("Ball-in-Play Timeline")
                    df = pd.DataFrame(timeline_data)
                    chart = alt.Chart(df).mark_bar().encode(
                        x='frame:Q',
                        y='has_ball:Q',
                        tooltip=['frame','has_ball']
                    ).properties(height=200)
                    st.altair_chart(chart, use_container_width=True)

                    st.success("Video processed successfully!")
                except Exception as e:
                    st.error(f"Video prediction failed: {e}")
            else:
                st.info("Press 'Start Prediction' in the sidebar to run video detection.")

# --- Footer ---
st.markdown("<div class='footer'>Developed by Milanto — Powered by Roboflow × Streamlit</div>", unsafe_allow_html=True)
