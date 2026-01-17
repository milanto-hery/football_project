import streamlit as st
from roboflow import Roboflow
from utils import draw_bboxes, process_video
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt
import tempfile
import os

# --- Page setup ---
st.set_page_config(page_title="⚽ Ball Possession Tracker", layout="wide")
st.title("⚽ Football Ball Possession Tracker")
st.markdown("Upload an **image or video** to detect which player possesses the ball.")

# Optional CSS
css_path = os.path.join("style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- Sidebar settings ---
st.sidebar.header("⚙️ Detection Settings")
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)

st.sidebar.header("Controls")
start_button = st.sidebar.button("▶️ Start Prediction")
stop_button  = st.sidebar.button("❌ Stop Prediction")

# Initialize running flag
if "running" not in st.session_state:
    st.session_state.running = False
if start_button:
    st.session_state.running = True
if stop_button:
    st.session_state.running = False

# --- Load Roboflow Model ---
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

# --- File uploader ---
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","jpeg","png","mp4","avi"])
image_placeholder = st.empty()
results_sidebar = st.sidebar.empty()

# --- Main prediction logic ---
if uploaded_file and model:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        image_placeholder.image(image, width=800)

        try:
            results = model.predict(uploaded_file, confidence=threshold).json()
            predictions = [p for p in results.get("predictions", []) if p["confidence"] >= threshold]
            output_img = draw_bboxes(image_np, predictions)
            image_placeholder.image(output_img, width=800)
        except Exception as e:
            results_sidebar.error(f"❌ Prediction failed: {e}")

    elif uploaded_file.type.startswith("video"):
        st.info("Video prediction requires pressing 'Start Prediction' in the sidebar.")
        if st.session_state.running:
            st_progress = st.progress(0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_vid:
                tmp_vid.write(uploaded_file.read())
                tmp_vid_path = tmp_vid.name

            try:
                processed_path, timeline = process_video(
                    tmp_vid_path, model, threshold, st_progress, lambda: st.session_state.running
                )
                st.video(processed_path)

                # Timeline chart
                df = pd.DataFrame(timeline)
                chart = alt.Chart(df).mark_bar().encode(
                    x='frame:Q',
                    y='has_ball:Q'
                ).properties(height=200)
                st.altair_chart(chart, use_container_width=True)

                st.success("✅ Video processed successfully!")
            except Exception as e:
                st.error(f"❌ Video prediction failed: {e}")
        else:
            st.warning("Press ▶️ Start Prediction to process video.")
else:
    if not model:
        st.warning("Model not loaded. Cannot perform prediction.")
