import streamlit as st
from roboflow import Roboflow
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt
from utils import draw_bboxes, process_video

st.set_page_config(page_title="⚽ Ball Possession Tracker", layout="wide")

# Optional CSS for professional look
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("⚽ Football Ball Possession Tracker")
st.markdown("Upload an **image or video** to detect which player possesses the ball.")

# Sidebar controls
st.sidebar.header("Controls")
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)
start = st.sidebar.button("Start Prediction")
stop  = st.sidebar.button("Stop Prediction")

# Initialize running flag
if "running" not in st.session_state:
    st.session_state.running = False
if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

# Load Roboflow model (cached)
@st.cache_resource
def load_model():
    api_key = st.secrets["ROBOFLOW_API_KEY"]
    workspace = st.secrets["WORKSPACE"]
    project_name = st.secrets["PROJECT"]
    version_number = st.secrets["VERSION"]

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    model = project.version(version_number).model
    return model

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","jpeg","png","mp4","avi"])
image_placeholder = st.empty()

if uploaded_file is not None and st.session_state.running:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        results = model.predict(image_np)
        predictions = [p for p in results.get("predictions", []) if p['confidence'] >= threshold]
        output_img = draw_bboxes(image_np, predictions)
        image_placeholder.image(output_img, use_column_width=True)

    elif uploaded_file.type.startswith("video"):
        st.info("Processing video frames...")
        st_progress = st.progress(0)

        # Save video temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        def stop_flag():
            return st.session_state.running

        processed_video_path, timeline_data = process_video(
            "temp_video.mp4", model, threshold, st_progress, stop_flag
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
