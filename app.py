import streamlit as st
from roboflow import Roboflow
from PIL import Image
import numpy as np
import pandas as pd
import altair as alt
from utils import draw_bboxes, process_video

# Page Config
st.set_page_config(page_title="⚽ Ball Possession Tracker", layout="wide")

# Custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("⚽ Football Ball Possession Tracker")
st.markdown("Upload an **image or video** to detect players who possess the ball in real time.")

# Load Roboflow Model (cached)
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

# File Upload
uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg","jpeg","png","mp4","avi"])
image_placeholder = st.empty()

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)

        results = model.predict(image_np).json()
        predictions = results.get("predictions", [])

        output_img = draw_bboxes(image_np, predictions)
        image_placeholder.image(output_img, use_column_width=True)

    elif uploaded_file.type.startswith("video"):
        st.info("Processing video frames...")
        st_progress = st.progress(0)

        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())

        processed_video_path, timeline_data = process_video("temp_video.mp4", model, st_progress)
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
