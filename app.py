import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
from gtts import gTTS
import os
import tempfile

st.set_page_config(page_title="Food Analysis App", layout="centered")
st.title("🍽️ Food Analysis App")
st.markdown("Instant meal scanning and health insights using AI")

# Load YOLOv5 from GitHub
with st.spinner("Loading YOLOv5 model..."):
    model = YOLO('yolov5s.pt')

# Mock nutritional database
mock_nutrition_data = {
    "pizza": {"calories": 285, "protein": 12, "fat": 10},
    "broccoli": {"calories": 55, "protein": 4, "fat": 0.5},
    "apple": {"calories": 95, "protein": 0.5, "fat": 0.3},
    "sandwich": {"calories": 250, "protein": 10, "fat": 8},
    "banana": {"calories": 105, "protein": 1.3, "fat": 0.4},
}

# Input selection
mode = st.radio("Choose input method:", ["📷 Webcam", "📤 Upload Image"])
image_input = None

# Webcam Mode
if mode == "📷 Webcam":
    st.info("Click the button below to take a photo with your webcam.")
    capture = st.button("📸 Take a Photo")

    if capture:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Could not open webcam.")
        else:
            for _ in range(10):  # warm-up
                ret, frame = cap.read()
            ret, frame = cap.read()
            cap.release()

            if not ret or frame is None:
                st.error("❌ Failed to capture image.")
            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.session_state["captured_image"] = rgb_frame
                st.image(rgb_frame, caption="Captured Meal", use_column_width=True)

    if "captured_image" in st.session_state:
        image_input = st.session_state["captured_image"]

# Upload Image Mode
elif mode == "📤 Upload Image":
    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_input = Image.open(uploaded_file)
        st.image(image_input, caption="Uploaded Meal", use_column_width=True)

# Analyze button
if image_input is not None and st.button("🔍 Analyze Meal"):
    with st.spinner("Analyzing..."):
        results = model(image_input)
        detected_items = results.pandas().xyxy[0]["name"].tolist()
        st.success("Detected items: " + ", ".join(detected_items))

        # Render results
        results.render()
        result_image = Image.fromarray(results.ims[0])
        st.image(result_image, caption="Detected Objects", use_column_width=True)

        # Filter + summarize
        valid_food_labels = list(mock_nutrition_data.keys())
        filtered = [item for item in detected_items if item.lower() in valid_food_labels]
        unique_items = list(set(filtered))

        summary = ""
        for item in unique_items:
            info = mock_nutrition_data[item.lower()]
            summary += f"{item.title()}: {info['calories']} kcal, {info['protein']}g protein, {info['fat']}g fat\n"

        if summary.strip():
            st.text_area("Nutrition Summary", summary, height=150)

            # TTS
            tts = gTTS(text=summary, lang="en")
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
            tts.save(temp_audio.name)
            st.audio(temp_audio.name)
        else:
            st.warning("No recognizable food items found.")
