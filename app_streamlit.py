import streamlit as st
import os
import cv2
import numpy as np
import tensorflow as tf
import tempfile
import logging
from werkzeug.utils import secure_filename

# Setup directories
UPLOAD_FOLDER = "static/uploads"
PROCESSED_FOLDER = "static/processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model
MODEL_PATH = "mobilenet_micro_expression_classifier.h5"
try:
    model = tf.h5.models.load_model(MODEL_PATH)
    logging.info("✅ Model loaded successfully.")
except Exception as e:
    logging.error(f"❌ Failed to load model: {e}")
    model = None

if model is None:
    st.error("❌ Model failed to load. Cannot proceed.")
    st.stop()

# Define emotions
emotions = ["disgust", "fear", "happiness", "repression", "sadness", "surprise"]

# Convert to MP4
def convert_to_mp4(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 10
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    cap.release()
    out.release()
    return output_path

# Process video
def process_video(video_path, output_mp4_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        logging.error("Video has no frames.")
        return None

    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(output_mp4_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

    for i, frame in enumerate(frames):
        try:
            resized = cv2.resize(frame, (64, 64))                         # Resize to model input
            normalized = resized.astype("float32") / 255.0               # Normalize
            input_tensor = np.expand_dims(normalized, axis=0)           # Add batch dimension
            predictions = model.predict(input_tensor, verbose=0)        # Get prediction
            pred_class = np.argmax(predictions[0])                      # Get class index
            label = emotions[pred_class]                                # Map to emotion label
        except Exception as e:
            logging.error(f"Prediction error at frame {i}: {e}")
            label = "error"

        # Annotate and write frame
        cv2.putText(frame, f"Emotion: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 255), 2)
        out.write(frame)

    out.release()
    return output_mp4_path if os.path.exists(output_mp4_path) else None

# Streamlit UI
st.title("🎥 Micro-Expression Video Analyzer")

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])
if uploaded_file is not None:
    filename = secure_filename(uploaded_file.name)
    file_path = os.path.join(UPLOAD_FOLDER, filename)

    # Save uploaded file to disk
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Convert to mp4 if not already
    mp4_path = os.path.join(UPLOAD_FOLDER, filename.rsplit('.', 1)[0] + ".mp4")
    if not filename.lower().endswith('.mp4'):
        file_path = convert_to_mp4(file_path, mp4_path)
    else:
        mp4_path = file_path

    # Process video
    output_filename = filename.rsplit('.', 1)[0] + "_processed.mp4"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    result = process_video(mp4_path, output_path)

    if result:
        st.success("✅ Video processed successfully!")
        st.video(result)
        with open(result, 'rb') as f:
            st.download_button("Download Processed Video", f, file_name=output_filename)
    else:
        st.error("❌ Failed to process video. See logs for details.")
