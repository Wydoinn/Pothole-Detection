import streamlit as st
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import time
from PIL import Image

st.set_page_config(layout="wide")

# Initialize the YOLO model and tracker
model = YOLO("runs/segment/train/weights/best_ncnn_model")
tracker = sv.ByteTrack()
mask_annotator = sv.MaskAnnotator()

def callback(frame: np.ndarray, confidence: float) -> np.ndarray:
    # Perform detection and tracking on a single frame
    results = model(frame, conf=confidence)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)
    
    # Annotate the frame with bounding boxes and traces
    return mask_annotator.annotate(frame.copy(), detections=detections)

def video_input(data_src, confidence):
    vid_file = None
    if data_src == 'Sample data':
        vid_file = "sample/vid.mp4"
    else:
        vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mov', 'avi', 'mkv', 'webm'])
        if vid_bytes:
            vid_file = "uploaded_video." + vid_bytes.name.split('.')[-1]
            with open(vid_file, 'wb') as out:
                out.write(vid_bytes.read())

    if vid_file:
        cap = cv2.VideoCapture(vid_file)
        custom_size = st.sidebar.checkbox("Custom frame size")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if custom_size:
            width = st.sidebar.number_input("Width", min_value=120, step=20, value=width)
            height = st.sidebar.number_input("Height", min_value=120, step=20, value=height)

        # Display video metadata like frame size and FPS
        fps = 0
        st1, st2, st3 = st.columns(3)
        
        with st1:
            st.markdown("## Height")
            st1_text = st.markdown(f"{height}")
        with st2:
            st.markdown("## Width")
            st2_text = st.markdown(f"{width}")
        with st3:
            st.markdown("## FPS")
            st3_text = st.markdown(f"{fps}")

        st.markdown("---")
        output = st.empty() # Placeholder for the processed frame
        prev_time = 0
        curr_time = 0
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                st.write("Can't read frame, stream ended? Exiting ....")
                break
            
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            processed_frame = callback(frame, confidence)
            output.image(processed_frame, caption="Pothole Detection Result", use_column_width=True)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            
            # Update video metadata during playback
            st1_text.markdown(f"**{height}**")
            st2_text.markdown(f"**{width}**")
            st3_text.markdown(f"**{fps:.2f}**")

        cap.release()

def image_input(data_src, confidence):
    img_file = None
    
    if data_src == 'Sample data':
        img_file = "sample/img.jpg"
    else:
        img_bytes = st.sidebar.file_uploader("Upload an image", type=['png', 'jpeg', 'jpg'])
        if img_bytes:
            img_file = "uploaded_image." + img_bytes.name.split('.')[-1]
            Image.open(img_bytes).save(img_file)

    if img_file:
        image = cv2.imread(img_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            processed_image = callback(image, confidence)
            st.image(processed_image, caption="Pothole Detection Result")

def main():
    st.title("Pothole Detection")
    st.sidebar.title("Settings")

    input_type = st.sidebar.radio("Select input type:", ['Image', 'Video'])
    input_source = st.sidebar.radio("Select input source:", ['Sample data', 'Upload your own data'])
    confidence = st.sidebar.slider('Confidence Threshold', min_value=0.1, max_value=1.0, value=0.3)

    # Call respective input handling functions based on user choice
    if input_type == 'Video':
        video_input(input_source, confidence)
    else:
        image_input(input_source, confidence)

if __name__ == "__main__":
    main()