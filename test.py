import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

# Initialize the YOLO model and tracker
model = YOLO("runs/segment/train/weights/best_ncnn_model")
tracker = sv.ByteTrack()
mask_annotator = sv.MaskAnnotator(color=sv.Color.RED)

def callback(frame: np.ndarray) -> np.ndarray:
    # Perform detection and tracking on a single frame
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    # Annotate the frame with bounding boxes and traces
    return mask_annotator.annotate(frame.copy(), detections=detections)

def display_video(source_path: str):
    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = callback(frame)

        cv2.imshow("Person Detection and Tracking", processed_frame)

        # Break the loop if the user presses 'q'
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with the path to your video file
    display_video("sample/vid.mp4")