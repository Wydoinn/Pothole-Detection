from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("runs/segment/train/weights/best.pt")

format = ['onnx', 'torchscript', 'ncnn']

# Export the model in the specified format
for i in format:
    model.export(format=i)