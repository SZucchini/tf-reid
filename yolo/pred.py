from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict(source="./yolo_input", save=True, classes=29)
