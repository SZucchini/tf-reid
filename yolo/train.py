from ultralytics import YOLO

model = YOLO("yolov8n.yaml")
model.train(data="./shoe2.yaml", epochs=1000)
model.val()
