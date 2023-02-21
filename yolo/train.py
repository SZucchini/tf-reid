from ultralytics import YOLO

model = YOLO("./weights/yolov8x.pt")
model.train(data="./shoes.yaml", epochs=300)
model.val()
