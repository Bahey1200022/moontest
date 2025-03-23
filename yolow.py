from ultralytics import YOLOWorld
import time
# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-world.pt")
start=time.time()
# Define custom classes
model.set_classes(["safety vest", "safety helmet"])

# Execute prediction on an image
results = model.predict("t6.jpeg")
wnd=time.time()
print("Time taken for prediction: ",wnd-start)
# Show results
results[0].show()