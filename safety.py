
from ultralytics import YOLO
import cv2
import torch
import time
# Load the YOLOv8 model
model = YOLO("best.pt")  # Your trained model
# Load the image
image_path = "t6.jpeg"
start=time.time()
image = cv2.imread(image_path)
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]  # Reduce quality
_, encoded_img = cv2.imencode('.jpg', image, encode_param)
image = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)

# Run inference
results = model.predict(image)
# Define class labels
class_names = {0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'}
# Process results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
        class_id = int(box.cls[0].item())  # Get class index
        confidence = box.conf[0].item()  # Confidence score

        # Get label text
        label = class_names.get(class_id, "Unknown")
        
        if label == "NO-Hardhat":
            color = (0, 255, 0)  # Green
        elif label == "NO-Mask":
            color = (255, 0, 0)  # Red
        elif label == "NO-Safety Vest":
            color = (0, 0, 255)  # Blue
        elif label == "Safety Cone":
            color = (255, 255, 0)  # Yellow
        elif label == "Safety Vest":
            color = (255, 0, 255)  # Magenta
        elif label == "Hardhat":
            color = (0, 0, 0)  # Black
        else:
            color = (128, 128, 128)  # Default gray if label is unknown

        
        if label == "Person" or label == "machinery" or label == "vehicle":
            continue

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# save the image
cv2.imwrite("result.jpg", image)


end=time.time()
print(f"Time taken : {end - start:.2f} seconds")                       
print("Done!")

