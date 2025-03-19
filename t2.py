
from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO("fine_tune_YOLOv8n.pt")  # Your trained model

# Load the image
image_path = "3.jpg"
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Define class labels
class_names = {0: 'cigarette', 1: 'person', 2: 'smoke'}

# Process results
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Extract bounding box coordinates
        class_id = int(box.cls[0].item())  # Get class index
        confidence = box.conf[0].item()  # Confidence score

        # Get label text
        label = class_names.get(class_id, "Unknown")
        color = (0, 255, 0) if label == "cigarette" else (255, 0, 0)  # Green for cigarette, blue for smoke
        
        if label == "person":
            continue

        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# save the image
cv2.imwrite("result.jpg", image)
print("Done!")
