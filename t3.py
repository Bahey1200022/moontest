import torch
import time
# Load the YOLOv5 model from the Ultralytics repository
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights.pt', source='github')
start=time.time()
# Run inference on an image
results = model('test2.jpg')  # Replace with your actual image file
end=time.time()
print(f"Time taken : {end - start:.2f} seconds")
# Show the results
results.show()  # Display the image with detections

# Save results
results.save()  # Saves detection results in 'runs/detect'
