import cvzone
import numpy as np

def detect_fall(frame, model):
    results = model(frame)
    bounding_boxes = []
    
    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = int(box.cls[0])
            
            # Ensure detection is a person
            if confidence > 0.4 and class_detect == 'person':  # Assuming class 0 is 'person'
                height = y2 - y1
                width = x2 - x1
                threshold = height - width  # Fall condition: lying down
                
                bounding_boxes.append((x1, y1, x2, y2))
                
                if threshold < 0:  # Fall detected
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=3, scale=2, colorR=(0, 0, 255))
    
    # Compute average bounding box if there are detections
    if bounding_boxes:
        avg_bbox = np.mean(bounding_boxes, axis=0).tolist()
    else:
        avg_bbox = None
    
    return frame, avg_bbox  # Return the processed frame and average bounding box




def map_falls_to_overall(overall_avg, avg_falls):
    fall_mappings = {}
    
    for unknown, avg in overall_avg.items():
        if avg_falls:
            closest_fall = min(avg_falls, key=lambda f: abs(f - avg))
            fall_mappings[unknown] = closest_fall
        else:
            fall_mappings[unknown] = 'Safe'
    
    return fall_mappings
