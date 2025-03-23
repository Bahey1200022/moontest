import cvzone


def detect_fall(frame, model):
   
    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = int(box.cls[0])

            # Ensure detection is a person
            if confidence > 0.8 and class_detect == 'person':  # Class 0 is usually 'person'
                height = y2 - y1
                width = x2 - x1
                threshold = height - width  # Fall condition: lying down

                if threshold < 0:  # Fall detected
                    cvzone.putTextRect(frame, 'Fall Detected', [x1, y1 - 40], thickness=3, scale=2, colorR=(0, 0, 255))

    return frame  # Return the processed frame