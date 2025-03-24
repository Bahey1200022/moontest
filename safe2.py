import cv2
import numpy as np

def process_image(image_path, models):
    """
    Processes an image using the given YOLO model, annotates detected objects, 
    and returns an array with bounding box averages per detected instance.

    Args:
        image_path (str): Path to the input image.
        models: YOLO model instance.

    Returns:
        tuple: (list of detected instance bounding box averages, output image path)
    """
    # Read and encode image to reduce quality
    image = cv2.imread(image_path)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    _, encoded_img = cv2.imencode('.jpg', image, encode_param)
    image = cv2.imdecode(encoded_img, cv2.IMREAD_UNCHANGED)
    

    # Run inference
    results = models.predict(image)

    # Define class labels
    class_names = {
        0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest',
        5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'
    }

    if not results or len(results[0].boxes) == 0:
        print("No objects detected. Returning empty list and original image path.")
        return [], image_path  # No detection, return original image path

    # Store individual detected bounding boxes with averages
    detected_instances = []

    # Process results
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0].item())
            confidence = box.conf[0].item()
            label = class_names.get(class_id, "Unknown")

            # Compute bounding box average
            bbox_avg = int(np.mean([x1, y1, x2, y2]))

            # Store individual bounding box instance
            detected_instances.append({
                "class": label,
                "bounding_box_avg": bbox_avg,
                # "bbox": [x1, y1, x2, y2],  # Optional: Keep the actual bounding box values
                # "confidence": round(confidence, 2)
            })

            # Define colors for bounding boxes
            colors = {
                "NO-Hardhat": (0, 255, 0),
                "NO-Safety Vest": (0, 0, 255),
                "Safety Vest": (255, 0, 255),
                "Hardhat": (0, 0, 0)
            }
            color = colors.get(label, (128, 128, 128))

            # Skip drawing bounding boxes for certain labels
            if label in ["Person", "machinery", "vehicle", "Safety Cone", "Mask", "NO-Mask"]:
                continue

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the processed image
    output_path = "resultss.jpg"
    cv2.imwrite(output_path, image)

    return detected_instances, output_path
