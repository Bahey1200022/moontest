import json
import time
import cv2
import numpy as np
import io
import base64
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from ultralytics import YOLO
import torch
from safe2 import *
from match import *
from datetime import date
from pymongo import MongoClient
import requests
from time_table import calc_time

home_assistant_url = "http://homeassistant.local:8123/api/states/sensor.face_recognition"
access_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiIyZjJmYjU1MDNjY2E0MTAxYTJkNDY5ZGM5MjQ4NzZiZiIsImlhdCI6MTc0NDExNDkyMiwiZXhwIjoyMDU5NDc0OTIyfQ.lqouYGvta5yAePjdy6vgLYYIlcprIK-cf1baDI5QcGc"  # Replace with yours
headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
}
client = MongoClient("mongodb+srv://bahey6224:skarpt@atlascluster.x07b3pp.mongodb.net/?retryWrites=true&w=majority&appName=AtlasCluster")
# Send a ping to confirm a successful connection

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
        # Step 2: Select your database
    db = client["test_db"]

    # Step 3: Select your collection (like a table)
    collection = db["test_collection"]
    calc_collection = db["calc_collection"]
except Exception as e:
    print(e)
    
def is_in_polygon(point, polygon):
    px, py = int(point[0]), int(point[1])
    return cv2.pointPolygonTest(polygon, (px, py), False) >= 0
# === Define your workstation polygon here ===
workstation_zone = np.array([
    [100, 100],   # Top-left
    [600, 100],  # Top-right
    [700, 600],  # Bottom-right
    [100, 600]    # Bottom-left
], dtype=np.int32)

    
# Load known faces
try:
    with open("known_faces.pkl", "rb") as f:
        known_faces = pickle.load(f)
except FileNotFoundError:
    known_faces = {}
    
    
    

app_insight = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
cig = torch.hub.load('ultralytics/yolov5', 'custom', path='weights.pt', source='github')
model = YOLO("best.pt")  # Your trained model
# fall = YOLO('yolov8s.pt') # Fall detection model

app_insight.prepare(ctx_id=-1, det_size=(640, 640))


def recognize_face(face_embedding, threshold=0.17):
    """Compare face embedding to known faces and return best match."""
    best_match = "Unknown"
    best_score = 0.0
    for name, known_embedding in known_faces.items():
        score = cosine_similarity([face_embedding], [known_embedding])[0][0]
        if score > best_score and score > threshold:
            best_match = name
            best_score = score
        print(f"Face match: {name} - Score: {score}")    
    return best_match

while True:
    start=time.time()
    image= cv2.imread("screenshots/image1743930652.jpg")
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR format
    
    
                # === Fill green semi-transparent zone ===
    overlay = frame.copy()
    cv2.fillPoly(overlay, [workstation_zone], color=(0, 255, 0))
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [workstation_zone], isClosed=True, color=(0, 255, 0), thickness=3)


    # Run face recognition
    faces = app_insight.get(frame)
    # frame,avg_falls = detect_fall(frame, fall)
    # print("Falls:", avg_falls)

    if faces:
        face_data = {}  # Dictionary to store bounding boxes per unique face
        overall_avg = {}  # Dictionary to store average bounding boxes per detected face

        recognized_names = []  # Store names after renaming
        face_boxes = []  # Store bounding boxes
        inzone_names = []  # Store names of people in the workstation zone

        for i, face in enumerate(faces, start=1):  # Add index to differentiate faces
            embedding = face.embedding
            name = recognize_face(embedding)
            x1, y1, x2, y2 = face.bbox.astype(int)
            foot = (int((x1 + x2) / 2), int(y2))
            in_zone = is_in_polygon(foot, workstation_zone)
            inzone_names.append(name) if in_zone else None
            known_names = list(known_faces.keys())
            calc_time(inzone_names, calc_collection,known_names)

            # If the name is "Unknown", make it unique
            if name == "Unknown":
                name = f"Unknown{i}"  

            bbox = face.bbox.astype(int)

            # Store bounding boxes per unique face
            face_data[name] = bbox
            recognized_names.append(name)  # Update recognized names list
            face_boxes.append(bbox)  # Update face boxes list

        # print(f"Recognized faces: {recognized_names}")

        # Compute and store average bounding box per unique face
        for name, box in face_data.items():
            overall_avg[name] = int(np.mean(box))  # Compute average of (x1, y1, x2, y2)

        # print("Overall Averages:", overall_avg)

        # Draw bounding boxes with corresponding names
        for name, box in face_data.items():  # Use updated data
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2, cv2.LINE_AA)

                            

        processed_img_path = "processed_image.jpg"
        # frame,avg_falls = detect_fall(frame, fall)
    
        cv2.imwrite(processed_img_path, frame)
        # print("Falls:", avg_falls)
        # fallings=map_falls_to_overall(overall_avg, avg_falls)
        # print('Fallings:',fallings)


        # **Step 2: Run the CIG Model**
        results = cig(processed_img_path)
        # results.show()  # Display results
        # results.show()  # Display results
        bounding_boxes = results.pandas().xyxy[0][['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'name']].to_dict(orient='records')
        
        cigs=[]
        for res in bounding_boxes:
            # get average of coordinates of each bb
            avg_bb = (res['xmin'] + res['xmax'] + res['ymin'] + res['ymax']) / 4
        
            cigs.append(avg_bb)
        # print("Cigs:", cigs)
        

        # **Step 3: Run YOLO Model**
        avg_safety,final_image_path = process_image(processed_img_path, model)
        # print("Safety:", avg_safety)
        cig_mapping=map_cigs_to_person(cigs, overall_avg)
        print("Cig Mappings:", cig_mapping)
        mappings=map_safety_to_overall(avg_safety, overall_avg)
        display_mappings(mappings,collection)
        
        

        # Convert final processed image to Base64
        with open(final_image_path, "rb") as img_file:
            processed_image_base64 = base64.b64encode(img_file.read()).decode("utf-8")
        response = {
        "recognized_names": recognized_names,
        "image_base64": processed_image_base64
    }
        json_response = json.dumps(response)
                
        data = {
            "state": ", ".join(response["recognized_names"]),  # example: "Alice, Bob"
            "attributes": {
                "image_base64": response["image_base64"]
            }
        }

        
    else:
        recognized_names = ["No face detected"]
        data = {
            "state": ", ".join(recognized_names),
            
        }
    r = requests.post(home_assistant_url, headers=headers, json=data)
    print(r.status_code)
    print(r.json())
    end=time.time()
    #send img in json 
    
    print(f"Time taken: {end-start}")