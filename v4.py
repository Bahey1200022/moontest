import os
import tempfile
import time
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
import cv2
from fastapi.templating import Jinja2Templates
from fpdf import FPDF
import numpy as np
import io
import base64
from PIL import Image
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import uvicorn
import logging
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
from safe2 import *
from match import *
from datetime import date
from pymongo import MongoClient

from time_table import calc_time
# Global dictionary to store daily data


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
client = MongoClient("mongodb+srv://bahey6224:skarpt@atlascluster.x07b3pp.mongodb.net/?retryWrites=true&w=majority&appName=AtlasCluster")
# Send a ping to confirm a successful connection
templates = Jinja2Templates(directory="templates")  
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
    
#if not os.path.exists('img_test'):
  #  os.makedirs('img_test')
                        #  
    
# create screenshots directory if it doesn't exist    
#if not os.path.exists('screenshots'):
   # os.makedirs('screenshots')

# Initialize face analysis model

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


@app.get("/", response_class=HTMLResponse)
async def get_date_picker(request: Request):
    return templates.TemplateResponse("date_picker.html", {"request": request})


@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    start=time.time()
    #reload the known faces
    
        
    
    
    """
    Mimics OpenAI's chat API.
    Expects JSON with a base64-encoded image and optional text messages.
    """
    try:
        # Ensure request format follows OpenAI style
        if "model" not in request or "messages" not in request:
            raise HTTPException(status_code=400, detail="Invalid OpenAI API format")

        # Extract image if present
        base64_image = None
        text_content = None  # Default value
        for msg in request["messages"]:
            if msg["role"] == "user":
                content = msg["content"]
                if isinstance(content, list):
                    for item in content:
                        if item.get("type") == "text":  
                            text_content = item.get("text").replace(":", "") # Extract the text value
                            print(f"Extracted text: {text_content}")
                        if isinstance(item, dict) and item.get("type") == "image_url":
                            image_url = item["image_url"]["url"]
                              # Case 1: Extract filename if image is from a URL
                            
                            if image_url.startswith("data:image/"):
                                base64_image = image_url.split(",")[1]  # Extract base64 data
                                break

        recognized_names = []
        processed_image_base64 = ""
        img_name = ""

        if base64_image:
            # Decode and process image
            try:
                
                image_data = base64.b64decode(base64_image)
                image = Image.open(io.BytesIO(image_data))
                frame = np.array(image)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to OpenCV BGR format
                

                #save image in a dir
                # file="image{}.jpg".format(time.time())
                # cv2.imwrite(file, frame)
                img_name=text_content
                
                
                
            
                
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

                        
                        

                   
                else:
                    recognized_names = ["No face detected"]
                end=time.time()
                print(f"Time taken: {end-start}")


            except Exception as img_err:
                raise HTTPException(status_code=400, detail=f"Invalid image format: {str(img_err)}")

        # Format response in OpenAI style
        response = {
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1710000000,
            "model": request["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        # "content": f"Recognized faces: {', '.join(recognized_names)}" if recognized_names else "No faces detected."
                          "content": {
                    "text": f"Image name: {img_name} - Recognized faces: {', '.join(recognized_names)}" if recognized_names else f"Image name: {img_name} - No faces detected.",
                    "image": f"data:image/jpeg;base64,{processed_image_base64}" , # Embed Base64 image
                    
                }
                    },
                    "finish_reason": "stop"
                }
            ],
            "processed_image": processed_image_base64
        }
       

        return JSONResponse(content=response, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

#add face to the known faces
@app.post("/api/add_face")
async def add_face(name: str = Form(...), image: UploadFile = File(...)):
    """Add a face to the known faces database."""
    try:
        # Read image file
        image_data = await image.read()
        image = Image.open(io.BytesIO(image_data))
        frame = np.array(image)

        # Run face recognition
        faces = app_insight.get(frame)

        if len(faces) != 1:
            raise HTTPException(status_code=400, detail="Image must contain exactly one face")

        # Add face to known faces
        embedding = faces[0].embedding
        known_faces[name] = embedding

        # Save known faces to file
        with open("known_faces.pkl", "wb") as f:
            pickle.dump(known_faces, f)

        return {"message": f"Face for '{name}' added successfully!"}

    except Exception as e:
        return {"error": str(e)}




@app.get("/api/get_tables_per_date")
async def get_tables_stats(date: str = Query(None, example="2025-04-07")):
    try:
        if not date:
            date = datetime.today().strftime("%Y-%m-%d")
        date_obj = datetime.strptime(date, "%Y-%m-%d").date().isoformat()

        # Collect table stats
        records = list(calc_collection.find({"date": date_obj}))
        time_table = {}
        for record in records:
            name = record.get("name", "Unknown")
            total_frames = record.get("total_frames", 0)
            detected_frames = record.get("detected_frames", 0)
            tot_time = (3 * total_frames) / 60
            det_time = (3 * detected_frames) / 60
            off_time = tot_time - det_time

            time_table[name] = {
                "total_time": round(tot_time, 2),
                "detected_time": round(det_time, 2),
                "off_time": round(off_time, 2),
            }

        # Get uniform data from another collection
        uniform_stats = {}
        uniform_records = list(collection.find({"date": date_obj}))
        for doc in uniform_records:
            detected = doc.get("detected_appearances", 0)
            total = doc.get("total_appearances", 0)
            ratio = round(detected / total, 2) if total else 0
            uniform_stats[doc['name']] = "Yes" if ratio >= 0.4 else "No"

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # Title
        pdf.cell(200, 10, txt=f"Table Statistics for {date_obj}", ln=1, align="C")

        # Table Header
        pdf.cell(50, 10, txt="Name", border=1)
        pdf.cell(30, 10, txt="Total Time", border=1)
        pdf.cell(30, 10, txt="work Time", border=1)
        pdf.cell(30, 10, txt="Off Time", border=1)
        pdf.cell(30, 10, txt="Uniform", border=1)
        pdf.ln()

        # Table Rows
        for name, stats in time_table.items():
            uniform_status = uniform_stats.get(name, "_")
            pdf.cell(50, 10, txt=name, border=1)
            pdf.cell(30, 10, txt=str(stats["total_time"]), border=1)
            pdf.cell(30, 10, txt=str(stats["detected_time"]), border=1)
            pdf.cell(30, 10, txt=str(stats["off_time"]), border=1)
            pdf.cell(30, 10, txt=uniform_status, border=1)
            pdf.ln()

        # Save to temporary file
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, f"tables_{date_obj}.pdf")
        pdf.output(pdf_path)

        # Return PDF
        return FileResponse(
            pdf_path,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename=tables_report_{date_obj}.pdf"}
        )

    except ValueError:
        return JSONResponse(
            status_code=400,
            content={"error": "Invalid date format. Use YYYY-MM-DD."}
        )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
