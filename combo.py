import time
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
import cv2
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
from safe import *
logging.basicConfig(
    filename="app.log",  # Log file name
    filemode="a",  # Append mode
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
    level=logging.INFO  # Log level
)
logging.info("Starting application...")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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

app_insight.prepare(ctx_id=-1, det_size=(640, 640))

def recognize_face(face_embedding, threshold=0.10):
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

@app.post("/v1/chat/completions")
async def chat_completions(request: dict):
    start=time.time()
    #reload the known faces
    
        
    
    
    """
    Mimics OpenAI's chat API.
    Expects JSON with a base64-encoded image and optional text messages.
    """
    try:
        logging.info(f"Received request: {request}")
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
                                logging.info(f"Extracted base64 image: {base64_image[:100]}...")  # Log first 100 chars
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
                logging.info("Image received and processed: Shape = %s", frame.shape)                # see image for debugging
                #save image in a dir
                #file="img_test/image{}.jpg".format(time.time())
                #cv2.imwrite(file, frame)
                img_name=text_content
                
                
                
            
                
                

                # Run face recognition
                faces = app_insight.get(frame)

                if faces:
                    recognized_names = []  # Reset list to store detected names for this frame
                    face_boxes = []  # Store bounding boxes

                    for face in faces:
                        embedding = face.embedding
                        name = recognize_face(embedding)
                        recognized_names.append(name)  # Store each recognized name
                        face_boxes.append(face.bbox.astype(int))  # Store corresponding bounding box
                    
                    print(f"Recognized faces: {recognized_names}")     

                    # Draw bounding boxes
                     # Draw bounding boxes with corresponding names
                    for name, box in zip(recognized_names, face_boxes):
                        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                        cv2.putText(frame, name, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.6, (0, 255, 0), 2, cv2.LINE_AA)
                        

                    processed_img_path = "processed_image.jpg"
                    cv2.imwrite(processed_img_path, frame)

                    # **Step 2: Run the CIG Model**
                    cig(processed_img_path)
                    

                    # **Step 3: Run YOLO Model**
                    final_image_path = process_image(processed_img_path, model)

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






if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
