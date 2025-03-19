import torch
import cv2
import numpy as np
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoModelForCausalLM , CodeGenTokenizerFast as Tokenizer
from PIL import Image , ImageDraw
import time
# Load Moondream model

#check device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device : {device}")

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    revision="2025-01-09",
    trust_remote_code=True,
    # Uncomment to run on GPU.
    # device_map={"": "cuda"}
)

# Load the image
image_path = "test1.jpeg"  # Change this to your image path
image = Image.open(image_path)

# #calc time
start = time.time()

#detect objects

# Object Detection
print("\nObject detection: 'cigarette'")
objects = model.detect(image, "cigarette")["objects"]
print(f"Found {len(objects)} cigarette(s)")




end= time.time()
print(f"Time taken : {end - start:.2f} seconds")                       