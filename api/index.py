import os
os.environ['TORCH_HOME'] = '/tmp'
os.environ['YOLO_CONFIG_DIR'] = '/tmp'

import cv2
import numpy as np
import torch
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import pathlib

# Fix path issue for Windows if torch hub uses pathlib.PosixPath
temp = pathlib.PosixPath
try:
    pathlib.PosixPath = pathlib.WindowsPath
except Exception:
    pass

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from notebook
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best.pt')
SKIN_MULTIPLIER = 5 * 2.3
PIXEL_TO_CM_CONSTANT = 5.0

DENSITY_DICT = {
    "Apple": 0.96, "Banana": 0.94, "Carrot": 0.641,
    "Onion": 0.513, "Orange": 0.482, "Tomato": 0.481, "Qiwi": 0.575
}

CALORIE_DICT = {
    "Apple": 52, "Banana": 89, "Carrot": 41,
    "Onion": 40, "Orange": 47, "Tomato": 18, "Qiwi": 44
}

# Load model globally to cache it during lambda lifecycle
model = None
try:
    # Use ultralytics to load the model locally or from hub
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=False, _verbose=False)
except Exception as e:
    print(f"Error loading model: {e}")

def get_volume(label, area_pixel, skin_area_pixel, pix_to_cm, contour):
    if skin_area_pixel == 0:
        return 0
    area_cm2 = (area_pixel / skin_area_pixel) * SKIN_MULTIPLIER
    if label in ["Apple", "Orange", "Qiwi", "Tomato", "Onion"]:
        radius = np.sqrt(area_cm2 / np.pi)
        return (4/3) * np.pi * (radius**3)
    if label in ["Banana", "Carrot"]:
        rect = cv2.minAreaRect(contour)
        height = max(rect[1]) * pix_to_cm
        radius = area_cm2 / (2.0 * height) if height > 0 else 0
        return np.pi * (radius**2) * height if radius > 0 else area_cm2 * 0.5
    return 0

def estimate_calories(label, volume):
    if label not in DENSITY_DICT: return 0, 0
    mass = volume * DENSITY_DICT[label]
    total_kcal = (CALORIE_DICT[label] / 100.0) * mass
    return mass, total_kcal

def segment_object(img, bbox, label):
    x, y, w, h = bbox
    pad = 5
    img_h, img_w = img.shape[:2]
    x1, y1, x2, y2 = max(0, x-pad), max(0, y-pad), min(img_w, x+w+pad), min(img_h, y+h+pad)
    crop = img[y1:y2, x1:x2]
    if crop.size == 0: return None, 0, 1.0
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask_plate = cv2.inRange(hsv, np.array([0, 0, 50]), np.array([200, 90, 250]))
    crop_filtered = cv2.bitwise_and(crop, crop, mask=cv2.bitwise_not(mask_plate))
    gray = cv2.cvtColor(crop_filtered, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None, 0, 1.0
    cnt = sorted(contours, key=cv2.contourArea)[-2 if label == "Carrot" and len(contours) >= 2 else -1]
    
    pix_to_cm = 1.0
    if label == "thumb":
        rect = cv2.minAreaRect(cnt)
        pix_height = max(rect[1])
        if pix_height > 0: pix_to_cm = PIXEL_TO_CM_CONSTANT / pix_height
            
    return cnt, cv2.contourArea(cnt), pix_to_cm

@app.get("/")
def read_root():
    return {"status": "ok", "message": "Food Calorie Tracker API is running."}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=500, content={"error": "Model not loaded properly."})

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        return JSONResponse(status_code=400, content={"error": "Invalid image format."})

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 1. Detection
    results = model(img_rgb)
    detections = results.pandas().xyxy[0]
    
    thumb_data = None
    food_items = []
    
    # Process thumb
    for _, row in detections.iterrows():
        label = row['name']
        if label == "thumb" or (isinstance(label, str) and "thumb" in label.lower()):
            bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']-row['xmin']), int(row['ymax']-row['ymin'])]
            cnt, area, pix_to_cm = segment_object(img, bbox, "thumb")
            if cnt is not None:
                thumb_data = {'area': area, 'pix_to_cm': pix_to_cm}
                break
    
    if not thumb_data:
        # Fallback if no thumb is found
        thumb_data = {'area': 1000, 'pix_to_cm': 0.05}

    # Process foods
    for _, row in detections.iterrows():
        label = row['name']
        if label == "thumb": continue
        
        if label not in DENSITY_DICT:
            continue
            
        bbox = [int(row['xmin']), int(row['ymin']), int(row['xmax']-row['xmin']), int(row['ymax']-row['ymin'])]
        cnt, area, _ = segment_object(img, bbox, label)
        
        if cnt is not None:
            vol = get_volume(label, area, thumb_data['area'], thumb_data['pix_to_cm'], cnt)
            mass, kcal = estimate_calories(label, vol)
            
            item_data = {
                'label': label,
                'bbox': bbox,
                'kcal': round(kcal, 1),
                'mass': round(mass, 1)
            }
            food_items.append(item_data)
            
            # Draw on image
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label_text = f"{label}: {round(kcal)} kcal"
            cv2.putText(img, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
    # Encode annotated image to base64
    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return JSONResponse(content={
        "status": "success",
        "foods": food_items,
        "annotated_image": img_base64
    })
