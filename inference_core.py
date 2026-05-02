from pathlib import Path
from ultralytics import YOLO
import cv2
import random

def calculate_local_auas(Nh: int, I: int) -> int:
    if Nh == 0:
        hives_score = 0
    elif 0 < Nh < 20:
        hives_score = 1
    elif 20 <= Nh <= 50:
        hives_score = 2
    else: # Nh > 50
        hives_score = 3
    return hives_score + I

def load_yolo_model(model_path: str):
    return YOLO(model_path)

def predict_image(model: YOLO, img_path: str, itch_severity: int = 0, conf_thres: float = 0.25, iou_thres: float = 0.5):
    # Predict with the model on the given image path, focusing on 'Urticaria' class (class_id=1)
    # based on the validation output in cell D2YkphuiaE7_ where Urticaria is class 1.
    preds = model.predict(source=img_path, conf=conf_thres, iou=iou_thres, show=False, verbose=False, classes=1)

    Nh = len(preds[0].boxes) if preds and preds[0].boxes else 0 # Number of hives
    
    # Use the itch severity provided by the user
    I = itch_severity

    local_auas = calculate_local_auas(Nh, I)

    # Optionally save the annotated image for visualization
    img_array = None
    if preds and preds[0].plot:
        img_array = preds[0].plot(labels=False, conf=False)
    else:
        img_array = cv2.imread(img_path)

    return {
        'Hives Count (Nh)': Nh,
        'Itch Severity (I)': I,
        'Local AUAS': local_auas,
        'AnnotatedImage': img_array
    }
