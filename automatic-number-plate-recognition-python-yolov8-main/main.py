import cv2
import torch
import datetime
import pandas as pd
import pytesseract
from ultralytics import YOLO
from difflib import get_close_matches

MODEL_PATH = 'runs/detect/train2/weights/best.pt' 

from ultralytics import YOLO
model = YOLO(MODEL_PATH)
print(model.names)  


license_plate_detector = YOLO(MODEL_PATH)

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

EXCEL_FILE = "vehicle_entries.xlsx"

def preprocess_plate(plate_img):
    """Enhance number plate for better OCR accuracy."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)  
    gray = cv2.resize(gray, (gray.shape[1] * 3, gray.shape[0] * 3), interpolation=cv2.INTER_CUBIC)  
    
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  
    
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
    
    return thresh

def is_valid_plate(text):
    return len(text) > 3 and any(char.isdigit() for char in text)

# def save_plate_details(plate_text):
#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     try:
#         df = pd.read_excel(EXCEL_FILE)
#     except FileNotFoundError:
#         df = pd.DataFrame(columns=["Number Plate", "Timestamp"])

#     new_entry = pd.DataFrame([[plate_text, timestamp]], columns=["Number Plate", "Timestamp"])
#     df = pd.concat([df, new_entry], ignore_index=True)

#     df.to_excel(EXCEL_FILE, index=False)
#     print(f"✅ Saved: {plate_text} at {timestamp}")

# Global slot tracker
assigned_slots = {}

MAX_SLOTS = 20

# def save_plate_details(plate_text):
#     global assigned_slots

#     # Avoid duplicates
#     if plate_text in assigned_slots:
#         return

#     # Check available slots
#     if len(assigned_slots) >= MAX_SLOTS:
#         print("All slots filled")
#         return

#     timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#     # Assign next available slot
#     assigned_slot = len(assigned_slots) + 1
#     assigned_slots[plate_text] = assigned_slot


#     try:
#         df = pd.read_excel(EXCEL_FILE)
#     except FileNotFoundError:
#         df = pd.DataFrame(columns=["Number Plate", "Timestamp", "Slot Number"])

#     new_entry = pd.DataFrame([[plate_text, timestamp, assigned_slot]],
#                              columns=["Number Plate", "Timestamp", "Slot Number"])
#     df = pd.concat([df, new_entry], ignore_index=True)
#     df.to_excel(EXCEL_FILE, index=False)
#     print(f"✅ Saved: {plate_text} at {timestamp} in slot {assigned_slot}")

def get_best_match(plate_text):
    known = list(assigned_slots.keys())
    matches = get_close_matches(plate_text, known, n=1, cutoff=0.8)
    return matches[0] if matches else plate_text


def save_plate_details(plate_text):
    global assigned_slots

    # Avoid duplicates
    if plate_text in assigned_slots:
        return

    if len(assigned_slots) >= MAX_SLOTS:
        print("All slots filled")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    assigned_slot = len(assigned_slots) + 1
    assigned_slots[plate_text] = assigned_slot

    direction = "Move East" if assigned_slot <= 10 else "Move West"
    print(f"Slot {assigned_slot}: {direction}")

    try:
        df = pd.read_excel(EXCEL_FILE)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Number Plate", "Timestamp", "Slot Number", "Direction"])

    new_entry = pd.DataFrame([[plate_text, timestamp, assigned_slot, direction]],
                             columns=["Number Plate", "Timestamp", "Slot Number", "Direction"])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    print(f"✅ Saved: {plate_text} at {timestamp} in slot {assigned_slot} ({direction})")


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    results = license_plate_detector(frame)
    
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  

            if x2 - x1 > 30 and y2 - y1 > 15:  
                plate_img = frame[y1:y2, x1:x2]
                processed_plate = preprocess_plate(plate_img)

                plate_text = pytesseract.image_to_string(
                    processed_plate, config='--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                ).strip()

                if is_valid_plate(plate_text):
                    normalized_plate = get_best_match(plate_text)
                    save_plate_details(normalized_plate)
  

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Vehicle Number Plate Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
