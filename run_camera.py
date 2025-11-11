import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import os
from collections import Counter
from threading import Thread  # <-- 1. NEW IMPORT

# --- 2. NEW THREADED WEBCAM CLASS ---
class WebcamStream:
    """
    A class to run cv2.VideoCapture in a dedicated thread.
    This prevents the main processing loop from blocking while
    waiting for the next camera frame.
    """
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print(f"Error: Could not open webcam at src={src}.")
            exit()
            
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        print("Webcam stream started...")

    def start(self):
        # Start the thread to read frames from the video stream
        Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        # Keep looping infinitely until the thread is stopped
        while True:
            if self.stopped:
                self.stream.release()
                return
            
            # Otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # Return the frame most recently read
        return self.frame

    def stop(self):
        # Indicate that the thread should be stopped
        self.stopped = True
        print("Webcam stream stopped.")

# ---------- CONFIG ----------
DATA_DIR = './asl_alphabet_train/asl_alphabet_train/'
ASL_MODEL_PATH = 'best_asl_model.pth'
HAND_DETECTOR_PATH = 'yolov8s-hand.pt' 
IMG_SIZE = 224
BUFFER_SIZE = 10
STABILITY_THRESHOLD = int(BUFFER_SIZE * 0.7) 
CROP_PADDING = 30
PREDICTION_CONF_THRESHOLD = 0.85 

# ---------- VERIFY PATHS ----------
if not os.path.exists(DATA_DIR):
    print(f"Error: Data directory not found at {DATA_DIR}")
    exit()
if not os.path.exists(ASL_MODEL_PATH):
    print(f"Error: ASL Model file not found at {ASL_MODEL_PATH}")
    exit()
if not os.path.exists(HAND_DETECTOR_PATH):
    print(f"Error: Hand detector model '{HAND_DETECTOR_PATH}' not found.")
    print("Please download it first.")
    exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- LOAD CLASSES ----------
print("Loading class names...")
temp_dataset = ImageFolder(DATA_DIR, transform=transforms.ToTensor())
class_names = temp_dataset.classes
NUM_CLASSES = len(class_names)
print(f"Loaded {NUM_CLASSES} classes: {class_names}")

# ---------- ASL MODEL DEFINITION ----------
def build_asl_model(num_classes):
    model = models.mobilenet_v2(weights=None) 
    for p in model.features.parameters():
        p.requires_grad = False
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes)
    )
    return model

print("Loading trained ASL model...")
asl_model = build_asl_model(NUM_CLASSES)
asl_model.load_state_dict(torch.load(ASL_MODEL_PATH, map_location=device))
asl_model.to(device)
asl_model.eval()
print("ASL model loaded successfully.")

# ---------- IMAGE TRANSFORMS ----------
test_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------- YOLO HAND DETECTOR ----------
print("Loading YOLO hand detector...")
hand_detector = YOLO(HAND_DETECTOR_PATH)
print("Hand detector ready.")

# ---------- CAMERA LOOP ----------
# --- 3. START THREADED CAPTURE ---
cap = WebcamStream(src=0).start()
# Note: We no longer need the 'if not cap.isOpened()' check

print("\nStarting camera feed... Press 'q' to quit.\n")
sentence = ""
current_stable_letter = ""
prediction_buffer = []

while True:
    # --- 4. READ FROM THREADED STREAM (non-blocking) ---
    frame = cap.read()
    if frame is None:
        continue

    # --- 5. FLIP REMOVED ---
    # The 'frame = cv2.flip(frame, 1)' line is REMOVED.
    # This fixes the mirrored text and the data mismatch with your model.
    
    display_frame = frame.copy()

    # ---------- YOLO HAND DETECTION (with confidence) ----------
    # We detect on the NON-flipped frame, which matches the training data
    results = hand_detector(frame, verbose=False, conf=0.6)
    hands = []
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls == 0: 
                hands.append(box.xyxy[0].cpu().numpy())
                
    predicted_class = "nothing" 

    if len(hands) > 0:
        hands.sort(key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
        x1, y1, x2, y2 = map(int, hands[0])

        x1 = max(0, x1 - CROP_PADDING)
        y1 = max(0, y1 - CROP_PADDING)
        x2 = min(frame.shape[1], x2 + CROP_PADDING)
        y2 = min(frame.shape[0], y2 + CROP_PADDING)
        
        hand_crop = frame[y1:y2, x1:x2]
        if hand_crop.size > 0:
            hand_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(hand_rgb)
            input_tensor = test_transform(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(device)

            with torch.no_grad():
                output = asl_model(input_batch)
                probs = F.softmax(output, dim=1)
                confidence, pred_index = torch.max(probs, 1)
                
                if confidence.item() > PREDICTION_CONF_THRESHOLD:
                    predicted_class = class_names[pred_index.item()]
                else:
                    predicted_class = "nothing"

        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    

    # --- DEBOUNCE LOGIC ---
    prediction_buffer.append(predicted_class)
    if len(prediction_buffer) > BUFFER_SIZE:
        prediction_buffer.pop(0)

    current_flicker_pred = predicted_class 

    if len(prediction_buffer) == BUFFER_SIZE:
        most_common_pred, count = Counter(prediction_buffer).most_common(1)[0]
        
        new_stable = ""
        if count >= STABILITY_THRESHOLD:
            new_stable = most_common_pred
        else:
            new_stable = "nothing"

        if new_stable != current_stable_letter:
            current_stable_letter = new_stable
            if current_stable_letter == "del":
                sentence = sentence[:-1]
            elif current_stable_letter == "space":
                sentence += " "
            elif current_stable_letter != "nothing":
                sentence += current_stable_letter

    # ---------- DRAW HUD ----------
    # Text will now be readable
    cv2.rectangle(display_frame, (10, 10), (320, 60), (0, 0, 0), -1) 
    cv2.putText(display_frame, f"Prediction: {current_flicker_pred}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    cv2.rectangle(display_frame, (10, 430), (630, 470), (0, 0, 0), -1)
    cv2.putText(display_frame, sentence, (20, 460),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    cv2.imshow("ASL Sign Recognition", display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("\nShutting down...")
# --- 6. STOP THE THREADED STREAM ---
cap.stop()
cv2.destroyAllWindows()