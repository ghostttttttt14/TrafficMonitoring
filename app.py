import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import torch
import os
from ultralytics import YOLO
from tracker import Tracker
import sqlite3


violation_folder = "violations"
os.makedirs(violation_folder, exist_ok=True)

# Load YOLO Model
model = YOLO('Models/yolov8n.pt')

# Initialize database for license plate storage
conn = sqlite3.connect("traffic_data.db")
c = conn.cursor()

# Ensure the violations table has the correct schema
c.execute("PRAGMA table_info(violations)")
columns = [row[1] for row in c.fetchall()]
if "image_path" not in columns:
    c.execute("ALTER TABLE violations ADD COLUMN image_path TEXT")
    conn.commit()

c.execute("""
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plate TEXT,
        speed INTEGER,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        image_path TEXT
    )
""")
conn.commit()

# Object detection and tracking function
def process_frame(frame, model, tracker):
    results = model(frame)
    bboxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()
    list_bbox = []
    vehicle_ids = {}  # Store class ID per object
    speed_violations = []
    violation_images = []

    # Count different types of detected objects
    person_count = sum(classes == 0)
    car_count = sum(classes == 2)
    bus_count = sum(classes == 5)
    motorcycle_count = sum(classes == 3)
    plate_count = sum(classes == 7)

    for bbox, class_id in zip(bboxes, classes):
        x1, y1, x2, y2 = bbox.astype(int)

        # Assign different colors for different object types
        if class_id == 0:  # Person
            color = (0, 0, 255)  # Red
        elif class_id == 2:  # Car
            color = (0, 255, 0)  # Green
        elif class_id == 5:  # Bus
            color = (255, 0, 0)  # Blue
        elif class_id == 3:  # Motorcycle
            color = (255, 255, 0)  # Yellow
        elif class_id == 7:  # License Plate
            color = (255, 0, 255)  # Pink
        else:
            continue  # Ignore non-vehicle detections

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"ID: {class_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        list_bbox.append([x1, y1, x2, y2])  # Send only (x1, y1, x2, y2) to the tracker
        vehicle_ids[(x1, y1, x2, y2)] = class_id  # Store class ID separately

    bbox_id = tracker.update(list_bbox)

    for bbox in bbox_id:
        x3, y3, x4, y4, obj_id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2

        # Get the class ID for this vehicle
        class_id = vehicle_ids.get((x3, y3, x4, y4), None)
        if class_id not in [2, 3, 5]:  # Only process vehicles
            continue

        # Simulated speed calculation (replace with actual speed logic)
        speed = np.random.randint(50, 150)
        if speed > 120:  # Assuming speed limit is 120 Km/h
            violation_text = f"Vehicle {obj_id}: {speed} Km/h"
            speed_violations.append(violation_text)

            # Mark the violating vehicle with a red bounding box
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 3)

            # Save full-frame screenshot for violation
            violation_image_path = os.path.join(violation_folder, f"violation_{obj_id}.jpg")
            high_res_frame = cv2.resize(frame, (1280, 720))  # HD resolution
            cv2.imwrite(violation_image_path, high_res_frame)

            # Store violation in the database
            c.execute("INSERT INTO violations (plate, speed, image_path) VALUES (?, ?, ?)", (str(obj_id), speed, violation_image_path))
            conn.commit()

            # Append image path to list for UI display
            violation_images.append(violation_image_path)

    return frame, person_count, car_count, bus_count, motorcycle_count, plate_count, speed_violations, violation_images

# Streamlit UI
st.title("🚦 Traffic Monitoring System")
st.write("Upload a video to process traffic and track vehicles, or use your webcam.")

stframe = st.empty()
st_stats = st.sidebar.empty()
st_violations = st.sidebar.empty()
st_images = st.sidebar.container()

# Webcam Option
use_webcam = st.checkbox("Use Webcam")

if use_webcam:
    cap = cv2.VideoCapture(0)
else:
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = None

if cap is not None:
    tracker = Tracker()
    speed_limit = 120
    
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.resize(frame, (1020, 500))
        frame, person_count, car_count, bus_count, motorcycle_count, plate_count, speed_violations, violation_images = process_frame(frame, model, tracker)  # Process frame with detection model
        
        inference_time = time.time() - start_time
        
        st_stats.markdown(f"""
        **Real-time Stats**  
        - **Inference Time:** {inference_time:.2f} sec  
        - **Speed Limit:** {speed_limit} Km/h  
        - **Persons:** {person_count}  
        - **Cars:** {car_count}  
        - **Buses:** {bus_count}  
        - **Motorcycles:** {motorcycle_count}  
        - **License Plates:** {plate_count}  
        """)

        if speed_violations:
            st_violations.markdown("**🚨 Speed Violations**")
            for violation in speed_violations:
                st_violations.markdown(f"- {violation}")

        # Display full-frame violation screenshots in Streamlit UI
        if violation_images:
            st.sidebar.markdown("## 📸 Captured Violations")
            for img_path in violation_images:
                st.sidebar.image(img_path, caption=os.path.basename(img_path), use_container_width=True)

        stframe.image(frame, channels="BGR")
        time.sleep(0.03)
    
    cap.release()
    conn.close()
    st.success("Processing complete! 🎉")
