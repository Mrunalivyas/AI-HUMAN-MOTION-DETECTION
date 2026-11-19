
import cv2
import os
import time
import math
import winsound
import threading
import sqlite3

from datetime import datetime
from ultralytics import YOLO


# ==========================================
# 1. DATABASE SETUP
# ==========================================

os.makedirs("database", exist_ok=True)
os.makedirs("recordings", exist_ok=True)

db_path = os.path.join("database", "security.db")

db = sqlite3.connect(db_path)

cursor = db.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detection_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    detected_at TEXT,
    people_count INTEGER,
    tracking_ids TEXT,
    movement_status TEXT,
    video_filename TEXT
)
""")

db.commit()

print("Database connected successfully!")


# ==========================================
# 2. CONTINUOUS ALARM
# ==========================================

alarm_active = threading.Event()
shutdown_alarm = threading.Event()


def continuous_alarm():

    while not shutdown_alarm.is_set():

        if alarm_active.is_set():

            winsound.Beep(1200, 400)
            shutdown_alarm.wait(0.1)

        else:

            shutdown_alarm.wait(0.1)


alarm_thread = threading.Thread(
    target=continuous_alarm,
    daemon=True
)

alarm_thread.start()


# ==========================================
# 3. LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

model = YOLO("yolo11n.pt")


# ==========================================
# 4. CONNECT WEBCAM
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("Cannot connect to webcam!")

    shutdown_alarm.set()
    alarm_thread.join()
    db.close()
    exit()


print("Webcam connected!")


# ==========================================
# 5. CAMERA SETTINGS
# ==========================================

fps = camera.get(cv2.CAP_PROP_FPS)

if fps <= 0 or fps > 60:
    fps = 20.0

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))


# ==========================================
# 6. RECORDING VARIABLES
# ==========================================

recording = False
video_writer = None

last_human_time = 0
post_record_seconds = 5

current_video_filename = None


# ==========================================
# 7. TRACKING VARIABLES
# ==========================================

previous_positions = {}

print("AI Security Camera Started!")
print("Press Q to stop.")


# ==========================================
# 8. MAIN APPLICATION
# ==========================================

try:

    while True:

        success, frame = camera.read()

        if not success:
            print("Camera frame error!")
            break

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.5,
            verbose=False
        )

        boxes = results[0].boxes

        human_detected = (
            boxes is not None and len(boxes) > 0
        )

        people_count = 0
        tracking_ids = []
        movement_statuses = []

        if human_detected:

            people_count = len(boxes)

            if boxes.id is not None:

                ids = boxes.id.int().cpu().tolist()
                coordinates = boxes.xyxy.int().cpu().tolist()

                for track_id, box in zip(ids, coordinates):

                    tracking_ids.append(track_id)

                    x1, y1, x2, y2 = box

                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    movement = "New Person"

                    if track_id in previous_positions:

                        old_x, old_y = previous_positions[track_id]

                        distance = math.sqrt(
                            (center_x - old_x) ** 2
                            + (center_y - old_y) ** 2
                        )

                        movement = (
                            "Moving" if distance > 5
                            else "Stationary"
                        )

                    previous_positions[track_id] = (
                        center_x, center_y
                    )

                    movement_statuses.append(movement)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = f"Person {track_id}: {movement}"

                    cv2.putText(
                        frame,
                        label,
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

            else:

                movement_statuses.append("Untracked")

                for box in boxes:

                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

        current_time = time.time()


        # ==================================
        # 9. ALARM CONTROL
        # ==================================

        if human_detected:
            alarm_active.set()
        else:
            alarm_active.clear()


        # ==================================
        # 10. AUTOMATIC VIDEO RECORDING
        # ==================================

        if human_detected:

            last_human_time = current_time

            if not recording:

                timestamp = datetime.now().strftime(
                    "%Y-%m-%d_%H-%M-%S"
                )

                current_video_filename = (
                    f"human_{timestamp}.mp4"
                )

                video_path = os.path.join(
                    "recordings",
                    current_video_filename
                )

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")

                video_writer = cv2.VideoWriter(
                    video_path,
                    fourcc,
                    fps,
                    (width, height)
                )

                if video_writer.isOpened():

                    recording = True

                    print("Recording started!")

                    # ==================================
                    # 11. SAVE DETECTION IN DATABASE
                    # ==================================

                    detected_at = datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )

                    ids_text = ", ".join(
                        map(str, tracking_ids)
                    )

                    movement_text = ", ".join(
                        movement_statuses
                    )

                    cursor.execute("""
                    INSERT INTO detection_history
                    (
                        detected_at,
                        people_count,
                        tracking_ids,
                        movement_status,
                        video_filename
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """, (
                        detected_at,
                        people_count,
                        ids_text,
                        movement_text,
                        current_video_filename
                    ))

                    db.commit()

                    print("Detection saved in database!")

                else:

                    print("Video recording failed!")

                    video_writer.release()
                    video_writer = None
                    current_video_filename = None

        if recording and video_writer is not None:

            video_writer.write(frame)

            if (
                not human_detected
                and current_time - last_human_time
                >= post_record_seconds
            ):

                video_writer.release()
                video_writer = None

                recording = False

                print("Video saved successfully!")


        # ==================================
        # 12. DISPLAY STATUS
        # ==================================

        status = (
            "HUMAN DETECTED"
            if human_detected
            else "NO HUMAN DETECTED"
        )

        color = (
            (0, 255, 0)
            if human_detected
            else (0, 0, 255)
        )

        cv2.putText(
            frame,
            status,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

        cv2.putText(
            frame,
            f"People: {people_count}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "ALARM ON" if human_detected else "ALARM OFF",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if human_detected
            else (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "RECORDING" if recording else "NOT RECORDING",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if recording
            else (255, 255, 255),
            2
        )

        cv2.imshow("AI Security Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


except KeyboardInterrupt:

    print("Application interrupted!")


finally:

    # ==================================
    # 13. CLEANUP
    # ==================================

    alarm_active.clear()
    shutdown_alarm.set()

    if video_writer is not None:
        video_writer.release()
        print("Final video saved!")

    camera.release()
    cv2.destroyAllWindows()

    alarm_thread.join(timeout=2)

    db.close()

    print("Database disconnected!")
    print("Security camera stopped!")
