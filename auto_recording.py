
import cv2
import os
import time
from datetime import datetime
from ultralytics import YOLO

# 1. Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

# 2. Connect webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Camera not connected!")
    exit()

# 3. Create recordings folder automatically
os.makedirs("recordings", exist_ok=True)

# 4. Get camera properties
fps = camera.get(cv2.CAP_PROP_FPS)

if fps <= 0 or fps > 60:
    fps = 20.0

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 5. Recording settings
recording = False
video_writer = None
last_human_time = 0
post_record_seconds = 5

print("Human detection recording started!")
print("Press Q to stop.")

# 6. Start camera loop
while True:

    success, frame = camera.read()

    if not success:
        print("Cannot read camera.")
        break

    # 7. Detect humans
    results = model(
        frame,
        classes=[0],
        conf=0.5,
        verbose=False
    )

    human_detected = False

    for result in results:

        for box in result.boxes:

            human_detected = True

            x1, y1, x2, y2 = map(
                int, box.xyxy[0].tolist()
            )

            confidence = float(box.conf[0])

            # Draw human rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Human {confidence:.2f}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    current_time = time.time()

    # 8. Start recording when human is detected
    if human_detected:

        last_human_time = current_time

        if not recording:

            timestamp = datetime.now().strftime(
                "%Y-%m-%d_%H-%M-%S"
            )

            filename = os.path.join(
                "recordings",
                f"human_{timestamp}.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                fps,
                (width, height)
            )

            if video_writer.isOpened():

                recording = True
                print("Recording started:", filename)

            else:

                print("Error: Cannot create video file!")
                video_writer.release()
                video_writer = None

    # 9. Stop after 5 seconds without a detected human
    if recording:

        if video_writer is not None:
            video_writer.write(frame)

        if not human_detected:

            elapsed = current_time - last_human_time

            if elapsed >= post_record_seconds:

                video_writer.release()
                video_writer = None

                recording = False

                print("Recording saved successfully!")

    # 10. Display status
    if recording:

        status = "RECORDING"
        color = (0, 0, 255)

    elif human_detected:

        status = "HUMAN DETECTED"
        color = (0, 255, 0)

    else:

        status = "MONITORING - NO HUMAN"
        color = (255, 255, 0)

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    # 11. Display camera
    cv2.imshow("AI Security Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 12. Save any active recording before exit
if video_writer is not None:
    video_writer.release()
    print("Final recording saved!")

camera.release()
cv2.destroyAllWindows()

print("Application stopped!")
