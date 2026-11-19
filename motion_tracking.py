
import cv2
import math
from ultralytics import YOLO

# 1. Load YOLO model
print("Loading YOLO model...")
model = YOLO("yolo11n.pt")

# 2. Connect webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Webcam not connected!")
    exit()

# Store previous positions of tracked people
previous_positions = {}

print("Human movement tracking started!")
print("Press Q to stop.")

# 3. Start tracking
while True:

    success, frame = camera.read()

    if not success:
        print("Cannot read camera.")
        break

    # 4. Detect and track humans
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0],
        conf=0.5,
        verbose=False
    )

    people_count = 0

    if results[0].boxes is not None:

        boxes = results[0].boxes

        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()
            coordinates = boxes.xyxy.int().cpu().tolist()

            for track_id, box in zip(track_ids, coordinates):

                people_count += 1

                x1, y1, x2, y2 = box

                # Calculate center of the person
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                current_position = (center_x, center_y)

                # 5. Check movement
                movement = "New Person"

                if track_id in previous_positions:

                    old_x, old_y = previous_positions[track_id]

                    distance = math.sqrt(
                        (center_x - old_x) ** 2 +
                        (center_y - old_y) ** 2
                    )

                    if distance > 5:
                        movement = "Moving"
                    else:
                        movement = "Stationary"

                # Save current position
                previous_positions[track_id] = current_position

                # 6. Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display tracking ID and movement
                label = f"Person ID: {track_id} | {movement}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

    # 7. Display number of people
    cv2.putText(
        frame,
        f"People Detected: {people_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # 8. Show live video
    cv2.imshow("AI Human Movement Tracking", frame)

    # Stop using Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 9. Release resources
camera.release()
cv2.destroyAllWindows()

print("Tracking stopped!")
