
import cv2
from ultralytics import YOLO

# Step 1: Load the YOLO AI model
print("Loading AI human detection model...")
model = YOLO("yolo11n.pt")

# Step 2: Connect laptop webcam
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Error: Cannot connect to webcam!")
    exit()

print("Webcam connected!")
print("Human detection started. Press Q to exit.")

# Step 3: Start live human detection
while True:

    success, frame = camera.read()

    if not success:
        print("Unable to read camera frame.")
        break

    # Step 4: Detect objects using YOLO
    results = model(frame, conf=0.5, verbose=False)

    human_detected = False

    # Step 5: Check detected objects
    for result in results:

        for box in result.boxes:

            class_id = int(box.cls[0])

            # COCO class 0 represents a person
            if class_id == 0:

                human_detected = True

                # Get rectangle coordinates
                x1, y1, x2, y2 = map(
                    int, box.xyxy[0].tolist()
                )

                confidence = float(box.conf[0])

                # Draw green rectangle around human
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Display human label
                label = f"Human {confidence:.2f}"

                cv2.putText(
                    frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    # Step 6: Display detection status
    if human_detected:
        status = "HUMAN DETECTED"
        color = (0, 255, 0)
    else:
        status = "NO HUMAN DETECTED"
        color = (0, 0, 255)

    cv2.putText(
        frame,
        status,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2
    )

    # Step 7: Display camera window
    cv2.imshow("AI Human Motion Detection System", frame)

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 8: Close camera
camera.release()
cv2.destroyAllWindows()

print("Human detection stopped.")
