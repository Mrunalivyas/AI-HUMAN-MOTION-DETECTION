
import cv2

# Step 1: Connect to the laptop webcam
camera = cv2.VideoCapture(0)

# Step 2: Check camera connection
if not camera.isOpened():
    print("Error: Webcam not connected!")
    exit()

print("Camera connected successfully!")
print("Press Q to close the camera.")

# Step 3: Start live video
while True:

    # Read one frame from webcam
    success, frame = camera.read()

    if not success:
        print("Error: Cannot read camera frame.")
        break

    # Display live video
    cv2.imshow("Human Motion Detection - Live Camera", frame)

    # Press Q to stop the camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Step 4: Release camera
camera.release()

# Step 5: Close all windows
cv2.destroyAllWindows()

print("Camera stopped successfully!")
