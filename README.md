# AI Human Motion Detection System

## Project Execution Documentation

### 1. Project Requirements

- Python 3.11
- VS Code
- Laptop webcam
- OpenCV, YOLO, ByteTrack, SQLite and Pillow

### 2. Steps to Run the Project

**Step 1:** Open the project folder in VS Code.

**Step 2:** Open the terminal.

**Step 3:** Activate the virtual environment.

```
.\venv\Scripts\Activate
```

**Step 4:** Install the required libraries.

```
python -m pip install opencv-python ultralytics pillow
```

**Step 5:** Execute the application.

```
python security_camera.py
```

### 3. System Working Process

1. Webcam captures live video.
2. YOLO detects humans in real time.
3. ByteTrack tracks human movement.
4. Alarm activates when a human is detected.
5. Video recording starts automatically.
6. Recording stops after 5 seconds without detection.
7. SQLite stores detection history.
8. Tkinter GUI provides system monitoring and controls (GUI development in progress).

### 4. Expected Output

- Real-time human detection.
- Human movement tracking.
- Automatic alarm activation.
- Automatic video recording.
- Detection history in SQLite.
- Desktop monitoring interface.
