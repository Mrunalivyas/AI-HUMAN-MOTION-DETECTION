
import sqlite3
import os

# Connect to database
db_path = os.path.join("database", "security.db")

connection = sqlite3.connect(db_path)

cursor = connection.cursor()

# Fetch all detection records
cursor.execute("""
SELECT
    id,
    detected_at,
    people_count,
    tracking_ids,
    movement_status,
    video_filename
FROM detection_history
ORDER BY id DESC
""")

records = cursor.fetchall()

print("\n========== HUMAN DETECTION HISTORY ==========\n")

if not records:

    print("No detection records found.")

else:

    for record in records:

        print("--------------------------------------------")

        print("Event ID:", record[0])
        print("Date and Time:", record[1])
        print("People Detected:", record[2])
        print("Tracking IDs:", record[3])
        print("Movement:", record[4])
        print("Video File:", record[5])

        video_path = os.path.join(
            "recordings",
            record[5]
        )

        if os.path.exists(video_path):
            print("Video Status: Available")
        else:
            print("Video Status: File not found")

print("\n============================================")

connection.close()
