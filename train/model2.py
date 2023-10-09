from collections import defaultdict
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')

# Define the standard frame size (change these values as needed)
standard_width = 640
standard_height = 480

# Open the video file
video_path = "./videos/walk_2.mp4"
cap = cv2.VideoCapture(video_path)

# Store the track history
track_history = defaultdict(lambda: [])

# Create an empty pandas DataFrame to store the data
columns = ['Frame', 'ID', 'X', 'Y', 'Width', 'Height']+ [f'Keypoint_{i}' for i in range(32)]
data = pd.DataFrame(columns=columns)

# Loop through the video frames
frame_count = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    frame_count += 1  # Increment frame count

    if success:
        frame = cv2.resize(frame, (standard_width, standard_height))

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        if results[0].boxes is not None:  # Check if there are results and boxes
            # Get the boxes
            boxes = results[0].boxes.xywh.cpu()
            
            if results[0].boxes.id is not None:
                # If 'int' attribute exists (there are detections), get the track IDs
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Loop through the detections and add data to the DataFrame
                for i, box in zip(range(0,len(track_ids)), results[0].boxes.xywhn.cpu()):
                    x, y, w, h = box
                    keypoints = results[0].keypoints.xyn[i].cpu().numpy().flatten().tolist()
                    data = data.append({'Frame': frame_count, 'ID': track_ids[i],
                                        'X': x, 'Y': y, 'Width': w, 'Height': h, **dict(enumerate(keypoints, 0))},
                                       ignore_index=True)
                print(data.tail())  # Print the last rows of the DataFrame
            else:
                # If 'int' attribute doesn't exist (no detections), set track_ids to an empty list
                track_ids = []

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

        else:
            # If no detections, display the original frame without annotations
            cv2.imshow("YOLOv8 Tracking", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
# Sort the data by 'ID' column
sorted_data = data.sort_values(by='ID')

# Save the sorted data as a CSV file
sorted_data.to_csv('datapoints.csv', index=False)

# Print the first few rows of the sorted data
print(sorted_data.head())