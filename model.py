import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m-pose.pt')

# Define the standard frame size (change these values as needed)
standard_width = 640
standard_height = 480

# Open the video file
video_path = "./videos/walk.mp4"
cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Resize the frame to the standard size
        frame = cv2.resize(frame, (standard_width, standard_height))

        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, tracker="botsort.yaml", persist=True)
        print(results.boxes)
        print(results.keypoints)
        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
