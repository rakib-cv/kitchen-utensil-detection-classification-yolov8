"""
Runs YOLOv8 inference on kitchen scene videos and saves annotated outputs.
Used for qualitative visualization of detection performance.
"""
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('v8sbest.pt')

# Define the mapping of class IDs to utensil names
class_names = [
    'Bottle-Opener', 'Bread-Knife', 'Can-Opener', 'Desert Spoon', 'Dinner-Fork',
    'Dinner-Knife', 'Fish-Slice', 'Kitchen-Knife', 'Ladle', 'Masher', 'Peeler',
    'Pizza-Cutter', 'Potato-peeler', 'Serving-Spoon', 'Soup-Spoon', 'Spatula',
    'Tea-Spoon', 'Tongs', 'Whisk', 'Wooden-Spoon'
]

# Define categories for danger and safe utensils
danger_utensils = {
    'Bread-Knife', 'Can-Opener', 'Dinner-Fork', 'Dinner-Knife', 'Fish-Slice', 
    'Kitchen-Knife', 'Masher', 'Peeler', 'Pizza-Cutter', 'Potato-peeler', 'Tongs'
}
safe_utensils = {
    'Bottle-Opener', 'Desert Spoon', 'Ladle', 'Serving-Spoon', 'Soup-Spoon', 
    'Spatula', 'Tea-Spoon', 'Wooden-Spoon'
}

# Open the video file
video_path = "Occluded 2.mp4"
cap = cv2.VideoCapture(video_path)

# Get the video frames' width, height, and FPS to save the output video with the same parameters
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object
output_video_path = "output_video_Occluded 2.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

# Loop through the video frames
n = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)
   
    # Process each detection
    for result in results:
        for box in result.boxes:
            label_id = int(box.cls.item())  # Convert label to int
            score = float(box.conf.item())  # Convert score to float
            x_min, y_min, x_max, y_max = map(int, box.xyxy[0])

            # Get the utensil name from the class ID
            label = class_names[label_id]

            # Determine the color based on the utensil category
            color = (0, 255, 0) if label in safe_utensils else (0, 0, 255)  # Green for safe, Red for danger

            # Draw bounding box and label
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{label} {score:.2f}", (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    # Write the annotated frame into the output video
    out.write(frame)
    
    # Optionally, save each frame as an image
    cv2.imwrite("Video_Results Occluded 2/frame_%d.png" % n, frame)
    cv2.imwrite("Video_Frames/Video/frame_%d.png" % n, frame)


    # Optionally, display the annotated frame
    cv2.imshow("YOLOv8 Inference", frame)
    n += 1

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything once done
cap.release()
out.release()
cv2.destroyAllWindows()
