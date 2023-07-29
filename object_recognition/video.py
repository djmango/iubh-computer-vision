from pathlib import Path

from PIL import Image
import cv2

from object_recognition.models import YolosObjectRecognition

HERE = Path(__file__).parent

# Initialize the model
model = YolosObjectRecognition()

# Open the video stream (use '0' for webcam input, or replace with a video file path)
cap = cv2.VideoCapture(0)

# Process the video stream
while True:
    # Read the frame from the video stream
    ret, frame = cap.read()
    
    # If frame reading was successful, process it
    if ret:
        # Convert the frame to PIL Image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Run the model
        results = model.run_model(iter([frame_pil]))

         # Draw the results onto the frame
        for segment in results[0]:
            # Unpack the bounding box coordinates
            xmin, ymin, xmax, ymax = segment.bounding_box

            # Convert to integers
            xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])

            # Draw the bounding box
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Draw the label and confidence score
            label = f"{segment.label}: {segment.confidence:.2f}"
            cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Video Stream', frame)
        
        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the video capture
cap.release()

# Destroy all windows
cv2.destroyAllWindows()
