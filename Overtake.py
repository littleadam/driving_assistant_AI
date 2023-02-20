import cv2

import numpy as np

# Load video or camera feed

cap = cv2.VideoCapture(0)

# Set up object detection model

car_cascade = cv2.CascadeClassifier('cars.xml')

# Set up variables for distance and speed

distance = 0

speed = 0

# Main loop

while True:

    # Capture frame from video or camera

    ret, frame = cap.read()

    

    # Convert frame to grayscale for object detection

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    

    # Detect cars in the frame using the cascade classifier

    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Draw bounding boxes around cars and calculate distance

    for (x, y, w, h) in cars:

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

        distance = (w * 170) / h

    

    # Calculate speed based on distance over time

    speed = distance / 0.1

    

    # Display distance and speed on the frame

    cv2.putText(frame, f"Distance: {distance:.2f} meters", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"Speed: {speed:.2f} m/s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Check if it's safe to overtake

    if distance > 10 and speed > 5:

        cv2.putText(frame, "Safe to overtake", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    else:

        cv2.putText(frame, "Not safe to overtake", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    

    # Display the frame

    cv2.imshow('frame', frame)

    # Press 'q' to quit the program

    if cv2.waitKey(1) == ord('q'):

        break

# Release the capture and close the window

cap.release()

cv2.destroyAllWindows()

