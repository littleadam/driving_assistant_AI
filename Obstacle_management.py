import cv2
import numpy as np

# Set up video capture and object detection
cap = cv2.VideoCapture(0)
car_cascade = cv2.CascadeClassifier('cars.xml')

# Set up PID controller
Kp = 1.0
Ki = 0.1
Kd = 0.01
prev_error = 0
integral = 0

# Set up variables for speed control
target_speed = 20
current_speed = 0
max_speed = 30

# Main loop
while True:
    # Read frame from video capture
    ret, frame = cap.read()
    
    # Convert frame to grayscale for object detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars in the frame using the cascade classifier
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Draw bounding boxes around cars and calculate distance
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        distance = (w * 170) / h

    # Set target speed based on distance
    if 'distance' in locals() and distance < 20:
        target_speed = 10
    else:
        target_speed = 20
    
    # Calculate error and update PID controller
    error = target_speed - current_speed
    integral += error
    derivative = error - prev_error
    prev_error = error
    
    # Calculate control output using PID algorithm
    output = Kp * error + Ki * integral + Kd * derivative
    
    # Update speed based on control output
    current_speed += output
    
    # Limit speed to maximum
    current_speed = min(current_speed, max_speed)
    
    # Display current speed and target speed on screen
    cv2.putText(frame, f"Current speed: {current_speed:.2f} km/h", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Target speed: {target_speed:.2f} km/h", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display frame on screen
    cv2.imshow('frame', frame)
    
    # Wait for 'q' key to be pressed to quit
    if cv2.waitKey(1) == ord('q'):
        break

# Release video capture and close window
cap.release()
cv2.destroyAllWindows()
