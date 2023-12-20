import cv2
import pyttsx3

# Load the face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the text-to-speech engine
engine = pyttsx3.init()


def detect_faces(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    return faces


# Open the webcam
cap = cv2.VideoCapture(0)

# Set the display window size
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Detect faces in the current frame
    faces = detect_faces(frame)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Send a message that a face is detected
        engine.say("Face detected!")
        engine.runAndWait()

    # Display the frame with face detections
    cv2.imshow('Face Detection', frame)

    # Exit the loop if the 'Esc' key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the webcam and close the windows
cap.release()
cv2.destroyAllWindows()
