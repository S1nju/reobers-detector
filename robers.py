import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras
import serial
import time

# ----------------- SERIAL TO ARDUINO -----------------
# Change COM4 to whatever your Arduino shows in Device Manager
arduino = serial.Serial('COM4', 9600)  
time.sleep(2)  # allow Arduino reset
print("✅ Connected to Arduino on", arduino.port)
# -----------------------------------------------------


model = keras.models.load_model('model2.keras', compile=False)
classes = ['normal', 'bad driver', 'rober']

faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
if faceCascade.empty():
    raise IOError("Unable to load the face cascade classifier XML file")

cap = cv.VideoCapture(1)
if not cap.isOpened():
    cap = cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Can't open webcam")


# ---------------------- Matplotlib Live Plot ----------------------
plt.ion()

fig = plt.figure(figsize=(10, 5))

# Left: Video display
ax1 = fig.add_subplot(1, 2, 1)
img_plot = ax1.imshow(np.zeros((480, 480, 3), dtype=np.uint8))
ax1.axis('off')
ax1.set_title("Live Video")

# Right: Bar graph
ax2 = fig.add_subplot(1, 2, 2)
bar_plot = ax2.bar(classes, [0, 0, 0], color=['green', 'orange', 'red'])
ax2.set_ylim([0, 1])
ax2.set_title("Prediction Confidence")
# -------------------------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    # Prediction input frame (128x128)
    resized = cv.resize(frame, (128, 128))
    resized_rgb = cv.cvtColor(resized, cv.COLOR_BGR2RGB) / 255.0
    input_frame = np.expand_dims(resized_rgb, axis=0)

    result = model.predict(input_frame, verbose=0)
    predicted_class = classes[np.argmax(result)]

    # ➤ Send class name to Arduino
    arduino.write((predicted_class + "\n").encode())  # <-- HERE

    # Draw face box on actual frame
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.putText(frame, predicted_class, (10, 25),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Convert for matplotlib
    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    img_plot.set_data(frame_rgb)

    for i, bar in enumerate(bar_plot):
        bar.set_height(result[0][i])

    plt.pause(0.001)
