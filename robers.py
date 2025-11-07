import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras
import serial
import time

# ---------------------- Arduino Serial ----------------------
# Replace COM4 with your Arduino COM port
arduino = serial.Serial('COM4', 9600, timeout=1)
time.sleep(2)  # wait for Arduino connection
# ------------------------------------------------------------

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
fig, ax = plt.subplots()
bar_plot = ax.bar(classes, [0, 0, 0])
ax.set_ylim([0, 1])
plt.title("Live Prediction Confidence")
# -------------------------------------------------------------------

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, (128, 128))

    frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    frame_rgb = frame_rgb / 255.0

    input_frame = np.expand_dims(frame_rgb, axis=0)
    result = model.predict(input_frame, verbose=0)

    predicted_class = classes[np.argmax(result)]

    # ---------------------- SEND TO ARDUINO ----------------------
    arduino.write(predicted_class.encode() + b"\n")  # send class name
    # --------------------------------------------------------------

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    cv.putText(frame, predicted_class, (10, 20), cv.FONT_HERSHEY_SIMPLEX,
               0.6, (0, 0, 255), 2)

    frame = cv.resize(frame, (480, 480))
    cv.imshow('video', frame)

    # ---------------------- UPDATE MATPLOTLIB ----------------------
    for i, b in enumerate(bar_plot):
        b.set_height(result[0][i])
    fig.canvas.draw()
    fig.canvas.flush_events()
    # --------------------------------------------------------------

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
arduino.close()
cv.destroyAllWindows()
