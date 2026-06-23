from flask import Flask, render_template, Response, jsonify
from collections import Counter
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import mediapipe as mp
from utils import (mediapipe_detection,draw_styled_landmarks,extract_keypoints)

app = Flask(__name__)

camera_running = False

actions = np.array([
'Hello',
'Thanks',
'I love you',
'Help',
'Sorry',
'Eat',
'Nice to meet you'
])

current_confidence = 0

model = load_model("sign_language_model.h5")

current_prediction = "Waiting..."

sequence = []
predictions = []

threshold = 0.80

def generate_frames():

    global current_prediction
    global current_confidence
    global sequence
    global predictions
    global camera_running

    sequence.clear()
    predictions.clear()
    current_prediction = "Waiting..."
    current_confidence = 0

    camera_running = True

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not access webcam")

    try:

        with mp.solutions.holistic.Holistic(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        ) as holistic:

            while camera_running:

                success, frame = cap.read()

                if not success:
                    current_prediction = "Camera Error"
                    current_confidence = 0
                    break

                image, results = mediapipe_detection(
                    frame,
                    holistic
                )

                draw_styled_landmarks(
                    image,
                    results
                )

                keypoints = extract_keypoints(results)

                sequence.append(keypoints)
                sequence = sequence[-65:]

                if len(sequence) == 65:

                    res = model.predict(
                        np.expand_dims(sequence, axis=0),
                        verbose=0
                    )[0]

                    predicted_class = np.argmax(res)
                    confidence = res[predicted_class]

                    current_confidence = confidence

                    predictions.append(predicted_class)
                    predictions = predictions[-30:]

                    if len(predictions) >= 15:

                        recent_predictions = predictions[-15:]

                        most_common = Counter(recent_predictions).most_common(1)[0]
                        if (most_common[1] >= 12 and confidence > threshold):
                            current_prediction = (
                                f"{actions[predicted_class]}"
                                f" ({confidence*100:.1f}%)"
                            )
                        else:
                            current_prediction = "Detecting..."

                ret, buffer = cv2.imencode('.jpg', image)

                if not ret:
                    continue

                frame = buffer.tobytes()

                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n'
                    + frame +
                    b'\r\n'
                )

    finally:
        camera_running = False
        cap.release()
        print("Webcam released")


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():

    return Response(
        generate_frames(),
        mimetype=
        "multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/prediction")
def prediction():

    return jsonify({
        "prediction": current_prediction,
        "confidence": float(current_confidence)
    })

@app.route("/stop_camera")
def stop_camera():

    global camera_running
    global current_prediction
    global current_confidence

    camera_running = False
    current_prediction = "Camera Off"
    current_confidence = 0

    return jsonify({
        "status": "Camera stopped"
    })

if __name__ == "__main__":
    app.run(
    host="0.0.0.0",
    port=5000,
    debug=True
    )
