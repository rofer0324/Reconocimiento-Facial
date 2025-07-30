from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import pandas as pd
import mediapipe as mp
from recognizer import FaceRecognizer

app = Flask(__name__)

# Configuración
CAMERA_URLS = {
    0: "rtsp://channels/1",
    1: "rtsp://channels/2",
    2: "rtsp://channels/3",
    3: "rtsp://channels/4"
}
DEVICE_INDEX = 0  # índice actual
VIDEO_CAPTURE = None
CSV_PATH = "out/asistencia.csv"

recognizer = FaceRecognizer(
    checkpoint="checkpoints/model_epoch10.pth",
    emb_path="embeddings/embeddings.npy",
    paths_path="embeddings/paths.npy",
    threshold=0.9,
    csv_path=CSV_PATH
)

# MediaPipe
mp_face_detection = mp.solutions.face_detection

def generate_frames():
    global VIDEO_CAPTURE
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8) as detector:
        while VIDEO_CAPTURE and VIDEO_CAPTURE.isOpened():
            success, frame = VIDEO_CAPTURE.read()
            if not success:
                VIDEO_CAPTURE.release()
                VIDEO_CAPTURE = None
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results.detections:
                ih, iw, _ = frame.shape
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = max(int(bbox.xmin * iw), 0)
                    y1 = max(int(bbox.ymin * ih), 0)
                    x2 = min(x1 + int(bbox.width * iw), iw)
                    y2 = min(y1 + int(bbox.height * ih), ih)
                    face = frame[y1:y2, x1:x2]
                    if face.size == 0:
                        continue

                    name, score = recognizer.recognize(face)
                    recognizer.registrar_asistencia(name)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route("/")
def index():
    df = pd.read_csv(CSV_PATH) if pd.io.common.file_exists(CSV_PATH) else pd.DataFrame(columns=["Nombre", "Fecha"])
    return render_template("landing.html", attendance_rows=df.to_dict(orient="records"), device_index=DEVICE_INDEX)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start-camera", methods=["POST"])
def start_camera():
    global VIDEO_CAPTURE, DEVICE_INDEX
    VIDEO_CAPTURE = cv2.VideoCapture(CAMERA_URLS[DEVICE_INDEX])
    return redirect(url_for("index"))

@app.route("/next-camera", methods=["POST"])
def next_camera():
    global DEVICE_INDEX, VIDEO_CAPTURE
    if DEVICE_INDEX < len(CAMERA_URLS) - 1:
        DEVICE_INDEX += 1
        if VIDEO_CAPTURE:
            VIDEO_CAPTURE.release()
        VIDEO_CAPTURE = cv2.VideoCapture(CAMERA_URLS[DEVICE_INDEX])
    return redirect(url_for("index"))


@app.route("/prev-camera", methods=["POST"])
def prev_camera():
    global DEVICE_INDEX, VIDEO_CAPTURE
    if DEVICE_INDEX > 0:
        DEVICE_INDEX -= 1
        if VIDEO_CAPTURE:
            VIDEO_CAPTURE.release()
        VIDEO_CAPTURE = cv2.VideoCapture(CAMERA_URLS[DEVICE_INDEX])
    return redirect(url_for("index"))


@app.route("/stop-camera", methods=["POST"])
def stop_camera():
    global VIDEO_CAPTURE
    if VIDEO_CAPTURE:
        VIDEO_CAPTURE.release()
        VIDEO_CAPTURE = None
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
