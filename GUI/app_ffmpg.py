from flask import Flask, render_template, Response, request, redirect, url_for
import subprocess
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from recognizer import FaceRecognizer

app = Flask(__name__)

# Configuración de cámaras
CAMERA_URLS = {
    0: "rtsp://channels/1",
    1: "rtsp://channels/2",
    2: "rtsp://channels/3",
    3: "rtsp://channels/4"
}
DEVICE_INDEX = 0
FFMPEG_PROCESS = None
CSV_PATH = "out/asistencia.csv"

# Inicializar el reconocedor facial
recognizer = FaceRecognizer(
    checkpoint="checkpoints/model_epoch10.pth",
    emb_path="embeddings/embeddings.npy",
    paths_path="embeddings/paths.npy",
    threshold=0.9,
    csv_path=CSV_PATH
)

mp_face_detection = mp.solutions.face_detection

def open_ffmpeg_stream(url):
    return subprocess.Popen(
        [
            "ffmpeg",
            "-rtsp_transport", "tcp",
            "-i", url,
            "-f", "image2pipe",
            "-pix_fmt", "bgr24",
            "-vcodec", "rawvideo",
            "-"
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )

def generate_frames_ffmpeg(process, width=1920, height=1080):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.8) as detector:
        while True:
            raw_image = process.stdout.read(width * height * 3)
            if not raw_image:
                break

            frame = np.frombuffer(raw_image, dtype=np.uint8).reshape((height, width, 3))

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
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/")
def index():
    df = pd.read_csv(CSV_PATH) if pd.io.common.file_exists(CSV_PATH) else pd.DataFrame(columns=["Nombre", "Fecha"])
    return render_template("landing.html", attendance_rows=df.to_dict(orient="records"), device_index=DEVICE_INDEX)

@app.route("/video_feed")
def video_feed():
    global FFMPEG_PROCESS
    if FFMPEG_PROCESS is None:
        return "Cámara no iniciada", 400
    return Response(generate_frames_ffmpeg(FFMPEG_PROCESS), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/start-camera", methods=["POST"])
def start_camera():
    global FFMPEG_PROCESS, DEVICE_INDEX
    if FFMPEG_PROCESS:
        FFMPEG_PROCESS.kill()
    url = CAMERA_URLS[DEVICE_INDEX]
    FFMPEG_PROCESS = open_ffmpeg_stream(url)
    return redirect(url_for("index"))

@app.route("/next-camera", methods=["POST"])
def next_camera():
    global DEVICE_INDEX, FFMPEG_PROCESS
    if DEVICE_INDEX < len(CAMERA_URLS) - 1:
        DEVICE_INDEX += 1
        if FFMPEG_PROCESS:
            FFMPEG_PROCESS.kill()
        FFMPEG_PROCESS = open_ffmpeg_stream(CAMERA_URLS[DEVICE_INDEX])
    return redirect(url_for("index"))

@app.route("/prev-camera", methods=["POST"])
def prev_camera():
    global DEVICE_INDEX, FFMPEG_PROCESS
    if DEVICE_INDEX > 0:
        DEVICE_INDEX -= 1
        if FFMPEG_PROCESS:
            FFMPEG_PROCESS.kill()
        FFMPEG_PROCESS = open_ffmpeg_stream(CAMERA_URLS[DEVICE_INDEX])
    return redirect(url_for("index"))

@app.route("/stop-camera", methods=["POST"])
def stop_camera():
    global FFMPEG_PROCESS
    if FFMPEG_PROCESS:
        FFMPEG_PROCESS.kill()
        FFMPEG_PROCESS = None
    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
