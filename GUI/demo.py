from flask import Flask, render_template, Response, request, redirect, url_for
import cv2
import threading

app = Flask(__name__)

camera_streams = {}
camera_locks = {}


def gen_frames(device_index):
    device_index = int(device_index)
    if device_index not in camera_streams:
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {device_index}")
        camera_streams[device_index] = cap
        camera_locks[device_index] = threading.Lock()

    cap = camera_streams[device_index]

    while cap.isOpened():
        with camera_locks[device_index]:
            success, frame = cap.read()
            if not success:
                break
            ret, buffer = cv2.imencode(".jpg", frame)
            frame = buffer.tobytes()
        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return render_template("landing.html", device_index=0)


@app.route("/start-camera", methods=["POST"])
def start_camera():
    device_index = request.form.get("device_index", "0")
    try:
        index = int(device_index)
    except ValueError:
        index = 0
    return render_template("landing.html", device_index=index)


@app.route("/video_feed")
def video_feed():
    device_index = request.args.get("device_index", default=0, type=int)
    return Response(
        gen_frames(device_index), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/stop-camera", methods=["POST"])
def stop_camera():
    for device_index, cap in camera_streams.items():
        with camera_locks[device_index]:
            if cap.isOpened():
                cap.release()
    camera_streams.clear()
    camera_locks.clear()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
