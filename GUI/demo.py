from flask import Flask, render_template, Response, request, redirect, url_for
import os
import pandas as pd
from face_recognition import recognize_and_stream, stop_all_cameras, get_latest_csv

app = Flask(__name__)


@app.route("/")
def index():
    csv_file = get_latest_csv()
    rows = []
    if csv_file and os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        rows = df.to_dict(orient="records")
    return render_template("landing.html", device_index=0, attendance_rows=rows)


@app.route("/start-camera", methods=["POST"])
def start_camera():
    device_index = request.form.get("device_index", "0")
    try:
        index = int(device_index)
    except ValueError:
        index = 0
    return redirect(url_for("index", device_index=index))


@app.route("/video_feed")
def video_feed():
    device_index = request.args.get("device_index", default=0, type=int)
    return Response(
        recognize_and_stream(device_index),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stop-camera", methods=["POST"])
def stop_camera():
    stop_all_cameras()
    return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(debug=True)
