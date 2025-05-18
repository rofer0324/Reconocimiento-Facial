import os
import cv2
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import mediapipe as mp
from datetime import datetime

confidence_threshold = 91
unknown_threshold = 65

class_names = [
    "Abir Ahmed",
    "Adriana Sanchez",
    "Adriana Solanilla",
    "Amy Olivares",
    "Blas de Leon",
    "Carlos Beitia",
    "Carlos Hernandez",
    "Cesar Rodriguez",
    "Javier Bustamante",
    "Jeremy Sanchez",
    "Jonathan Peralta",
    "Kevin Rodriguez",
    "Mahir Arcia",
    "Michael Jordan",
    "Alejandro Tulipano",
]

model_path = "models/re_fa_15_25e_64b.pth"
base_dir = "results"
os.makedirs(base_dir, exist_ok=True)

fecha_actual = datetime.now().strftime("%Y%m%d_%H%M%S")
csv_file = os.path.join(base_dir, f"asistencia_{fecha_actual}.csv")
pd.DataFrame(columns=["Nombre", "Fecha"]).to_csv(csv_file, index=False)

model = mobilenet_v2(num_classes=15)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, min_detection_confidence=0.15
)


def registrar_asistencia(nombre):
    df = pd.read_csv(csv_file)
    fecha_actual = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    nuevo_registro = pd.DataFrame({"Nombre": [nombre], "Fecha": [fecha_actual]})
    df = pd.concat([df, nuevo_registro], ignore_index=True)
    df.to_csv(csv_file, index=False)


def recognize_and_stream(device_index=0):
    cap = cv2.VideoCapture(device_index)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w_box = int(bboxC.width * iw)
                h_box = int(bboxC.height * ih)

                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(iw, x + w_box), min(ih, y + h_box)
                face = frame[y1:y2, x1:x2]
                if face.size == 0:
                    continue

                face_resized = cv2.resize(face, (224, 224))
                face_tensor = (
                    torch.from_numpy(face_resized).permute(2, 0, 1).unsqueeze(0).float()
                    / 255.0
                )

                with torch.no_grad():
                    outputs = model(face_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    confidence, predicted_class = torch.max(probabilities, 1)
                    confidence = confidence.item() * 100
                    predicted_class = predicted_class.item()

                if confidence > unknown_threshold:
                    person_name = class_names[predicted_class]
                    label = f"{person_name}: {confidence:.2f}%"
                    color = (0, 255, 0)
                    if confidence > confidence_threshold:
                        try:
                            registrar_asistencia(person_name)
                        except Exception as e:
                            print(f"Error al registrar: {e}")
                else:
                    label = "Desconocido"
                    color = (0, 0, 255)

                cv2.putText(
                    frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
                )
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

    cap.release()
    postprocess_csv(csv_file)


def postprocess_csv(csv_file):
    try:
        df = pd.read_csv(csv_file)
        df["Fecha"] = pd.to_datetime(df["Fecha"])
        df = df.sort_values(by=["Nombre", "Fecha"])

        df_processed = pd.DataFrame(columns=["Nombre", "Fecha"])
        for nombre in df["Nombre"].unique():
            df_nombre = df[df["Nombre"] == nombre]
            df_processed = pd.concat([df_processed, df_nombre.iloc[[0, -1]]])
        df_processed.to_csv(csv_file, index=False)
    except Exception as e:
        print(f"Error al procesar CSV: {e}")


def stop_all_cameras():
    pass  # Placeholder for multi-camera cleanup if needed


def get_latest_csv():
    csv_files = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
    if not csv_files:
        return None
    csv_files = sorted(
        csv_files,
        key=lambda f: os.path.getmtime(os.path.join(base_dir, f)),
        reverse=True,
    )
    return os.path.join(base_dir, csv_files[0])
