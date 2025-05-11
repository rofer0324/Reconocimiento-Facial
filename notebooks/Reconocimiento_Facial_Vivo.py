import os
import cv2
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v2
import mediapipe as mp
from datetime import datetime

# Configuraciones
confidence_threshold = 91  # Umbral de confianza en porcentaje
unknown_threshold = 65     # Umbral para considerar una persona como desconocida

# Cargar el modelo entrenado
model_path = '../models/re_fa_15_personas_v5.pth'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"El modelo no se encontró en la ruta: {model_path}")

# Recrear la arquitectura exactamente como la entrenaste
model = mobilenet_v2(num_classes=15)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Inicializar MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.15)

# Crear un nombre único para el archivo CSV con la fecha y un número incremental
base_dir = '../results'
os.makedirs(base_dir, exist_ok=True)

# Obtener fecha actual
fecha_actual = datetime.now().strftime('%Y%m%d_%H%M%S')

# Contar archivos previos con nombre que empiece por 'asistencia'
archivos_previos = [f for f in os.listdir(base_dir) if f.startswith('asistencia')]
numero = len(archivos_previos) + 1

# Crear el nuevo nombre de archivo
csv_file = os.path.join(base_dir, f'asistencia{numero}_{fecha_actual}.csv')

# Crear el archivo .csv con encabezado si no existe
pd.DataFrame(columns=["Nombre", "Fecha"]).to_csv(csv_file, index=False)

# Definir la función para registrar la asistencia
def registrar_asistencia(nombre):
    df = pd.read_csv(csv_file)
    fecha_actual = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    nuevo_registro = pd.DataFrame({"Nombre": [nombre], "Fecha": [fecha_actual]})
    df = pd.concat([df, nuevo_registro], ignore_index=True)
    df.to_csv(csv_file, index=False)

# Lista de nombres de las 15 personas
class_names = [
    'Abir Ahmed',
    'Adriana Sanchez',
    'Adriana Solanilla',
    'Amy Olivares',
    'Blas de Leon',
    'Carlos Beitia',
    'Carlos Hernandez',
    'Cesar Rodriguez',
    'Javier Bustamante',
    'Jeremy Sanchez',
    'Jonathan Peralta',
    'Kevin Rodriguez',
    'Mahir Arcia',
    'Michael Jordan',
    'Alejandro Tulipano'
]

# Inicializar la cámara web
cap = cv2.VideoCapture(0)

# Bucle principal para la detección y reconocimiento en tiempo real
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir el frame a RGB
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

            # Asegurarse de que las coordenadas estén dentro de los límites de la imagen
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(iw, x + w_box)
            y2 = min(ih, y + h_box)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # Preprocesar el rostro
            face_resized = cv2.resize(face, (224, 224))
            face_tensor = torch.from_numpy(face_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # Realizar predicción
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
                        print(f"Error al registrar la asistencia: {e}")
            else:
                label = "Desconocido"
                color = (0, 0, 255)

            # Mostrar el nombre y la precisión sobre el rostro en el video
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Mostrar el video en tiempo real
    cv2.imshow('Reconocimiento Facial en Tiempo Real', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()

# Procesamiento del archivo CSV para quedarse con la primera y última hora registrada de cada persona
df = pd.read_csv(csv_file)
df['Fecha'] = pd.to_datetime(df['Fecha'])
df = df.sort_values(by=['Nombre', 'Fecha'])

df_processed = pd.DataFrame(columns=["Nombre", "Fecha"])

for nombre in df['Nombre'].unique():
    df_nombre = df[df['Nombre'] == nombre]
    primera_fecha = df_nombre.iloc[0]
    ultima_fecha = df_nombre.iloc[-1]
    df_processed = pd.concat([df_processed, pd.DataFrame([primera_fecha])])
    if not primera_fecha.equals(ultima_fecha):
        df_processed = pd.concat([df_processed, pd.DataFrame([ultima_fecha])])

df_processed.to_csv(csv_file, index=False)