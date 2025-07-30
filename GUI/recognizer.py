import os
import numpy as np
import pandas as pd
import torch
from torchvision import transforms
from PIL import Image
from datetime import datetime, timedelta
from models.resnet_arcface import ResNetArcModel
from utils.utils import get_device

class FaceRecognizer:
    def __init__(self, checkpoint, emb_path, paths_path, threshold=0.6, csv_path="out/asistencia.csv"):
        self.device = get_device()
        self.threshold = threshold
        self.csv_path = csv_path

        # Modelo
        self.model = ResNetArcModel(num_classes=17, embedding_size=512).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=self.device))
        self.model.eval()

        # Embeddings y etiquetas
        self.known_embeddings = np.load(emb_path)
        self.known_embeddings = self.known_embeddings / np.linalg.norm(self.known_embeddings, axis=1, keepdims=True)
        self.known_paths = np.load(paths_path)
        self.known_labels = [p.split("/")[-1].split(".")[0] for p in self.known_paths]

        # Transformación
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def recognize(self, face_bgr):
        face_pil = Image.fromarray(face_bgr[:, :, ::-1])
        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(input_tensor).cpu().numpy()
            embedding = embedding / np.linalg.norm(embedding)

        similarities = np.dot(self.known_embeddings, embedding.T).squeeze()
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]
        name = "Desconocido"
        if best_score >= self.threshold:
            name = self.known_labels[best_idx]
        return name, best_score

    def registrar_asistencia(self, nombre, intervalo_minutos=60):
        if nombre == "Desconocido":
            return

        ahora = datetime.now()
        ahora_str = ahora.strftime("%Y-%m-%d %H:%M:%S")

        if not os.path.exists(self.csv_path):
            pd.DataFrame(columns=["Nombre", "Fecha"]).to_csv(self.csv_path, index=False)

        df = pd.read_csv(self.csv_path)
        if not df.empty:
            df["Fecha"] = pd.to_datetime(df["Fecha"])
            recientes = df[(df["Nombre"] == nombre) &
                           (df["Fecha"] > ahora - timedelta(minutes=intervalo_minutos))]
            if not recientes.empty:
                return

        df.loc[len(df)] = [nombre, ahora_str]
        df.to_csv(self.csv_path, index=False)
