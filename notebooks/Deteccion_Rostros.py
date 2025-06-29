import os
import cv2
import random
import mediapipe as mp
from tqdm import tqdm

# Parámetros configurables
MAX_IMAGES_PER_PERSON = 200
TRAIN_SPLIT = 0.8
FRAME_SKIP = 5
OUTPUT_IMAGE_SIZE = (224, 224)

def create_directories(base_path, person_name):
    train_dir = os.path.join(base_path, 'preprocessed', 'train', person_name)
    test_dir = os.path.join(base_path, 'preprocessed', 'test', person_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    return train_dir, test_dir

def capture_faces_from_video(video_path, max_images, frame_skip=FRAME_SKIP):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"No se pudo abrir el video: {video_path}")
        return []

    count = 0
    faces = []
    frame_counter = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc=f"Extrayendo {os.path.basename(video_path)}", unit="frame")

    mp_fd = mp.solutions.face_detection
    detector = mp_fd.FaceDetection(model_selection=1, min_detection_confidence=0.15)

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        progress_bar.update(1)

        frame_counter += 1
        if frame_counter % frame_skip != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y = int(bbox.xmin * iw), int(bbox.ymin * ih)
                w, h = int(bbox.width * iw), int(bbox.height * ih)

                # Validar rangos
                if x < 0 or y < 0 or w <= 0 or h <= 0:
                    continue
                if y + h > ih or x + w > iw:
                    continue

                face = frame[y:y+h, x:x+w]
                if face.size == 0:
                    continue

                face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face_resized = cv2.resize(face_gray, OUTPUT_IMAGE_SIZE)

                faces.append(face_resized)
                count += 1

                # Mostrar progreso en la ventana
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f'Capturas: {count}/{max_images}', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if count >= max_images:
                    break

        #cv2.imshow("Captura de rostros", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Interrupción manual con Q.")
            break

    progress_bar.close()
    cap.release()
    cv2.destroyAllWindows()
    return faces

def get_total_faces_from_videos(videos, base_video_path, max_images, frame_skip=FRAME_SKIP):
    all_faces = []
    per_video_quota = max_images // len(videos)

    for video in videos:
        path = os.path.join(base_video_path, video)
        if not os.path.exists(path):
            print(f"Video no encontrado: {path}")
            continue

        faces = capture_faces_from_video(path, per_video_quota, frame_skip=frame_skip)
        all_faces.extend(faces)

        if len(all_faces) >= max_images:
            break

    return all_faces[:max_images]

def split_dataset(faces, train_dir, test_dir, train_split=TRAIN_SPLIT):
    random.shuffle(faces)
    split_idx = int(len(faces) * train_split)
    train_faces = faces[:split_idx]
    test_faces = faces[split_idx:]

    train_count = len(os.listdir(train_dir))
    test_count = len(os.listdir(test_dir))

    print("\nGuardando imágenes de entrenamiento...")
    for i, face in enumerate(tqdm(train_faces, desc="Train set", unit="img")):
        filename = f'face_{train_count + i + 1}.jpg'
        cv2.imwrite(os.path.join(train_dir, filename), face)

    print("\nGuardando imágenes de prueba...")
    for i, face in enumerate(tqdm(test_faces, desc="Test set", unit="img")):
        filename = f'face_{test_count + i + 1}.jpg'
        cv2.imwrite(os.path.join(test_dir, filename), face)

def iterate_persons(person_dict, base_path, base_video_path):
    for person_name in tqdm(person_dict, desc="Procesando personas", unit="persona"):
        print(f"\n\nPersona: {person_name}")
        train_dir, test_dir = create_directories(base_path, person_name)

        faces = get_total_faces_from_videos(
            videos=person_dict[person_name],
            base_video_path=base_video_path,
            max_images=MAX_IMAGES_PER_PERSON,
            frame_skip=FRAME_SKIP
        )
        print(f"\nTotal rostros capturados: {len(faces)}")

        if faces:
            split_dataset(faces, train_dir, test_dir)
            print(f"\nDivisión completada: {len(faces)} imágenes en train/test")
        else:
            print("No se capturaron rostros.")

def main():
    personas_videos = {
        'Abir Ahmed': ['Abir1.mp4', 'Abir2.mp4'],
        'Adriana Sanchez': ['AdriSa1.mp4', 'AdriSa2.mp4'],
        'Adriana Solanilla': ['AdriSo1.mp4', 'AdriSo2.mp4'],
        'Amy Olivares': ['Amy1.mp4', 'Amy2.mp4'],
        'Blas de Leon': ['Blas1.mp4', 'Blas2.mp4'],
        'Carlos Beitia': ['CarlosB1.mp4', 'CarlosB2.mp4'],
        'Carlos Hernandez': ['CarlosH1.mp4', 'CarlosH2.mp4'],
        'Cesar Rodriguez': ['Cesar1.mp4', 'Cesar2.mp4'],
        'Javier Bustamante': ['Javier1.mp4', 'Javier2.mp4'],
        'Jeremy Sanchez': ['Jeremy1.mp4', 'Jeremy2.mp4'],
        'Jonathan Peralta': ['Jonathan1.mp4', 'Jonathan2.mp4'],
        'Kevin Rodriguez': ['Kevin1.mp4', 'Kevin2.mp4'],
        'Mahir Arcia': ['Mahir1.mp4', 'Mahir2.mp4'],
        'Michael Jordan': ['Michael1.mp4', 'Michael2.mp4'],
        'Alejandro Tulipano': ['Tulipano1.mp4', 'Tulipano2.mp4'],
    }

    base_path = '../data'
    base_video_path = os.path.join(base_path, 'crudo')

    iterate_persons(personas_videos, base_path, base_video_path)

if __name__ == "__main__":
    main()
