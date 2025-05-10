import cv2
import os
import random
import mediapipe as mp

def create_directories(base_path, person_name):
    # Crear las carpetas de train y validation
    preprocessed_path = os.path.join(base_path, 'preprocessed')
    train_dir = os.path.join(preprocessed_path, 'train', person_name)
    test_dir = os.path.join(preprocessed_path, 'test', person_name)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    print(f'Carpetas creadas: {train_dir} - {test_dir}')
    return train_dir, test_dir

def capture_faces_mediapipe(video_path, max_images):
    cap = cv2.VideoCapture(video_path)
    count = 0
    captured_faces = []
    
    # MediaPipe setup
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.15)

    while True:
        ret, frame = cap.read()
        if not ret or count >= max_images:
            break

        # Procesamiento...
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(frame_rgb)

        if results.detections:
            for detection in results.detections:
                # Extracción de rostro...
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x = int(bboxC.xmin * iw)
                y = int(bboxC.ymin * ih)
                w_box = int(bboxC.width * iw)
                h_box = int(bboxC.height * ih)

                face = frame[y:y+h_box, x:x+w_box]
                if face.size == 0:
                    continue

                face = cv2.resize(face, (224, 224), interpolation=cv2.INTER_CUBIC)
                captured_faces.append(face)
                count += 1

                if count >= max_images:
                    break

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return captured_faces

def split_dataset(captured_faces, train_dir, test_dir):
    total_images = len(captured_faces)
    
    if total_images == 0:
        print("No hay imágenes para dividir.")
        return

    # Mezclar aleatoriamente las imágenes
    random.shuffle(captured_faces)

    # Dividir el conjunto en 80% para entrenamiento y 20% para test
    split_index = int(total_images * 0.8)
    train_faces = captured_faces[:split_index]
    test_faces = captured_faces[split_index:]
    
    # Saber cuántas imágenes ya existen en las carpetas
    crops_train = len(os.listdir(train_dir))
    crops_test = len(os.listdir(test_dir))


    # Guardar las imágenes en las carpetas correspondientes SIN sobreescribir
    for i, face in enumerate(train_faces):
        filename = f'face_{crops_train + i}.jpg'
        cv2.imwrite(os.path.join(train_dir, filename), face)

    for i, face in enumerate(test_faces):
        filename = f'face_{crops_test + i}.jpg'
        cv2.imwrite(os.path.join(test_dir, filename), face)

    print('Distribución de imágenes completada.')
    
def main():
    # Diccionario de personas y sus respectivos videos
    personas_videos = {
        'Abir Ahmed': ['Abir1.mp4', 'Abir2.mp4'],
        'Adriana Sanchez': ['AdriSa1.mp4', 'AdriSa2.mp4'],
        'Adriana Solanilla': ['AdriSo1.mp4', 'AdriSo2.mp4'],
        'Amy Olivares': ['Amy1.mp4', 'Amy2.mp4'],
        'Blas de Leon': ['Blas1.mp4', 'Blas2.mp4'],
        'Carlos Beitia': ['CarlosB1.mp4', 'CarlosB2.mp4'],
        'Carlos Hernandez': ['CarlosH1.mp4', 'CarlosH2.mp4'],
        'Cesar Rodriguez': ['Cesar1.mp4', 'Cesar2.mp4'],
        #'David Rodriguez': ['David1.mp4', 'David2.mp4'],
        'Javier Bustamante': ['Javier1.mp4', 'Javier2.mp4'],
        'Jeremy Sanchez': ['Jeremy1.mp4', 'Jeremy2.mp4'],
        'Jonathan Peralta': ['Jonathan1.mp4', 'Jonathan2.mp4'],
        'Kevin Rodriguez': ['Kevin1.mp4', 'Kevin2.mp4'],
        'Mahir Arcia': ['Mahir1.mp4', 'Mahir2.mp4'],
        'Michael Jordan': ['Michael1.mp4', 'Michael2.mp4'],
        #'Maria Donadio': ['Teresa1.mp4', 'Teresa2.mp4']
        'Alejandro Tulipano': ['Tulipano1.mp4', 'Tulipano2.mp4'],
    }
    
    # Ruta base 
    base_path = '../data'  
    
    MAX_IMAGES_PER_PERSON = 300

    # Iterar por cada persona
    for person_name, videos in personas_videos.items():
        print(f"\nProcesando a {person_name}...")

        train_dir, test_dir = create_directories(base_path, person_name)
        total_captured_for_person = 0

        for video_name in videos:
            video_path = os.path.join(base_path, 'crudo', video_name)
            if not os.path.exists(video_path):
                print(f"VIDEO NO ENCONTRADO !!: {video_path}")
                continue

            print(f"Procesando video...: {video_name}")
            remaining_images = MAX_IMAGES_PER_PERSON - total_captured_for_person
            if remaining_images <= 0:
                break

            # Captura de rostros
            captured_faces = capture_faces_mediapipe(video_path, remaining_images)
            total_captured_for_person += len(captured_faces)

            if captured_faces:
                print(f'Se han capturado {len(captured_faces)} rostros del video {video_name}.')
                split_dataset(captured_faces, train_dir, test_dir)
            else:
                print(f'No se capturaron rostros en {video_name}.')

if __name__ == "__main__":
    main()