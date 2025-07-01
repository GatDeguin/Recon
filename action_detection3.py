import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import argparse
import os
import locale

# Configurar localización en español (Argentina)
locale.setlocale(locale.LC_ALL, 'es_AR.utf8')

# MediaPipe Holistic y Face Mesh setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh

# Parámetros de entrada del modelo
SEQUENCE_LENGTH = 30  # Número de frames por secuencia

# Etiquetas de acciones en español
ACTIONS = ['hola', 'gracias', 'te amo']  # Ajusta según las clases de tu modelo entrenado

# Carga el modelo entrenado desde un archivo .h5
def load_trained_model(model_path):
    model = tf.keras.models.load_model(model_path)
    return model

# Extrae keypoints de un frame usando MediaPipe Holistic, incluyendo visibilidad en pose
# Completa con ceros cuando no se detecta
def extract_keypoints(results):
    # Pose: 33 landmarks (x,y,z,visibility)
    if results.pose_landmarks:
        pose = np.array([
            [res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark
        ]).flatten()
    else:
        pose = np.zeros(33 * 4)
    # Face Mesh: 468 landmarks (x,y,z)
    if results.face_landmarks:
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face = np.zeros(468 * 3)
    # Mano izquierda: 21 landmarks (x,y,z)
    if results.left_hand_landmarks:
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        lh = np.zeros(21 * 3)
    # Mano derecha: 21 landmarks (x,y,z)
    if results.right_hand_landmarks:
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, face, lh, rh])

# Dibuja la acción y su confianza en el frame
def annotate_frame(frame, action, confidence):
    cv2.rectangle(frame, (0, 0), (300, 60), (245, 117, 16), -1)
    texto = f'{action.upper()} {confidence*100:.1f}%'
    cv2.putText(frame, texto,
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    return frame

# Procesa un vídeo o cámara en tiempo real
def process_video(source, model, output_path=None):
    cap = cv2.VideoCapture(source)
    if output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    sequence = []
    with mp_holistic.Holistic(min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convertir color y procesar
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Dibujar Face Mesh: primero tesselación, luego contornos
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=results.face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
                )

            # Dibujar pose y manos
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

            # Extraer y acumular keypoints
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            if len(sequence) > SEQUENCE_LENGTH:
                sequence.pop(0)

            # Predecir acción
            if len(sequence) == SEQUENCE_LENGTH:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                action_idx = np.argmax(res)
                action = ACTIONS[action_idx]
                confidence = res[action_idx]
                # Anotar frame
                frame = annotate_frame(frame, action, confidence)

            # Mostrar y guardar
            cv2.imshow('Detección LSA', frame)
            if output_path:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

# Punto de entrada CLI
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Detección de acciones en tiempo real de LSA (Lengua de Señas Argentina)')
    parser.add_argument('--model',
                        type=str,
                        required=True,
                        help='Ruta al archivo .h5 de tu modelo entrenado')
    parser.add_argument('--video',
                        type=str,
                        required=True,
                        help='Ruta al vídeo o índice de la webcam (por ej. 0)')
    parser.add_argument('--output',
                        type=str,
                        help='Ruta para guardar el vídeo anotado (opcional)')
    args = parser.parse_args()

    # Verifica existencia de modelo y vídeo
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"No se encontró el modelo: {args.model}")
    if not args.video.isdigit() and not os.path.isfile(args.video):
        raise FileNotFoundError(f"No se encontró el vídeo: {args.video}")

    source = int(args.video) if args.video.isdigit() else args.video
    model = load_trained_model(args.model)
    process_video(source, model, args.output)
