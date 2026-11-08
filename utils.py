import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

N_FEATURES = 258  # pose (33*4) + mano izq (21*3) + mano der (21*3)


def crear_holistic():
    return mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)


def deteccion_mediapipe(frame_bgr, modelo):
    """Procesa un frame y devuelve (frame, resultados)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    resultados = modelo.process(rgb)
    return frame_bgr, resultados


def dibujar_landmarks(frame, resultados):
    mp_drawing.draw_landmarks(frame, resultados.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, resultados.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(frame, resultados.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)


def extraer_keypoints(resultados):
    """Vector de 258 valores por frame, normalizado respecto a los hombros.

    Normalizar hace que el modelo no dependa de dónde está parada la persona
    ni de qué tan cerca esté de la cámara. Esto es clave cuando entrenas con
    videos de internet, que tienen encuadres muy distintos.
    """
    pose = (
        np.array([[p.x, p.y, p.z, p.visibility] for p in resultados.pose_landmarks.landmark])
        if resultados.pose_landmarks else None
    )
    mano_izq = (
        np.array([[p.x, p.y, p.z] for p in resultados.left_hand_landmarks.landmark])
        if resultados.left_hand_landmarks else None
    )
    mano_der = (
        np.array([[p.x, p.y, p.z] for p in resultados.right_hand_landmarks.landmark])
        if resultados.right_hand_landmarks else None
    )

    # Referencia: centro de los hombros (puntos 11 y 12) y ancho de hombros
    centro, escala = np.zeros(3), 1.0
    if pose is not None:
        hombro_i, hombro_d = pose[11, :3], pose[12, :3]
        centro = (hombro_i + hombro_d) / 2
        ancho = np.linalg.norm(hombro_i[:2] - hombro_d[:2])
        escala = ancho if ancho > 1e-6 else 1.0

    if pose is not None:
        pose_n = np.concatenate([(pose[:, :3] - centro) / escala, pose[:, 3:4]], axis=1).flatten()
    else:
        pose_n = np.zeros(33 * 4)

    izq_n = ((mano_izq - centro) / escala).flatten() if mano_izq is not None else np.zeros(21 * 3)
    der_n = ((mano_der - centro) / escala).flatten() if mano_der is not None else np.zeros(21 * 3)

    return np.concatenate([pose_n, izq_n, der_n]).astype(np.float32)


def muestrear(secuencia, n):
    """Reduce (o estira) una lista de frames a exactamente n frames repartidos en el tiempo."""
    total = len(secuencia)
    indices = np.linspace(0, total - 1, n).round().astype(int)
    return np.array([secuencia[i] for i in indices], dtype=np.float32)


def keypoints_de_video(ruta):
    """Lee un video completo y devuelve la lista de keypoints de cada frame."""
    cap = cv2.VideoCapture(ruta)
    frames = []
    with crear_holistic() as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            _, resultados = deteccion_mediapipe(frame, holistic)
            frames.append(extraer_keypoints(resultados))
    cap.release()
    return frames
