"""Paso 3: reconocimiento de señas en tiempo real con la cámara."""
import json
from collections import deque

import cv2
import numpy as np
from tensorflow.keras.models import load_model

import config
from utils import crear_holistic, deteccion_mediapipe, dibujar_landmarks, extraer_keypoints, muestrear


def main():
    modelo = load_model(config.MODELO_PATH)
    with open(config.ETIQUETAS_PATH, encoding="utf-8") as f:
        etiquetas = json.load(f)

    ventana = deque(maxlen=config.VENTANA_TIEMPO_REAL)   # últimos frames de la cámara
    manos_recientes = deque(maxlen=10)                   # ¿hubo manos en los últimos frames?
    ultimas_preds = deque(maxlen=6)                      # suavizado
    oracion = []
    contador = 0

    cap = cv2.VideoCapture(config.CAMARA_INDEX)
    with crear_holistic() as holistic:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            frame, resultados = deteccion_mediapipe(frame, holistic)
            dibujar_landmarks(frame, resultados)

            kp = extraer_keypoints(resultados)
            ventana.append(kp)
            manos_recientes.append(bool(np.any(kp[132:])))
            contador += 1

            if not any(manos_recientes):
                ultimas_preds.clear()          # sin manos a la vista no se predice nada
            elif len(ventana) >= config.FRAMES_POR_SECUENCIA and contador % 2 == 0:
                # Se resume la ventana a 30 frames, igual que los videos de entrenamiento
                secuencia = muestrear(list(ventana), config.FRAMES_POR_SECUENCIA)
                probs = modelo.predict(np.expand_dims(secuencia, axis=0), verbose=0)[0]
                idx = int(np.argmax(probs))

                if probs[idx] > config.UMBRAL_CONFIANZA:
                    ultimas_preds.append(idx)
                else:
                    ultimas_preds.clear()

                # Se acepta la seña solo si es estable en las últimas predicciones
                if len(ultimas_preds) == ultimas_preds.maxlen and len(set(ultimas_preds)) == 1:
                    palabra = etiquetas[str(idx)]
                    if not oracion or oracion[-1] != palabra:
                        oracion.append(palabra)
                    oracion = oracion[-6:]

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (30, 30, 30), -1)
            cv2.putText(frame, " ".join(oracion), (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Traductor de senas", frame)

            tecla = cv2.waitKey(10) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("c"):       # 'c' limpia el texto
                oracion = []

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
