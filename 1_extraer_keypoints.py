"""Paso 1: convierte los videos descargados en datos de entrenamiento.

Estructura esperada:
    videos/
      hola/      video1.mp4  video2.mp4 ...
      gracias/   video1.mp4  ...

Cada video se procesa con MediaPipe, se resume en FRAMES_POR_SECUENCIA frames
y se guarda como un archivo .npy en data/<sena>/<nombre_del_video>.npy.
Si ya existe el .npy de un video, se omite (puedes agregar videos y volver a ejecutar).
"""
import os
import numpy as np

import config
from utils import keypoints_de_video, muestrear


def main():
    if not os.path.isdir(config.VIDEOS_DIR):
        print(f"No existe la carpeta '{config.VIDEOS_DIR}'.")
        print("Créala y dentro pon una subcarpeta por seña, con sus videos. Ejemplo:")
        print(f"  {config.VIDEOS_DIR}/hola/video1.mp4")
        return

    senas = sorted(d for d in os.listdir(config.VIDEOS_DIR)
                   if os.path.isdir(os.path.join(config.VIDEOS_DIR, d)))
    if not senas:
        print(f"No hay subcarpetas dentro de '{config.VIDEOS_DIR}'.")
        return

    procesados, omitidos = 0, 0
    for sena in senas:
        carpeta_in = os.path.join(config.VIDEOS_DIR, sena)
        carpeta_out = os.path.join(config.DATA_DIR, sena)
        os.makedirs(carpeta_out, exist_ok=True)

        videos = sorted(f for f in os.listdir(carpeta_in)
                        if f.lower().endswith(config.EXTENSIONES_VIDEO))
        print(f"\n[{sena}] {len(videos)} videos")

        for nombre in videos:
            salida = os.path.join(carpeta_out, os.path.splitext(nombre)[0] + ".npy")
            if os.path.exists(salida):
                continue

            frames = keypoints_de_video(os.path.join(carpeta_in, nombre))
            if len(frames) < 5:
                print(f"  - {nombre}: omitido (video vacío o ilegible)")
                omitidos += 1
                continue

            # Descarta videos donde casi nunca se detectan las manos
            con_manos = sum(1 for kp in frames if np.any(kp[132:]))
            if con_manos / len(frames) < 0.3:
                print(f"  - {nombre}: omitido (se detectaron manos en pocos frames)")
                omitidos += 1
                continue

            np.save(salida, muestrear(frames, config.FRAMES_POR_SECUENCIA))
            procesados += 1
            print(f"  + {nombre}: ok ({len(frames)} frames)")

    print(f"\nListo. Procesados: {procesados}  |  Omitidos: {omitidos}")
    print(f"Los datos están en la carpeta '{config.DATA_DIR}'. Siguiente paso: python 2_entrenar.py")


if __name__ == "__main__":
    main()
