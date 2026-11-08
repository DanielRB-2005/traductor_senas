# Configuración central del proyecto

# Carpeta con los videos descargados, organizados así:
#   videos/hola/video1.mp4, videos/hola/video2.mp4, ...
#   videos/gracias/video1.mp4, ...
# El nombre de cada subcarpeta es el texto que el modelo aprenderá a reconocer.
VIDEOS_DIR = "videos"

# Aquí se guardan los keypoints extraídos de cada video (los crea el paso 1)
DATA_DIR = "data"

EXTENSIONES_VIDEO = (".mp4", ".avi", ".mov", ".mkv", ".webm")

FRAMES_POR_SECUENCIA = 30         # cada video se resume en 30 frames repartidos en el tiempo

MODELO_PATH = "modelo_senas.keras"
ETIQUETAS_PATH = "etiquetas.json"

# Tiempo real
VENTANA_TIEMPO_REAL = 45          # frames de la cámara que se analizan (~1.5 s); ajusta según la duración de tus señas
UMBRAL_CONFIANZA = 0.80           # mínimo para aceptar una predicción
CAMARA_INDEX = 0
