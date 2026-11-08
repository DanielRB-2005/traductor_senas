# Traductor de lenguaje de señas a texto

Reconocimiento de señas en tiempo real con cámara. El modelo (LSTM) aprende a partir de **videos de señas descargados de internet**: MediaPipe extrae los puntos clave de cuerpo y manos de cada video y la red aprende a clasificar el movimiento.

## Instalación

Requiere Python 3.9 a 3.12.

```
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

## Uso

1. Descarga videos de señas y organízalos en `videos/<sena>/`:
   ```
   videos/
     hola/      video1.mp4  video2.mp4 ...
     gracias/   video1.mp4  ...
   ```
   El nombre de cada carpeta es el texto que mostrará el programa.
2. Extrae los keypoints: `python 1_extraer_keypoints.py`
3. Entrena el modelo: `python 2_entrenar.py`
4. Prueba en tiempo real: `python 3_tiempo_real.py` (`c` limpia el texto, `q` sale)

## Estructura

| Archivo | Función |
|---|---|
| `config.py` | Rutas y parámetros |
| `utils.py` | MediaPipe, extracción y normalización de keypoints |
| `1_extraer_keypoints.py` | Convierte los videos en datos de entrenamiento |
| `2_entrenar.py` | Entrenamiento del modelo LSTM (con aumento de datos) |
| `3_tiempo_real.py` | Reconocimiento en vivo con la cámara |

## Pendientes

- Más señas y más personas en el dataset
- Reconocimiento de oraciones continuas
