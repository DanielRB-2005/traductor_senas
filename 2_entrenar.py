"""Paso 2: entrena el modelo (LSTM) con los keypoints extraídos de los videos."""
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import config
from utils import N_FEATURES


def cargar_dataset():
    senas = sorted(d for d in os.listdir(config.DATA_DIR)
                   if os.path.isdir(os.path.join(config.DATA_DIR, d)))
    X, y = [], []
    for i, sena in enumerate(senas):
        carpeta = os.path.join(config.DATA_DIR, sena)
        for archivo in sorted(os.listdir(carpeta)):
            if archivo.endswith(".npy"):
                arr = np.load(os.path.join(carpeta, archivo))
                if arr.shape == (config.FRAMES_POR_SECUENCIA, N_FEATURES):
                    X.append(arr)
                    y.append(i)
    return np.array(X, dtype=np.float32), np.array(y), senas


def aumentar(X, y, copias=4, semilla=0):
    """Genera variaciones (escala y ruido leve) para compensar pocos videos por seña."""
    rng = np.random.default_rng(semilla)
    lista_X, lista_y = [X], [y]
    for _ in range(copias):
        escala = rng.uniform(0.9, 1.1, size=(len(X), 1, 1)).astype(np.float32)
        ruido = rng.normal(0, 0.01, size=X.shape).astype(np.float32)
        Xa = np.where(X != 0, X * escala + ruido, 0).astype(np.float32)
        Xa[:, :, 3:132:4] = X[:, :, 3:132:4]      # la visibilidad de la pose no se altera
        lista_X.append(Xa)
        lista_y.append(y)
    return np.concatenate(lista_X), np.concatenate(lista_y)


def main():
    if not os.path.isdir(config.DATA_DIR):
        print("No hay datos. Ejecuta primero: python 1_extraer_keypoints.py")
        return

    X, y, senas = cargar_dataset()
    if len(X) == 0:
        print("No se encontraron datos válidos. Ejecuta primero: python 1_extraer_keypoints.py")
        return

    print("Señas encontradas:")
    for i, s in enumerate(senas):
        print(f"  {s}: {int((y == i).sum())} videos")

    # Se ignoran las carpetas con menos de 2 videos (vacías o ajenas al proyecto)
    validas = [i for i in range(len(senas)) if int((y == i).sum()) >= 2]
    ignoradas = [senas[i] for i in range(len(senas)) if i not in validas]
    if ignoradas:
        print("\nSe ignoran por tener menos de 2 videos:", ignoradas)
    if len(validas) < 2:
        print("Necesitas al menos 2 señas con 2 o más videos cada una (idealmente 20 o más).")
        return
    mascara = np.isin(y, validas)
    mapa = {viejo: nuevo for nuevo, viejo in enumerate(validas)}
    X = X[mascara]
    y = np.array([mapa[v] for v in y[mascara]])
    senas = [senas[i] for i in validas]
    print("\nSe entrenará con:", senas)

    n_test = max(int(0.2 * len(X)), len(senas))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_test, random_state=42, stratify=y
    )

    # El aumento de datos se aplica solo al conjunto de entrenamiento
    X_train, y_train = aumentar(X_train, y_train)
    y_train_cat = to_categorical(y_train, num_classes=len(senas))
    y_test_cat = to_categorical(y_test, num_classes=len(senas))
    print("Entrenamiento:", X_train.shape, "| Prueba:", X_test.shape)

    modelo = Sequential([
        Input(shape=(config.FRAMES_POR_SECUENCIA, N_FEATURES)),
        LSTM(64, return_sequences=True, activation="tanh"),
        Dropout(0.3),
        LSTM(128, return_sequences=False, activation="tanh"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dense(len(senas), activation="softmax"),
    ])
    modelo.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"])

    modelo.fit(
        X_train, y_train_cat,
        validation_data=(X_test, y_test_cat),
        epochs=150, batch_size=16,
        callbacks=[EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True)],
    )

    loss, acc = modelo.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nPrecisión en test: {acc:.2%}")

    modelo.save(config.MODELO_PATH)
    with open(config.ETIQUETAS_PATH, "w", encoding="utf-8") as f:
        json.dump({str(i): s for i, s in enumerate(senas)}, f, ensure_ascii=False)
    print("Modelo guardado en", config.MODELO_PATH)


if __name__ == "__main__":
    main()
