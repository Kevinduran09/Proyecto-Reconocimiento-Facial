import cv2
import os
import time

# Cargar el clasificador de rostros
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Crear una carpeta para guardar las imágenes si no existe


def crear_carpeta_imagenes(path="imagenes_rostros"):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

# Función para detectar rostros y guardar las imágenes


def detect_bounding_box(vid, save_path):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(
        gray_image, 1.1, 5, minSize=(40, 40))

    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recortar la región de la cara detectada
        face_image = vid[y:y+h, x:x+w]

        # Guardar la imagen con un nombre único
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        face_filename = f"{save_path}/rostro_{timestamp}.jpg"
        cv2.imwrite(face_filename, face_image)
        print(f"Rostro guardado en: {face_filename}")

    return faces

# Función para intentar abrir la cámara


def try_open_camera(camera_index=0):
    video_capture = cv2.VideoCapture(camera_index)
    if not video_capture.isOpened():
        return None
    return video_capture
