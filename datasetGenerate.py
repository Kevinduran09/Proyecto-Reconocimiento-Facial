import cv2
import os
import time
import random

# Configuración
dataset_dir = 'dataset'  # Directorio donde se guardarán las imágenes
num_images_per_person = 100  # Número de imágenes por persona
image_size = (200, 200)  # Tamaño de las imágenes capturadas

# Asegurarse de que el directorio de datos exista
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Inicializar la cámara
video_capture = cv2.VideoCapture(0)
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def augment_image(image):
    """Genera variaciones de la imagen."""
    augmented_images = []

    # Volteo horizontal
    flipped_image = cv2.flip(image, 1)
    augmented_images.append(flipped_image)

    # Rotación aleatoria
    angle = random.choice([-10, 0, 10])  # Rango de rotación
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    augmented_images.append(rotated_image)

    return augmented_images


def capture_images(person_name):
    """Captura imágenes de una persona."""
    person_dir = os.path.join(dataset_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)

    print(f"Capturando imágenes de {person_name}. Presiona 'q' para salir.")
    count = 0

    while count < num_images_per_person:
        ret, frame = video_capture.read()
        if not ret:
            print("No se pudo capturar el video. Saliendo...")
            break

        # Detectar rostros
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray_frame, 1.1, 5)

        # Si se detecta un rostro, se guarda
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                # Extraer el rostro
                face = frame[y:y + h, x:x + w]
                face_resized = cv2.resize(face, image_size)

                # Guardar la imagen original
                img_name = f"{person_name}_{count}.jpg"
                cv2.imwrite(os.path.join(person_dir, img_name), face_resized)

                # Guardar imágenes aumentadas
                for augmented_face in augment_image(face_resized):
                    augmented_name = f"{person_name}_aug_{count}.jpg"
                    cv2.imwrite(os.path.join(
                        person_dir, augmented_name), augmented_face)

                count += 1
                print(f"Imagen {count}/{num_images_per_person} capturada.")

        # Mostrar el frame
        cv2.imshow("Capturando Imágenes", frame)

        # Salir del bucle si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    print(f"Se han capturado {count} imágenes de {person_name}.")


try:
    person_name = input("Introduce el nombre de la persona (ejemplo: kevin): ")
    capture_images(person_name)
finally:
    video_capture.release()
    cv2.destroyAllWindows()
