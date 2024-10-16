import cv2
import os
import numpy as np
from sklearn.utils import shuffle


class FacialRecognition:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.label_dict = {}

    def augment_image(self, image):
        """Aplica aumentos de datos a la imagen (rotación y cambio de brillo)."""
        # Rotación
        angle = np.random.uniform(-15, 15)  # Rango de rotación
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (width, height))

        # Cambio de brillo
        brightness_factor = np.random.uniform(0.5, 1.5)  # Factor de brillo
        bright_image = cv2.convertScaleAbs(
            rotated_image, alpha=brightness_factor)

        return bright_image

    def train_recognizer(self):
        faces = []
        labels = []
        current_label = 0

        for person_name in os.listdir(self.dataset_path):
            person_path = os.path.join(self.dataset_path, person_name)
            if os.path.isdir(person_path):
                self.label_dict[current_label] = person_name
                for image_name in os.listdir(person_path):
                    image_path = os.path.join(person_path, image_name)
                    image = cv2.imread(image_path)
                    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    # Redimensionar imagen
                    gray_image = cv2.resize(gray_image, (200, 200))
                    faces.append(gray_image)
                    labels.append(current_label)

                    # Aumento de datos
                    for _ in range(3):  # Crear 3 imágenes aumentadas por cada imagen
                        augmented_image = self.augment_image(gray_image)
                        faces.append(augmented_image)
                        labels.append(current_label)

                current_label += 1

        faces, labels = shuffle(faces, labels)  # Mezclar los datos
        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save('face_recognizer_model.yml')
        np.save('label_dict.npy', self.label_dict)
        print("Modelo entrenado y guardado.")
        # Verificar etiquetas
        print("Diccionario de etiquetas:", self.label_dict)

    def recognize_face(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
            gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label, confidence = self.recognizer.predict(
                gray_frame[y:y + h, x:x + w])

            # Umbral de confianza
            if confidence < 80:  # Ajusta este valor según tus pruebas
                name = self.label_dict[label]
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Desconocido", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame
