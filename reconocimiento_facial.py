import cv2
import os
import numpy as np


class FacialRecognition:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.face_classifier = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.label_dict = {}

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
                current_label += 1

        self.recognizer.train(faces, np.array(labels))
        self.recognizer.save('face_recognizer_model.yml')
        np.save('label_dict.npy', self.label_dict)
        print("Modelo entrenado y guardado.")
        # Verificar etiquetas
        print("Diccionario de etiquetas:", self.label_dict)

    def recognize_face(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(
            gray_frame, 1.1, 5, minSize=(40, 40))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label, confidence = self.recognizer.predict(
                gray_frame[y:y + h, x:x + w])

            if confidence < 100:  # Si la confianza es baja, se reconoce la cara
                name = self.label_dict[label]
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Desconocido", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        return frame
