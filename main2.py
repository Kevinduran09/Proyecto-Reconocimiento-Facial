import cv2
from reconocimiento_facial import FacialRecognition

# Inicializa la cámara
video_capture = cv2.VideoCapture(0)

# Inicializa el reconocimiento facial con la ruta a tu dataset
reconocimiento = FacialRecognition('dataset')
reconocimiento.train_recognizer()  # Entrena el modelo con el dataset

while True:
    ret, frame = video_capture.read()
    if not ret:
        break  # Termina el bucle si no se puede capturar el video

    # Detectar y reconocer caras
    frame = reconocimiento.recognize_face(frame)

    cv2.imshow("Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
