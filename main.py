import cv2
import time
from functions import detect_bounding_box, try_open_camera, crear_carpeta_imagenes

# Crear la carpeta para guardar los rostros detectados
save_path = crear_carpeta_imagenes()

# Intentar abrir la cámara
video_capture = try_open_camera()

while True:
    if video_capture is None or not video_capture.isOpened():
        print("Cámara no disponible. Reintentando en 2 segundos...")
        time.sleep(2)
        video_capture = try_open_camera()
        continue  # Asegúrate de continuar al siguiente ciclo si no se puede abrir la cámara

    result, video_frame = video_capture.read()
    if not result:
        print("No se puede capturar el video. Intentando reconectar...")
        video_capture.release()
        video_capture = None
        continue

    # Detectar rostros y guardar las imágenes
    detect_bounding_box(video_frame, save_path)

    # Mostrar el video con las detecciones
    cv2.imshow("Detección de Rostros", video_frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break  # Aquí solo salimos del bucle

# Liberar la cámara y cerrar las ventanas
if video_capture is not None:
    video_capture.release()
cv2.destroyAllWindows()
