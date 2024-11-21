import os
import cv2
import boto3
from ultralytics import YOLO


def load_model():
    # Cargar el modelo de YOLO
    model = YOLO("yolov10m.pt")
    return model

def download_video_from_s3(bucket_name, s3_key, local_path):
    """
    Descarga un archivo de S3 a la ubicación local.
    """
    s3 = boto3.client("s3")
    try:
        print(f"Descargando video de S3: s3://{bucket_name}/{s3_key} -> {local_path}")
        s3.download_file(bucket_name, s3_key, local_path)
        print("Descarga completada.")
        return local_path
    except Exception as e:
        print(f"Error al descargar el archivo: {e}")
        raise

def detect_objects_in_video(model, video_path, output_path):
    """
    Procesa un video y detecta objetos.
    """
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame)
        img_with_boxes = results[0].plot()
        out.write(img_with_boxes)

    cap.release()
    out.release()
    print(f"Video guardado en: {output_path}")

def main():
    # Parámetros de S3
    bucket_name = os.getenv("S3_BUCKET", "v1deo-red1s")
    s3_key = os.getenv("S3_KEY", "https://v1deo-red1s.s3.us-east-1.amazonaws.com/DJI_20241111150629_0050_D.MP4")
    local_video_path = "/app/data/DJI_20241111150629_0050_D.mp4"

    # Descargar el video desde S3
    video_path = download_video_from_s3(bucket_name, s3_key, local_video_path)

    # Cargar modelo y procesar video
    model = load_model()
    output_path = "/app/data/output/detected_objects.mp4"
    detect_objects_in_video(model, video_path, output_path)

if __name__ == "__main__":
    main()
