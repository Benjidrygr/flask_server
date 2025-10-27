# Configuración mínima para procesamiento de video (sin credenciales)

import os

# Umbral de confianza para YOLO
DETECTION_THRESHOLD = float(os.getenv('DETECTION_THRESHOLD', '0.6'))

# Parámetros por defecto para el tracker SORT (config 9 usada en el pipeline)
DEFAULT_MAX_AGE = int(os.getenv('DEFAULT_MAX_AGE', '90'))    # frames
DEFAULT_MIN_HITS = int(os.getenv('DEFAULT_MIN_HITS', '4'))
DEFAULT_IOU_THRESHOLD = float(os.getenv('DEFAULT_IOU_THRESHOLD', '0.10'))

# Alias para compatibilidad con video_proccesor_impr.py
DEFAULT_MAX_AGE9 = DEFAULT_MAX_AGE
DEFAULT_MIN_HITS9 = DEFAULT_MIN_HITS
DEFAULT_IOU_THRESHOLD9 = DEFAULT_IOU_THRESHOLD

# Tamaño de imagen usado para inferencia (mantener en sync con el código)
IMGSZ_WIDTH = int(os.getenv('IMGSZ_WIDTH', '640'))
IMGSZ_HEIGHT = int(os.getenv('IMGSZ_HEIGHT', '640'))
IMGSZ = (IMGSZ_WIDTH, IMGSZ_HEIGHT)

# Codec de salida recomendado para visualizaciones temporales
OUTPUT_CODEC = os.getenv('OUTPUT_CODEC', 'avc1')

# Directorios por defecto
DEFAULT_VIS_OUTPUT_DIR = os.getenv('DEFAULT_VIS_OUTPUT_DIR', 'visualized_videos')
DEFAULT_JSON_DETECTIONS = os.getenv('DEFAULT_JSON_DETECTIONS', 'detection_results.json')
DEFAULT_JSON_STATS = os.getenv('DEFAULT_JSON_STATS', 'processing_stats.json')

