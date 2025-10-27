import cv2  # type: ignore
import numpy as np
import time
import os
import sys
from datetime import datetime
import json
from ultralytics import YOLO
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import psutil
from typing import List, Dict, Any
import subprocess

def convert_numpy_types(obj):
    """
    Convierte tipos de NumPy a tipos nativos de Python para serialización JSON
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


from .config import (
    DETECTION_THRESHOLD
)

def letterbox_image(image, target_size=(640, 640), color=(114, 114, 114)):
    """
    Redimensiona la imagen manteniendo su relación de aspecto y añade relleno
    para ajustarla al tamaño objetivo.
    """
    src_h, src_w = image.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / src_w, target_h / src_h)
    new_w, new_h = int(src_w * scale), int(src_h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    dw, dh = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas[dh:dh + new_h, dw:dw + new_w, :] = resized
    return canvas, scale, (dw, dh)

class VideoProcessor:
    def __init__(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Obtener la raíz del proyecto dinámicamente
        # El archivo video_processor.py está en la raíz del proyecto
        project_root = os.path.dirname(__file__)
        model_weights_path = os.path.join(project_root, 'weigths', '20251001-best.pt')
        
        # Buscar el modelo YOLO en varias ubicaciones posibles
        possible_paths = [
            'best.pt',
            'yolov8n.pt',  # Modelo por defecto de YOLO
            os.path.join(os.path.dirname(__file__), 'best.pt'),
            model_weights_path,  # Path dinámico basado en la raíz del proyecto
            os.path.join(project_root, 'weigths', '20251001-best.pt')  # Ruta específica para tu estructura
        ]
        
        model_path = None
        for path in possible_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if model_path is None:
            print(f"Error: Modelo YOLO no encontrado en ninguna de las rutas:")
            print(f"  - Raíz del proyecto detectada: {project_root}")
            print(f"  - Path dinámico calculado: {model_weights_path}")
            for path in possible_paths:
                print(f"  - {path}")
            print("Por favor, asegúrate de que existe un archivo de modelo YOLO (.pt) en una de estas ubicaciones.")
            sys.exit(1)
        self.model = YOLO(model_path).to(device)
        self.imgsz = (640, 640)
        print(f"Usando dispositivo: {device}")
        print(f"Modelo YOLO cargado desde: {model_path}")

    def visualize_sequence(self, video_folder: str, skip_frames: bool = False, video_range: tuple = None) -> Dict[str, Any]:
        """
        Procesa y visualiza una secuencia de videos con detecciones usando codec avc1 en memoria.
        
        Args:
            video_folder (str): Ruta a la carpeta que contiene los videos
            skip_frames (bool): Si se debe saltar frames para optimizar el procesamiento
            video_range (tuple): Tupla opcional (start_idx, end_idx) para procesar solo un rango de videos
            
        Returns:
            Dict[str, Any]: Diccionario con resultados y estadísticas del procesamiento
        """
        tiempo_inicio = time.time()
        
        # Encontrar y ordenar videos
        video_files = glob.glob(os.path.join(video_folder, "**/*.mp4"), recursive=True)
        if not video_files:
            return {"success": False, "message": "No videos found", "detections": [], "stats": {}}
        
        video_files = sort_videos_by_timestamp(video_files)
        total_videos = len(video_files)
        
        # Filtrar videos según el rango especificado
        if video_range is not None:
            start_idx, end_idx = video_range
            # Asegurarse de que los índices estén dentro de los límites
            start_idx = max(0, min(start_idx, total_videos))
            end_idx = max(start_idx, min(end_idx, total_videos))
            video_files = video_files[start_idx:end_idx]
            print(f"Procesando videos del rango {start_idx} a {end_idx-1} ({len(video_files)} videos)")
        else:
            print(f"Procesando {len(video_files)} videos")
        
        # Preparar visualización
        np.random.seed(0)
        colors = np.random.randint(0, 255, size=(256, 3))
        output_dir = "visualized_videos"
        os.makedirs(output_dir, exist_ok=True)
        
        # Estructuras para detecciones
        all_detections = []
        global_frame_counter = 0
        current_global_timestamp = None
        
        # Procesar cada video
        for i, video_file in enumerate(video_files):
            print(f"Procesando video {i+1}/{len(video_files)}: {os.path.basename(video_file)}")
            base_name = os.path.basename(video_file)
            output_path = os.path.join(output_dir, f"visualized_{base_name}")
            
            cap = cv2.VideoCapture(video_file)
            if not cap.isOpened():
                print(f"Advertencia: No se pudo abrir el archivo de video {video_file}. Omitiendo.")
                continue
            
            # Configuración del video
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps_video = cap.get(cv2.CAP_PROP_FPS)
            total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            output_fps = fps_video / 2 if skip_frames else fps_video
            
            # Asegurar que las dimensiones sean pares (requerido por algunos codecs)
            width = width if width % 2 == 0 else width - 1
            height = height if height % 2 == 0 else height - 1
            
            # Configurar VideoWriter con codec avc1
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))
            
            if not out.isOpened():
                print(f"Error: No se pudo crear el archivo de video de salida con avc1: {output_path}")
                cap.release()
                continue
            
            # Manejo de timestamps
            try:
                file_name = os.path.basename(video_file)
                begin_str = file_name.replace('.mp4', '').split('_')[0]
                if current_global_timestamp is None:
                    current_global_timestamp = int(datetime.strptime(begin_str, '%Y-%m-%d %H:%M:%S%z').timestamp())
            except:
                if current_global_timestamp is None:
                    current_global_timestamp = int(time.time())
                else:
                    last_video_duration = (total_frames_video / fps_video) if fps_video > 0 else (total_frames_video / 30.0)
                    current_global_timestamp += int(last_video_duration)
                print(f"Advertencia: No se pudo obtener timestamp del nombre de {video_file}.")
            
            # Procesar frames directamente en memoria
            frames_processed = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Asegurar que el frame tiene las dimensiones correctas
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                
                process_this_frame = not skip_frames or global_frame_counter % 2 == 0
                
                if process_this_frame:
                    # Procesar el frame con YOLO
                    frame_letterbox, scale, (dw, dh) = letterbox_image(frame, self.imgsz)
                    results = self.model(frame_letterbox, verbose=False)[0]
                    det_list = results.boxes.data.cpu().numpy()
                    
                    # Filtrar detecciones
                    detections = [
                        [x1, y1, x2, y2, conf, cls]
                        for x1, y1, x2, y2, conf, cls in det_list
                        if conf > DETECTION_THRESHOLD
                    ]
                    
                    # Calcular timestamp para este frame
                    frame_offset_seconds = (global_frame_counter / (fps_video if fps_video > 0 else 30.0))
                    frame_timestamp = current_global_timestamp + frame_offset_seconds
                    
                    # Procesar cada detección
                    for det in detections:
                        x1, y1, x2, y2, conf, cls = det
                        
                        # Convertir coordenadas al espacio original
                        x1_adj = int((x1 - dw) / scale)
                        y1_adj = int((y1 - dh) / scale)
                        x2_adj = int((x2 - dw) / scale)
                        y2_adj = int((y2 - dh) / scale)
                        
                        # Asegurar que las coordenadas estén dentro de los límites
                        x1_adj = max(0, min(width-1, x1_adj))
                        y1_adj = max(0, min(height-1, y1_adj))
                        x2_adj = max(0, min(width-1, x2_adj))
                        y2_adj = max(0, min(height-1, y2_adj))
                        
                        # Normalizar coordenadas [0, 1]
                        x1_norm = max(0.0, min(1.0, x1_adj / width))
                        y1_norm = max(0.0, min(1.0, y1_adj / height))
                        w_norm = min(1.0, (x2_adj - x1_adj) / width)
                        h_norm = min(1.0, (y2_adj - y1_adj) / height)
                        
                        # Calcular centro normalizado
                        center_x_norm = x1_norm + (w_norm / 2)
                        center_y_norm = y1_norm + (h_norm / 2)
                        
                        # Obtener el nombre de la clase
                        class_name = self.model.names[int(cls)]
                        
                        # Crear detección
                        detection = {
                            "timestamp": int(frame_timestamp),
                            "frame_number": global_frame_counter,
                            "box": [x1_norm, y1_norm, w_norm, h_norm],
                            "center_x": center_x_norm,
                            "center_y": center_y_norm,
                            "confidence": float(conf),
                            "class": class_name
                        }
                        
                        all_detections.append(detection)
                        
                        # Dibujar bounding box
                        color = tuple([int(c) for c in colors[int(cls) % len(colors)]])
                        cv2.rectangle(frame, (x1_adj, y1_adj), (x2_adj, y2_adj), color, 2)
                        
                        # Dibujar clase y confianza
                        text = f"{class_name} {conf:.2f}"
                        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                        if y1_adj - 20 > 0:
                            label_y, text_y = y1_adj - 20, y1_adj - 8
                        else:
                            label_y, text_y = y2_adj, y2_adj + 15
                        cv2.rectangle(frame, (x1_adj, label_y), (x1_adj + text_size[0], label_y + 20), (0, 0, 0), -1)
                        cv2.putText(frame, text, (x1_adj, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Escribir frame directamente al video
                out.write(frame)
                frames_processed += 1
                
                # Actualizar contadores y timestamps
                global_frame_counter += 1
                if fps_video > 0:
                    current_global_timestamp += (1.0 / fps_video)
                else:
                    current_global_timestamp += (1.0 / 30.0)
            
            # Cerrar recursos
            cap.release()
            out.release()
            
            if frames_processed > 0:
                print(f"✅ Video visualizado completado: {output_path} ({frames_processed} frames)")
            else:
                print(f"⚠️ No se procesaron frames para {video_file}")
        
        tiempo_total = time.time() - tiempo_inicio
        
        result = {
            "success": True,
            "message": "Processing completed successfully",
            "detections": all_detections,
            "stats": {
                "total_videos": len(video_files),
                "total_detections": len(all_detections),
                "processing_time_seconds": round(tiempo_total, 2),
                "processing_mode": "skip_frames" if skip_frames else "full_frames"
            }
        }
        
        return result

def sort_videos_by_timestamp(video_files: List[str]) -> List[str]:
    """
    Ordena los videos por timestamp en el nombre del archivo.
    Si no puede ordenar por timestamp, ordena alfabéticamente.
    """
    try:
        video_files.sort(key=lambda x: datetime.strptime(
            os.path.basename(x).split('_')[0], 
            '%Y-%m-%d %H:%M:%S%z'
        ).timestamp())
        print("Videos ordenados por timestamp en el nombre.")
    except:
        print("No se pudo ordenar videos por timestamp en el nombre. Ordenando alfabéticamente.")
        video_files.sort() # Si falla, intenta ordenar alfabéticamente (menos seguro)
    
    return video_files

def get_memory_usage():
    """Obtiene el uso actual de memoria en MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss': mem_info.rss / (1024 * 1024),  # Resident Set Size en MB
        'vms': mem_info.vms / (1024 * 1024),  # Virtual Memory Size en MB
    }

def process_video_sequence(video_folder: str, 
                         skip_frames: bool = False, 
                         worker: int = 0, 
                         video_range: tuple = None,
                         frame_offset: int = 0) -> Dict[str, Any]:
    """
    Procesa una secuencia de videos, los ordena por timestamp y extrae detecciones.
    
    Args:
        video_folder (str): Carpeta que contiene los videos.
        skip_frames (bool): Si es True, procesa solo frames seleccionados.
        worker (int): ID del worker (para logs).
        video_range (tuple): Rango de videos a procesar (start, end).
        frame_offset (int): Offset inicial para el contador de frames global.

    Returns:
        Dict[str, Any]: Resultados del procesamiento.
    """
    tiempo_inicio = time.time()
    processor = VideoProcessor()
    
    # Obtener y ordenar videos
    video_files = glob.glob(os.path.join(video_folder, "**/*.mp4"), recursive=True)
    if not video_files:
        return {"success": False, "message": "No videos found", "detections": [], "stats": {}}

    video_files = sort_videos_by_timestamp(video_files)
    total_videos = len(video_files)
    
    # Filtrar videos según el rango especificado
    if video_range is not None:
        start_idx, end_idx = video_range
        # Asegurarse de que los índices estén dentro de los límites
        start_idx = max(0, min(start_idx, total_videos))
        end_idx = max(start_idx, min(end_idx, total_videos))
        video_files = video_files[start_idx:end_idx]
        print(f"Worker {worker}: Procesando videos del rango {start_idx} a {end_idx-1} ({len(video_files)} videos)")
    else:
        print(f"Worker {worker}: Procesando todos los videos ({len(video_files)} videos)")

    if not video_files:
        return {"success": False, "message": "No videos in specified range", "detections": [], "stats": {}}

    # Procesar videos en el rango especificado
    global_frame_counter = frame_offset
    all_detections = []
    
    for i, video_file in enumerate(video_files):
        print(f"Worker {worker}: Procesando video {i+1}/{len(video_files)}: {os.path.basename(video_file)}")
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Advertencia: No se pudo abrir el archivo de video {video_file}. Omitiendo.")
            continue

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 30

        # Obtener el timestamp de inicio para el video actual desde su nombre de archivo
        try:
            file_name = os.path.basename(video_file)
            begin_str = file_name.replace('.mp4', '').split('_')[0]
            video_start_timestamp = datetime.strptime(begin_str, '%Y-%m-%d %H:%M:%S%z').timestamp()
        except (ValueError, IndexError) as e:
            print(f"Advertencia: No se pudo parsear el timestamp del video {file_name}. Usando timestamp por defecto. Error: {e}")
            # Usar timestamp por defecto en lugar de omitir el video
            video_start_timestamp = time.time()
        
        local_frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            process_this_frame = not skip_frames or global_frame_counter % 2 == 0

            if process_this_frame:
                frame_letterbox, scale, (dw, dh) = letterbox_image(frame, processor.imgsz)
                results = processor.model(frame_letterbox, verbose=False)[0]
                det_list = results.boxes.data.cpu().numpy()

                # Filtrar detecciones solo por umbral de confianza
                detections = [
                    [x1, y1, x2, y2, conf, cls]
                    for x1, y1, x2, y2, conf, cls in det_list
                    if conf > DETECTION_THRESHOLD
                ]

                # Calcular timestamp para este frame basado en el inicio del video actual
                frame_offset_seconds = local_frame_count / fps_video
                frame_timestamp = video_start_timestamp + frame_offset_seconds

                # Procesar cada detección
                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    
                    # Convertir coordenadas de vuelta al tamaño original del frame
                    x1_adj = (x1 - dw) / scale
                    y1_adj = (y1 - dh) / scale
                    x2_adj = (x2 - dw) / scale
                    y2_adj = (y2 - dh) / scale

                    # Normalizar coordenadas [0, 1]
                    x1_norm = max(0.0, min(1.0, x1_adj / orig_width))
                    y1_norm = max(0.0, min(1.0, y1_adj / orig_height))
                    w_norm = min(1.0, (x2_adj - x1_adj) / orig_width)
                    h_norm = min(1.0, (y2_adj - y1_adj) / orig_height)

                    # Calcular centro normalizado
                    center_x_norm = x1_norm + (w_norm / 2)
                    center_y_norm = y1_norm + (h_norm / 2)

                    # Obtener el nombre de la clase
                    class_name = processor.model.names[int(cls)]
                    
                    # Crear detección
                    detection = {
                        "timestamp": int(frame_timestamp),
                        "frame_number": global_frame_counter,
                        "box": [x1_norm, y1_norm, w_norm, h_norm],
                        "center_x": center_x_norm,
                        "center_y": center_y_norm,
                        "confidence": float(conf),
                        "class": class_name
                    }
                    
                    all_detections.append(detection)

            # Incrementar contadores
            global_frame_counter += 1
            local_frame_count += 1

        cap.release()

    tiempo_total = time.time() - tiempo_inicio

    result = {
        "success": True,
        "message": "Processing completed successfully",
        "detections": all_detections,
        "stats": {
            "total_videos": len(video_files),
            "total_detections": len(all_detections),
            "processing_time_seconds": round(tiempo_total, 2),
            "processing_mode": "skip_frames" if skip_frames else "full_frames"
        }
    }
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process video sequence with YOLOv8')
    parser.add_argument('--video-folder', type=str, default='AWSKinesisVideos/6762ead17d8603f0c9524688',
                        help='Carpeta que contiene los videos a procesar')
    parser.add_argument('--skip', action='store_true', help='Activar el procesamiento solo de frames seleccionados')
    parser.add_argument('--visualize', action='store_true', help='Generar videos con visualización de detecciones')
    args = parser.parse_args()

    video_folder = args.video_folder
    if not os.path.exists(video_folder):
        print(f"Error: La carpeta {video_folder} no existe")
        sys.exit(1)

    # Registrar uso de memoria inicial
    mem_start = get_memory_usage()

    if args.visualize:
        # Llamar a la función para visualización de secuencia
        processor = VideoProcessor()
        result = processor.visualize_sequence(video_folder, skip_frames=args.skip)
    else:
        # Llamar a la función para procesamiento de secuencia (solo datos)
        result = process_video_sequence(video_folder, skip_frames=args.skip)
    
    if len(result["detections"]) == 0:
        print("ADVERTENCIA: No se encontraron detecciones en los videos procesados")
    
    # Guardar los resultados de las detecciones
    output_file = 'detection_results.json'
    with open(output_file, 'w') as f:
        json.dump(convert_numpy_types(result["detections"]), f, indent=2)
    
    # Guardar las estadísticas
    stats_result = {
        "success": result["success"],
        "message": result["message"],
        "stats": result["stats"]
    }
    with open('processing_stats.json', 'w') as f:
        json.dump(convert_numpy_types(stats_result), f, indent=2)

    print("\nEstadísticas del procesamiento:")
    print(json.dumps(convert_numpy_types(stats_result), indent=2))
    
    # Medir y mostrar uso de memoria final
    mem_end = get_memory_usage()
    print("\n📊 Estadísticas de memoria:")
    print(f"Memoria usada (RSS): {mem_end['rss']:.2f} MB")
    print(f"Incremento de memoria: {mem_end['rss'] - mem_start['rss']:.2f} MB")
    if torch.cuda.is_available():
        gpu_mem_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
        gpu_mem_cached = torch.cuda.memory_reserved() / (1024 * 1024)
        print(f"Memoria GPU (allocada): {gpu_mem_alloc:.2f} MB")
        print(f"Memoria GPU (reservada): {gpu_mem_cached:.2f} MB")