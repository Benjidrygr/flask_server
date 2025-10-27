"""
Procesador de videos compatible con Celery
Versión adaptada de video_proccesor_impr.py para integración con Celery
"""

import os
import glob
import json
import time
import logging
from typing import Dict, Any, List
import subprocess
import numpy as np
import cv2  # type: ignore
from .sort import Sort
import shutil
import sys
import base64

# Importar desde el archivo original
from .video_processor import process_video_sequence, sort_videos_by_timestamp

from .config import (
    DEFAULT_MAX_AGE9, DEFAULT_MIN_HITS9,
    DEFAULT_IOU_THRESHOLD9
)

# Configurar logging
logger = logging.getLogger(__name__)

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

def get_memory_usage():
    """Obtener uso de memoria actual del proceso"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        return {
            'rss': memory_info.rss / 1024 / 1024,  # MB
            'vms': memory_info.vms / 1024 / 1024,  # MB
        }
    except ImportError:
        logger.warning("psutil no disponible. No se puede monitorear memoria.")
        return {'rss': 0, 'vms': 0}
    except Exception as e:
        logger.warning("Error obteniendo uso de memoria: %s", e)
        return {'rss': 0, 'vms': 0}

def compute_iou(boxA, boxB):
    """Calcula el IoU entre dos cajas en formato [x1, y1, x2, y2]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # Evitar división por cero si alguna área es 0
    unionArea = float(boxAArea + boxBArea - interArea)
    return interArea / unionArea if unionArea > 0 else 0.0


def get_frame_count_for_video(video_path: str) -> int:
    """Abre un video y devuelve su conteo total de frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning(f"No se pudo abrir {video_path} para contar frames.")
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return frame_count

def serialize_detections(detections):
    """Serializar detecciones para Celery"""
    try:
        return base64.b64encode(json.dumps(detections, default=str).encode()).decode()
    except Exception as e:
        logger.error("Error serializando detecciones: %s", e)
        return None

def deserialize_detections(serialized_data):
    """Deserializar detecciones desde Celery"""
    try:
        return json.loads(base64.b64decode(serialized_data.encode()).decode())
    except Exception as e:
        logger.error("Error deserializando detecciones: %s", e)
        return []

def check_video_dependencies():
    """Verificar que todas las dependencias de video estén disponibles"""
    try:
        # Verificar ffmpeg
        subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True)
        logger.info("✅ Dependencias de video verificadas correctamente")
        return True
    except FileNotFoundError:
        logger.error("❌ FFmpeg no encontrado. Instálalo y asegúrate de que esté en el PATH")
        return False
    except subprocess.CalledProcessError as e:
        logger.error("❌ Error verificando dependencias: %s", e)
        return False

def process_videos_sequential(video_folder: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Procesa videos secuencialmente (compatible con Celery).
    Reemplaza el multiprocessing con procesamiento secuencial.
    """
    if config is None:
        config = {}
    
    tiempo_inicio = time.time()
    logger.info("🎬 Iniciando procesamiento secuencial de videos en: %s", video_folder)
    
    # Monitorear memoria inicial
    mem_start = get_memory_usage()
    logger.info("📊 Memoria inicial: %.2f MB (RSS), %.2f MB (VMS)", 
               mem_start['rss'], mem_start['vms'])
    
    # Verificar dependencias
    if not check_video_dependencies():
        return {"success": False, "message": "Dependencias no disponibles", "detections": [], "stats": {}}
    
    # Obtener y ordenar todos los videos
    video_files = glob.glob(os.path.join(video_folder, "**/*.mp4"), recursive=True)
    if not video_files:
        logger.warning("No se encontraron videos para procesar")
        return {"success": False, "message": "No videos found", "detections": [], "stats": {}}
    
    video_files = sort_videos_by_timestamp(video_files)
    total_videos = len(video_files)
    
    # Configuración
    skip_frames = config.get('skip_frames', False)
    max_age = config.get('max_age', DEFAULT_MAX_AGE9)
    min_hits = config.get('min_hits', DEFAULT_MIN_HITS9)
    iou_threshold = config.get('iou_threshold', DEFAULT_IOU_THRESHOLD9)
    
    logger.info("📊 Procesando %d videos secuencialmente", total_videos)
    logger.info("⚙️ Configuración: skip_frames=%s, max_age=%d, min_hits=%d, iou_threshold=%.2f", 
                skip_frames, max_age, min_hits, iou_threshold)
    
    # Pre-calcular conteos de frames
    logger.info("📊 Pre-calculando conteos de frames...")
    video_frame_counts = {path: get_frame_count_for_video(path) for path in video_files}
    total_frames = sum(video_frame_counts.values())
    logger.info("📊 Total de frames a procesar: %d", total_frames)
    
    # Procesar videos secuencialmente
    all_detections = []
    processed_videos = 0
    global_frame_offset = 0
    
    # Configuración de memoria
    max_detections_in_memory = config.get('max_detections_in_memory', 10000)
    save_intermediate_results = config.get('save_intermediate_results', False)
    intermediate_file = None
    
    if save_intermediate_results:
        intermediate_file = f"temp_detections_{int(time.time())}.json"
        logger.info("💾 Guardando resultados intermedios en: %s", intermediate_file)
    
    for i, video_path in enumerate(video_files):
        logger.info("🎬 Procesando video %d/%d: %s", i+1, total_videos, os.path.basename(video_path))
        
        try:
            # Procesar video individual
            result = process_video_sequence(
                video_folder, 
                skip_frames=skip_frames, 
                worker=0,  # Sin multiprocessing
                video_range=(i, i+1),  # Procesar un video a la vez
                frame_offset=global_frame_offset
            )
            
            logger.info(f"🔍 Debug: result type: {type(result)}")
            if result:
                logger.info(f"🔍 Debug: result keys: {result.keys() if isinstance(result, dict) else 'Not a dict'}")
            
            if result and "detections" in result:
                video_detections = result["detections"]
                logger.info(f"🔍 Debug: video_detections from result: {len(video_detections)}")
                
                # Control de memoria: guardar intermedio si hay muchas detecciones
                if len(all_detections) + len(video_detections) > max_detections_in_memory:
                    if save_intermediate_results and intermediate_file:
                        logger.info("💾 Guardando %d detecciones intermedias...", len(all_detections))
                        with open(intermediate_file, 'w') as f:
                            json.dump(all_detections, f, indent=2)
                        all_detections = []  # Limpiar memoria
                    else:
                        logger.warning("⚠️ Muchas detecciones en memoria (%d). Considera usar save_intermediate_results=True", 
                                     len(all_detections) + len(video_detections))
                
                logger.info(f"🔍 Debug: video_detections type: {type(video_detections)}, length: {len(video_detections)}")
                if len(video_detections) > 0:
                    logger.info(f"🔍 Debug: Primer video_detection: {type(video_detections[0])}")
                
                all_detections.extend(video_detections)
                processed_videos += 1
                
                # Monitorear memoria cada 5 videos
                if processed_videos % 5 == 0:
                    mem_current = get_memory_usage()
                    logger.info("📊 Memoria actual (video %d): %.2f MB (RSS), %.2f MB (VMS)", 
                               processed_videos, mem_current['rss'], mem_current['vms'])
                
                logger.info("✅ Video procesado: %d detecciones (total: %d)", 
                           len(video_detections), len(all_detections))
            else:
                logger.warning("⚠️ Video sin detecciones: %s", os.path.basename(video_path))
            
            # Actualizar offset global
            global_frame_offset += video_frame_counts.get(video_path, 0)
            
        except Exception as e:
            logger.error("❌ Error procesando video %s: %s", video_path, e)
            continue
    
    # Ordenar detecciones por timestamp
    all_detections.sort(key=lambda x: x.get("timestamp", 0))
    
    # Calcular tiempo total
    tiempo_total = time.time() - tiempo_inicio
    
    # Monitorear memoria final
    mem_end = get_memory_usage()
    logger.info("📊 Memoria final: %.2f MB (RSS), %.2f MB (VMS)", 
               mem_end['rss'], mem_end['vms'])
    logger.info("📊 Incremento de memoria: %.2f MB (RSS)", mem_end['rss'] - mem_start['rss'])
    
    # Estadísticas
    stats = {
        "total_videos": total_videos,
        "processed_videos": processed_videos,
        "total_detections": len(all_detections),
        "total_processing_time_seconds": round(tiempo_total, 2),
        "total_frames": total_frames,
        "processing_mode": "sequential",
        "memory_usage": {
            "initial_mb": round(mem_start['rss'], 2),
            "final_mb": round(mem_end['rss'], 2),
            "increment_mb": round(mem_end['rss'] - mem_start['rss'], 2)
        },
        "tracker_params": {
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_threshold": iou_threshold
        }
    }
    
    result = {
        "success": True,
        "message": "Procesamiento secuencial completado exitosamente",
        "detections": all_detections,
        "stats": stats
    }
    
    logger.info(f"🎉 Procesamiento completado: {processed_videos}/{total_videos} videos, {len(all_detections)} detecciones")
    logger.info(f"🔍 Debug: all_detections type: {type(all_detections)}, length: {len(all_detections)}")
    if len(all_detections) > 0:
        logger.info(f"🔍 Debug: Primer elemento: {type(all_detections[0])}")
    
    logger.info(f"🔍 Debug: result['detections'] type: {type(result['detections'])}, length: {len(result['detections'])}")
    logger.info(f"🔍 Debug: result keys: {result.keys()}")
    
    return result

def apply_tracking_to_detections(detections_data: List[Dict], 
                                max_age: int = DEFAULT_MAX_AGE9,
                                min_hits: int = DEFAULT_MIN_HITS9,
                                iou_threshold: float = DEFAULT_IOU_THRESHOLD9) -> Dict[str, Any]:
    """
    Aplica tracking a las detecciones (compatible con Celery).
    Versión adaptada sin archivos temporales.
    """
    logger.info("🔄 Iniciando aplicación de tracking SORT")
    
    if not detections_data:
        logger.warning("No hay detecciones para aplicar tracking")
        return {"success": False, "message": "No detecciones encontradas", "detections": [], "stats": {}}
    
    # Ordenar detecciones por frame_number
    detections_data.sort(key=lambda x: x.get("frame_number", 0))
    total_initial_detections = len(detections_data)
    
    # Inicializar tracker
    tracker = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
    tracked_detections = []
    
    # Agrupar detecciones por frame_number
    frames = {}
    for det in detections_data:
        fn = det.get("frame_number", 0)
        if fn not in frames:
            frames[fn] = []
        frames[fn].append(det)
    
    logger.info(f"📊 Aplicando tracking a {total_initial_detections} detecciones en {len(frames)} frames")
    
    # Procesar cada frame
    for frame_num in sorted(frames.keys()):
        frame_dets = frames[frame_num]
        
        if not frame_dets:
            tracker.update(np.empty((0, 5)))
            continue
        
        # Convertir detecciones al formato SORT
        dets_for_sort = np.array([
            [d["box"][0], d["box"][1], d["box"][0] + d["box"][2], d["box"][1] + d["box"][3], d["confidence"]]
            for d in frame_dets
        ])
        
        tracked_objects = tracker.update(dets_for_sort)
        
        # Emparejar objetos trackeados con detecciones originales
        unmatched_det_indices = list(range(len(frame_dets)))
        for track in tracked_objects:
            track_box = track[:4]
            best_iou, best_match_idx = 0.0, -1
            
            for i in unmatched_det_indices:
                det = frame_dets[i]
                det_box_iou = [det['box'][0], det['box'][1], det['box'][0] + det['box'][2], det['box'][1] + det['box'][3]]
                iou = compute_iou(track_box, det_box_iou)
                if iou > best_iou:
                    best_iou, best_match_idx = iou, i
            
            # Crear detección trackeada si hay buen emparejamiento
            if best_match_idx != -1 and best_iou > 0.2:
                original_det = frame_dets[best_match_idx]
                unmatched_det_indices.remove(best_match_idx)
                
                tracked_det = {
                    "timestamp": original_det.get("timestamp", 0),
                    "frame_number": frame_num,
                    "track_id": int(track[4]),
                    "box": [float(track[0]), float(track[1]), float(track[2] - track[0]), float(track[3] - track[1])],
                    "center_x": float(track[0] + (track[2] - track[0]) / 2),
                    "center_y": float(track[1] + (track[3] - track[1]) / 2),
                    "confidence": float(original_det.get("confidence", 0)),
                    "class": original_det.get("class", "unknown")
                }
                tracked_detections.append(tracked_det)
    
    if not tracked_detections:
        logger.warning("No se generaron tracks")
        return {"success": True, "message": "No se generaron tracks.", "detections": [], "stats": {}}
    
    # Agrupar detecciones por clase
    class_groups = {}
    for det in tracked_detections:
        class_name = det.get("class", "unknown")
        if class_name not in class_groups:
            class_groups[class_name] = []
        class_groups[class_name].append(det)
    
    # Ordenar grupos por frame_number
    for class_name in class_groups:
        class_groups[class_name].sort(key=lambda x: x.get("frame_number", 0))
    
    # Segmentar grupos basados en saltos en frame_number
    initial_groups = []
    for class_name, group_detections in class_groups.items():
        if not group_detections:
            continue
        
        current_group = [group_detections[0]]
        for i in range(1, len(group_detections)):
            if group_detections[i].get("frame_number", 0) - group_detections[i-1].get("frame_number", 0) > 90:
                initial_groups.append({"class": class_name, "detections": current_group})
                current_group = []
            current_group.append(group_detections[i])
        if current_group:
            initial_groups.append({"class": class_name, "detections": current_group})
    
    # Aplicar filtro post-tracking
    logger.info("🔍 Aplicando filtro post-tracking...")
    final_groups = []
    removed_count = 0
    
    for group in initial_groups:
        detections_in_group = group['detections']
        
        # Agrupar por timestamp para encontrar duplicados
        timestamp_groups = {}
        for det in detections_in_group:
            ts = det.get('timestamp', 0)
            if ts not in timestamp_groups:
                timestamp_groups[ts] = []
            timestamp_groups[ts].append(det)
        
        # Filtrar duplicados
        filtered_dets_for_group = []
        for ts, ts_dets in timestamp_groups.items():
            if len(ts_dets) <= 1:
                filtered_dets_for_group.extend(ts_dets)
                continue
            
            to_remove = [False] * len(ts_dets)
            for i in range(len(ts_dets)):
                if to_remove[i]: 
                    continue
                for j in range(i + 1, len(ts_dets)):
                    if to_remove[j]: 
                        continue
                    
                    box_i = ts_dets[i]['box']
                    box_j = ts_dets[j]['box']
                    box_i_iou = [box_i[0], box_i[1], box_i[0] + box_i[2], box_i[1] + box_i[3]]
                    box_j_iou = [box_j[0], box_j[1], box_j[0] + box_j[2], box_j[1] + box_j[3]]
                    
                    if compute_iou(box_i_iou, box_j_iou) > 0.70:
                        if ts_dets[i].get('confidence', 0) < ts_dets[j].get('confidence', 0):
                            to_remove[i] = True
                            break
                        else:
                            to_remove[j] = True
            
            filtered_dets_for_group.extend([d for i, d in enumerate(ts_dets) if not to_remove[i]])
        
        removed_count += len(detections_in_group) - len(filtered_dets_for_group)
        
        # Reconstruir grupo si tiene detecciones
        if filtered_dets_for_group:
            filtered_dets_for_group.sort(key=lambda x: x.get('frame_number', 0))
            final_groups.append({
                "class": group['class'],
                "beginFrame": filtered_dets_for_group[0].get("frame_number", 0),
                "endFrame": filtered_dets_for_group[-1].get("frame_number", 0),
                "beginTimestamp": int(filtered_dets_for_group[0].get("timestamp", 0)),
                "endTimestamp": int(filtered_dets_for_group[-1].get("timestamp", 0)),
                "detections": filtered_dets_for_group
            })
    
    logger.info(f"✅ Filtrado completado. Eliminadas {removed_count} detecciones duplicadas")
    
    # Ordenar grupos por timestamp
    final_groups.sort(key=lambda x: x.get("beginTimestamp", 0))
    
    # Estadísticas finales
    total_tracked_detections_after_filter = sum(len(g['detections']) for g in final_groups)
    
    stats = {
        "total_initial_detections": total_initial_detections,
        "total_tracked_detections_before_filter": len(tracked_detections),
        "removed_in_post_filter": removed_count,
        "total_tracked_detections_after_filter": total_tracked_detections_after_filter,
        "total_groups": len(final_groups),
        "unique_tracks": len(set(det.get("track_id", 0) for det in tracked_detections)),
        "groups_by_class": {
            cn: len([g for g in final_groups if g["class"] == cn])
            for cn in class_groups
        },
        "tracker_params": {
            "max_age": max_age,
            "min_hits": min_hits,
            "iou_threshold": iou_threshold
        }
    }
    
    result = {
        "success": True,
        "message": "Tracking y filtrado completados exitosamente",
        "detections": final_groups,
        "stats": stats
    }
    
    logger.info(f"🎉 Tracking completado: {len(final_groups)} grupos, {total_tracked_detections_after_filter} detecciones finales")
    return result

def visualize_from_json(video_folder: str, detections_data: List[Dict], output_folder: str = "output_videos") -> Dict[str, Any]:
    """
    Visualiza las detecciones (compatible con Celery).
    Versión adaptada sin archivos JSON temporales.
    """
    logger.info(f"🎨 Iniciando visualización de {len(detections_data)} detecciones")
    
    os.makedirs(output_folder, exist_ok=True)
    
    if not detections_data:
        logger.warning("No hay detecciones para visualizar")
        return {"success": False, "message": "No hay detecciones para visualizar", "stats": {}}
    
    # Pre-procesar detecciones por frame
    detections_by_frame = {}
    for group in detections_data:
        for det in group.get("detections", []):
            fn = det.get("frame_number", 0)
            if fn is None:
                logger.warning(f"Detección sin frame_number ignorada: {det}")
                continue
            
            if fn not in detections_by_frame:
                detections_by_frame[fn] = []
            
            det_to_draw = det.copy()
            det_to_draw["class"] = group.get("class", "unknown")
            detections_by_frame[fn].append(det_to_draw)
    
    # Obtener videos
    video_files = sort_videos_by_timestamp(glob.glob(os.path.join(video_folder, "**/*.mp4"), recursive=True))
    if not video_files:
        logger.error("No se encontraron videos para visualizar")
        return {"success": False, "message": "No se encontraron videos", "stats": {}}
    
    # Colores para visualización
    np.random.seed(42)
    colors = {i: np.random.randint(0, 255, size=3).tolist() for i in range(2000)}
    
    # Procesar videos
    global_frame_counter = 0
    total_detections_drawn = 0
    total_frames_processed = 0
    
    logger.info(f"🎬 Procesando {len(video_files)} videos para visualización...")
    
    # Crear carpeta temporal
    temp_dir = os.path.join(output_folder, "temp_videos")
    os.makedirs(temp_dir, exist_ok=True)
    
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        logger.info(f"🎬 Procesando video: {video_name}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"No se pudo abrir el video {video_name}")
            continue
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.warning(f"FPS inválido ({fps}) para {video_name}. Usando 3.")
            fps = 3
        
        # Archivos de salida
        temp_output_path = os.path.join(temp_dir, f"temp_{video_name}")
        final_output_path = os.path.join(output_folder, f"visualized_{video_name}")
        
        out = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"No se pudo crear el archivo de salida {temp_output_path}")
            cap.release()
            continue
        
        local_frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Buscar detecciones para el frame actual
            if global_frame_counter in detections_by_frame:
                for det in detections_by_frame[global_frame_counter]:
                    total_detections_drawn += 1
                    box = det["box"]
                    
                    x1 = int(box[0] * width)
                    y1 = int(box[1] * height)
                    w = int(box[2] * width)
                    h = int(box[3] * height)
                    
                    # Validar coordenadas
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    w = max(1, min(w, width - x1))
                    h = max(1, min(h, height - y1))
                    
                    track_id = det.get("track_id", 0)
                    color = colors.get(track_id % 2000, (255, 255, 255))
                    
                    cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
                    text = f"ID:{track_id} {det.get('class', 'unknown')}"
                    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
                    cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            out.write(frame)
            global_frame_counter += 1
            local_frame_count += 1
        
        total_frames_processed += local_frame_count
        cap.release()
        out.release()
        logger.info(f"✅ Video temporal: {video_name} ({local_frame_count} frames)")
        
        # Convertir con ffmpeg
        try:
            logger.info(f"🔄 Convirtiendo {video_name} con ffmpeg...")
            ffmpeg_command = [
                'ffmpeg', '-y', '-i', temp_output_path,
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'fast', '-crf', '23',
                final_output_path
            ]
            result = subprocess.run(ffmpeg_command, capture_output=True, text=True, check=True)
            logger.info(f"✅ Conversión exitosa: {os.path.basename(final_output_path)}")
        except FileNotFoundError:
            logger.error("❌ FFmpeg no encontrado. Instálalo y asegúrate de que esté en el PATH")
            break
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Error en conversión ffmpeg para {video_name}: {e.stderr}")
            continue
    
    # Limpiar carpeta temporal
    try:
        shutil.rmtree(temp_dir)
        logger.info("🧹 Carpeta temporal eliminada")
    except OSError as e:
        logger.error(f"Error eliminando carpeta temporal {temp_dir}: {e}")
    
    stats = {
        "total_videos_processed": len(video_files),
        "total_frames_processed": total_frames_processed,
        "total_detections_drawn": total_detections_drawn
    }
    
    result = {
        "success": True,
        "message": "Visualización completada exitosamente",
        "stats": stats
    }
    
    logger.info(f"🎉 Visualización completada: {total_detections_drawn} detecciones dibujadas en {total_frames_processed} frames")
    return result

# Función principal para compatibilidad
def process_videos_celery_compatible(video_folder: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Función principal compatible con Celery.
    Combina detección, tracking y visualización.
    """
    if config is None:
        config = {}
    
    logger.info(f"🚀 Iniciando procesamiento completo de videos")
    
    # Paso 1: Detección
    logger.info("🔍 Paso 1: Detección de objetos")
    
    try:
        detection_result = process_videos_sequential(video_folder, config)
        
        if not detection_result["success"]:
            error_msg = detection_result.get("message", "Error desconocido en detección")
            logger.error(f"❌ Error en detección: {error_msg}")
            return detection_result
        
        detections = detection_result["detections"]
        logger.info(f"📊 Detecciones encontradas: {len(detections)}")
        logger.info(f"🔍 Debug: detection_result keys: {detection_result.keys()}")
        logger.info(f"🔍 Debug: detection_result['detections'] type: {type(detections)}")
        logger.info(f"🔍 Debug: detection_result['detections'] length: {len(detections) if hasattr(detections, '__len__') else 'No length'}")
        
        if not detections:
            logger.warning("No se encontraron detecciones")
            result = {
                "success": True,
                "message": "No se encontraron detecciones",
                "detections": [],
                "tracked_detections": [],
                "stats": detection_result["stats"]
            }
            return convert_numpy_types(result)
            
    except Exception as e:
        error_msg = f"Excepción en detección: {str(e)}"
        logger.error(f"❌ {error_msg}")
        result = {
            "success": False,
            "message": error_msg,
            "detections": [],
            "tracked_detections": [],
            "stats": {}
        }
        return convert_numpy_types(result)
    
    # Paso 2: Tracking
    logger.info("🔄 Paso 2: Aplicando tracking SORT")
    logger.info(f"📊 Detecciones a procesar: {len(detections)}")
    
    try:
        tracking_result = apply_tracking_to_detections(
            detections,
            max_age=config.get('max_age', DEFAULT_MAX_AGE9),
            min_hits=config.get('min_hits', DEFAULT_MIN_HITS9),
            iou_threshold=config.get('iou_threshold', DEFAULT_IOU_THRESHOLD9)
        )
        
        if not tracking_result["success"]:
            error_msg = tracking_result.get("message", "Error desconocido en tracking")
            logger.error(f"❌ Error aplicando tracking: {error_msg}")
            result = {
                "success": False,
                "message": f"Error aplicando tracking: {error_msg}",
                "detections": detections,
                "tracked_detections": [],
                "stats": detection_result["stats"]
            }
            return convert_numpy_types(result)
            
        # El resultado del tracking tiene estructura anidada: grupos -> detections
        tracked_groups = tracking_result.get('detections', [])
        total_tracked_detections = 0
        
        # Contar detecciones en todos los grupos
        for group in tracked_groups:
            if isinstance(group, dict) and 'detections' in group:
                total_tracked_detections += len(group['detections'])
        
        logger.info(f"✅ Tracking completado: {len(tracked_groups)} grupos, {total_tracked_detections} detecciones trackeadas")
        
    except Exception as e:
        error_msg = f"Excepción en tracking: {str(e)}"
        logger.error(f"❌ {error_msg}")
        result = {
            "success": False,
            "message": error_msg,
            "detections": detections,
            "tracked_detections": [],
            "stats": detection_result["stats"]
        }
        return convert_numpy_types(result)
    
    tracked_detections = tracking_result["detections"]  # Esto son los grupos, no las detecciones individuales
    
    # Paso 3: Visualización (opcional)
    if config.get('visualize', False):
        logger.info("🎨 Paso 3: Generando visualización")
        visualization_result = visualize_from_json(
            video_folder, 
            tracked_detections, 
            config.get('output_folder', 'output_videos')
        )
        
        if not visualization_result["success"]:
            logger.warning(f"Error en visualización: {visualization_result['message']}")
    
    # Resultado final
    final_result = {
        "success": True,
        "message": "Procesamiento completo exitoso",
        "detections": detections,
        "tracked_detections": tracked_detections,
        "stats": {
            "detection_stats": detection_result["stats"],
            "tracking_stats": tracking_result["stats"]
        }
    }
    
    # Convertir tipos NumPy a tipos nativos de Python para serialización JSON
    final_result = convert_numpy_types(final_result)
    
    # Guardar solo las detecciones trackeadas en un archivo separado
    tracked_only_file = f"tracked_detections_{int(time.time())}.json"
    try:
        with open(tracked_only_file, 'w') as f:
            json.dump(tracked_detections, f, indent=2)
        logger.info(f"💾 Detecciones trackeadas guardadas en: {tracked_only_file}")
    except Exception as e:
        logger.error(f"❌ Error guardando detecciones trackeadas: {e}")
    
    logger.info("🎉 Procesamiento completo finalizado exitosamente")
    return final_result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Procesa videos secuencialmente (compatible con Celery)')
    parser.add_argument('--video-folder', type=str, required=True,
                    help='Carpeta que contiene los videos a procesar')
    parser.add_argument('--skip', action='store_true',
                    help='Activar el procesamiento solo de frames seleccionados')
    parser.add_argument('--visualize', action='store_true',
                    help='Generar videos visualizados con el tracking')
    parser.add_argument('--max-age', type=int, default=DEFAULT_MAX_AGE9,
                    help='Parámetro max_age para SORT')
    parser.add_argument('--min-hits', type=int, default=DEFAULT_MIN_HITS9,
                    help='Parámetro min_hits para SORT')
    parser.add_argument('--iou-threshold', type=float, default=DEFAULT_IOU_THRESHOLD9,
                    help='Parámetro iou_threshold para SORT')
    
    args = parser.parse_args()
    
    # Configuración
    config = {
        'skip_frames': args.skip,
        'visualize': args.visualize,
        'max_age': args.max_age,
        'min_hits': args.min_hits,
        'iou_threshold': args.iou_threshold,
        'output_folder': 'output_videos'
    }
    
    # Ejecutar procesamiento
    result = process_videos_celery_compatible(args.video_folder, config)
    
    if result["success"]:
        print("\n✅ Procesamiento completado exitosamente")
        print("📊 Estadísticas:")
        print(json.dumps(result["stats"], indent=2))
        
        # Guardar resultados
        with open('celery_compatible_results.json', 'w') as f:
            json.dump(result, f, indent=2)
        print("💾 Resultados guardados en celery_compatible_results.json")
    else:
        print(f"\n❌ Error en procesamiento: {result['message']}")
        sys.exit(1)
