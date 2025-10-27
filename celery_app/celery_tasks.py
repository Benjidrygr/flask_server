"""
Tareas de Celery para el procesador de videos YOLO
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from celery import Celery, chain
from celery.utils.log import get_task_logger

# Importar el procesador de videos
from yolo.video_processor_celery import process_videos_celery_compatible
# Removed unified_pipeline imports
from celery_config import CELERY_CONFIG, YOLO_CONFIG

# Configurar logging
logger = get_task_logger(__name__)

# Crear instancia de Celery
app = Celery('yolo_processor')
app.config_from_object(CELERY_CONFIG)

# ============================================================================
# TAREAS DEL PIPELINE UNIFICADO ELIMINADAS
# ============================================================================
# Las tareas del pipeline unificado han sido removidas del sistema Celery

# Tarea process_mixed_data eliminada

# ============================================================================
# TAREAS DE YOLO (DESPUÉS DEL PIPELINE UNIFICADO)
# ============================================================================

@app.task(bind=True, name='celery_tasks.log_start_yolo')
def log_start_yolo(self, video_folder: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Tarea 1: Log de inicio del proceso YOLO
    
    Args:
        video_folder: Carpeta que contiene los videos
        config: Configuración opcional para el procesamiento
        
    Returns:
        Dict con información del inicio del proceso
    """
    task_id = getattr(self.request, 'id', 'unknown')
    start_time = datetime.now()
    
    logger.info(f"🚀 [TAREA {task_id}] Iniciando proceso YOLO")
    logger.info(f"📁 Carpeta de videos: {video_folder}")
    logger.info(f"⚙️ Configuración: {config or 'Por defecto'}")
    logger.info(f"🕐 Hora de inicio: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificar que la carpeta existe
    if not os.path.exists(video_folder):
        error_msg = f"❌ La carpeta {video_folder} no existe"
        logger.error(error_msg)
        return {
            "success": False,
            "message": error_msg,
            "task_id": task_id,
            "start_time": start_time.isoformat(),
            "video_folder": video_folder
        }
    
    # Contar videos en la carpeta
    import glob
    video_files = glob.glob(os.path.join(video_folder, "**/*.mp4"), recursive=True)
    total_videos = len(video_files)
    
    logger.info(f"📊 Total de videos encontrados: {total_videos}")
    
    # Información del sistema
    import psutil
    memory_info = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    logger.info(f"💻 Recursos del sistema:")
    logger.info(f"   - CPU cores: {cpu_count}")
    logger.info(f"   - Memoria total: {memory_info.total / (1024**3):.2f} GB")
    logger.info(f"   - Memoria disponible: {memory_info.available / (1024**3):.2f} GB")
    
    result = {
        "success": True,
        "message": "Inicio de proceso YOLO registrado exitosamente",
        "task_id": task_id,
        "start_time": start_time.isoformat(),
        "video_folder": video_folder,
        "total_videos": total_videos,
        "system_info": {
            "cpu_cores": cpu_count,
            "total_memory_gb": round(memory_info.total / (1024**3), 2),
            "available_memory_gb": round(memory_info.available / (1024**3), 2)
        },
        "config": config or YOLO_CONFIG
    }
    
    logger.info(f"✅ [TAREA {task_id}] Log de inicio completado")
    return result

@app.task(bind=True, name='celery_tasks.process_videos_yolo')
def process_videos_yolo(self, video_folder: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Tarea 2: Procesamiento de videos con YOLO
    
    Args:
        video_folder: Carpeta que contiene los videos
        config: Configuración para el procesamiento
        
    Returns:
        Dict con resultados del procesamiento
    """
    task_id = getattr(self.request, 'id', 'unknown')
    start_time = datetime.now()
    
    logger.info(f"🎬 [TAREA {task_id}] Iniciando procesamiento de videos YOLO")
    logger.info(f"📁 Carpeta: {video_folder}")
    
    # Combinar configuración por defecto con la proporcionada
    final_config = YOLO_CONFIG.copy()
    if config:
        final_config.update(config)
    
    logger.info(f"⚙️ Configuración final: {final_config}")
    
    try:
        # Actualizar el progreso de la tarea
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Iniciando procesamiento...'}
        )
        
        # Ejecutar el procesamiento
        result = process_videos_celery_compatible(video_folder, final_config)
        
        # No actualizar el estado final para no sobrescribir el resultado
        # El resultado completo se devuelve directamente
        
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ [TAREA {task_id}] Procesamiento completado en {processing_duration:.2f} segundos")
        
        # Agregar información adicional al resultado
        result.update({
            "task_id": task_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "processing_duration_seconds": round(processing_duration, 2),
            "video_folder": video_folder,
            "config_used": final_config
        })
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Error en procesamiento: {str(e)}"
        logger.error(f"[TAREA {task_id}] {error_msg}")
        
        # Actualizar estado de error
        self.update_state(
            state='FAILURE',
            meta={'error': error_msg, 'task_id': task_id}
        )
        
        return {
            "success": False,
            "message": error_msg,
            "task_id": task_id,
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "video_folder": video_folder,
            "config_used": final_config,
            "error": str(e)
        }

@app.task(bind=True, name='celery_tasks.log_end_yolo')
def log_end_yolo(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tarea 3: Log de finalización del proceso YOLO
    
    Args:
        processing_result: Resultado del procesamiento de videos
        
    Returns:
        Dict con resumen final del proceso
    """
    task_id = getattr(self.request, 'id', 'unknown')
    end_time = datetime.now()
    
    logger.info(f"🏁 [TAREA {task_id}] Finalizando proceso YOLO")
    
    # Extraer información del resultado de procesamiento
    # Manejar diferentes formatos de resultado
    if isinstance(processing_result, dict):
        success = processing_result.get("success", False)
        video_folder = processing_result.get("video_folder", "Desconocido")
        processing_duration = processing_result.get("processing_duration_seconds", 0)
        
        # Intentar extraer estadísticas de diferentes formatos
        stats = processing_result.get("stats", {})
        if isinstance(stats, dict):
            detection_stats = stats.get("detection_stats", {})
            tracking_stats = stats.get("tracking_stats", {})
        else:
            detection_stats = {}
            tracking_stats = {}
    else:
        # Si no es un diccionario, asumir que fue exitoso
        success = True
        video_folder = "Desconocido"
        processing_duration = 0
        detection_stats = {}
        tracking_stats = {}
    
    if success:
        total_videos = detection_stats.get("total_videos", 0)
        processed_videos = detection_stats.get("processed_videos", 0)
        total_detections = detection_stats.get("total_detections", 0)
        total_groups = tracking_stats.get("total_groups", 0)
        
        logger.info(f"✅ Procesamiento exitoso:")
        logger.info(f"   📁 Carpeta procesada: {video_folder}")
        logger.info(f"   🎬 Videos procesados: {processed_videos}/{total_videos}")
        logger.info(f"   🔍 Total detecciones: {total_detections}")
        logger.info(f"   🎯 Grupos de tracking: {total_groups}")
        logger.info(f"   ⏱️ Tiempo total: {processing_duration:.2f} segundos")
        
        # Calcular estadísticas de rendimiento
        if processed_videos > 0 and processing_duration > 0:
            videos_per_second = processed_videos / processing_duration
            logger.info(f"   📊 Rendimiento: {videos_per_second:.2f} videos/segundo")
        
        # Información de memoria final
        import psutil
        memory_info = psutil.virtual_memory()
        logger.info(f"   💾 Memoria disponible: {memory_info.available / (1024**3):.2f} GB")
        
        result = {
            "success": True,
            "message": "Proceso YOLO completado exitosamente",
            "task_id": task_id,
            "end_time": end_time.isoformat(),
            "summary": {
                "video_folder": video_folder,
                "total_videos": total_videos,
                "processed_videos": processed_videos,
                "total_detections": total_detections,
                "total_groups": total_groups,
                "processing_duration_seconds": processing_duration,
                "videos_per_second": round(videos_per_second, 2) if processed_videos > 0 and processing_duration > 0 else 0,
                "final_memory_available_gb": round(memory_info.available / (1024**3), 2)
            },
            "processing_result": processing_result
        }
        
    else:
        # Manejar diferentes tipos de errores
        if isinstance(processing_result, dict):
            error_message = processing_result.get("message", "Error desconocido")
        else:
            error_message = f"Resultado inesperado: {type(processing_result).__name__}"
        
        logger.error(f"❌ Procesamiento falló: {error_message}")
        
        result = {
            "success": False,
            "message": f"Proceso YOLO falló: {error_message}",
            "task_id": task_id,
            "end_time": end_time.isoformat(),
            "error_summary": {
                "video_folder": video_folder,
                "error_message": error_message,
                "processing_duration_seconds": processing_duration
            },
            "processing_result": processing_result
        }
    
    logger.info(f"🏁 [TAREA {task_id}] Log de finalización completado")
    return result

# ============================================================================
# TAREAS COMPUESTAS (PIPELINE COMPLETO)
# ============================================================================

# Tarea process_complete_pipeline_with_unified eliminada

# Tarea compuesta que ejecuta las 3 tareas en secuencia (solo YOLO)
@app.task(bind=True, name='celery_tasks.process_videos_complete_pipeline')
def process_videos_complete_pipeline(self, video_folder: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Pipeline completo que ejecuta las 3 tareas en secuencia:
    1. Log de inicio
    2. Procesamiento de videos
    3. Log de finalización
    
    Args:
        video_folder: Carpeta que contiene los videos
        config: Configuración para el procesamiento
        
    Returns:
        Dict con resultado completo del pipeline
    """
    task_id = getattr(self.request, 'id', 'unknown')
    logger.info(f"🔄 [PIPELINE {task_id}] Iniciando pipeline completo de procesamiento YOLO")
    
    try:
        # Paso 1: Log de inicio - LLAMADA DIRECTA
        logger.info("📝 Paso 1/3: Ejecutando log de inicio...")
        start_log = log_start_yolo(video_folder, config)
        
        if not start_log.get("success", False):
            raise Exception(f"Error en log de inicio: {start_log.get('message', 'Error desconocido')}")
        
        # Paso 2: Procesamiento de videos - LLAMADA DIRECTA
        logger.info("🎬 Paso 2/3: Ejecutando procesamiento de videos...")
        processing = process_videos_yolo(video_folder, config)
        
        # Paso 3: Log de finalización - LLAMADA DIRECTA
        logger.info("📝 Paso 3/3: Ejecutando log de finalización...")
        # Crear un objeto mock para self
        class MockSelf:
            def update_state(self, state, meta):
                pass
        mock_self = MockSelf()
        end_log = log_end_yolo(mock_self, processing)
        
        # Resultado final
        final_result = {
            "success": True,
            "message": "Pipeline completo ejecutado exitosamente",
            "pipeline_task_id": task_id,
            "start_log": start_log,
            "processing_result": processing,
            "end_log": end_log,
            "video_folder": video_folder,
            "config_used": config or YOLO_CONFIG
        }
        
        logger.info(f"🎉 [PIPELINE {task_id}] Pipeline completo finalizado exitosamente")
        return final_result
        
    except Exception as e:
        error_msg = f"❌ Error en pipeline: {str(e)}"
        logger.error(f"[PIPELINE {task_id}] {error_msg}")
        
        return {
            "success": False,
            "message": error_msg,
            "pipeline_task_id": task_id,
            "video_folder": video_folder,
            "config_used": config or YOLO_CONFIG,
            "error": str(e)
        }

if __name__ == '__main__':
    app.start()
