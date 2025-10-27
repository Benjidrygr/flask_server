#!/usr/bin/env python3
"""
Tareas de Celery OPTIMIZADAS para reducir overhead
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from celery import Celery
from celery.utils.log import get_task_logger

# Importar el procesador de videos
from yolo.video_processor_celery import process_videos_celery_compatible
# Removed unified_pipeline imports
from .celery_config import CELERY_CONFIG, YOLO_CONFIG

# Importar sistema de cache
try:
    from .cache_manager import cache_video_processing, set_video_processing_cache
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    print("⚠️ Sistema de cache no disponible")

# Configurar logging
logger = get_task_logger(__name__)

# Crear instancia de Celery
app = Celery('yolo_processor_optimized')
app.config_from_object(CELERY_CONFIG)

# ============================================================================
# TAREAS OPTIMIZADAS CON CACHE
# ============================================================================

@app.task(bind=True, name='celery_tasks_optimized.process_videos_yolo_cached')
def process_videos_yolo_cached(self, video_folder: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Procesamiento de videos YOLO con cache para reducir overhead
    
    Args:
        video_folder: Carpeta que contiene los videos
        config: Configuración para el procesamiento
        
    Returns:
        Dict con resultados del procesamiento
    """
    task_id = self.request.id
    start_time = datetime.now()
    
    logger.info(f"🎬 [TAREA CACHEADA {task_id}] Iniciando procesamiento de videos YOLO con cache")
    logger.info(f"📁 Carpeta: {video_folder}")
    
    # Combinar configuración por defecto con la proporcionada
    final_config = YOLO_CONFIG.copy()
    if config:
        final_config.update(config)
    
    # Verificar cache si está disponible
    if CACHE_AVAILABLE:
        cached_result = cache_video_processing(video_folder, final_config)
        if cached_result:
            logger.info(f"🎯 Cache hit para procesamiento YOLO: {video_folder}")
            cached_result.update({
                "task_id": task_id,
                "cached": True,
                "cache_timestamp": datetime.now().isoformat()
            })
            return cached_result
    
    try:
        # Actualizar el progreso de la tarea
        self.update_state(
            state='PROGRESS',
            meta={'current': 0, 'total': 100, 'status': 'Iniciando procesamiento...'}
        )
        
        # Ejecutar el procesamiento
        result = process_videos_celery_compatible(video_folder, final_config)
        
        end_time = datetime.now()
        processing_duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ [TAREA CACHEADA {task_id}] Procesamiento completado en {processing_duration:.2f} segundos")
        
        # Agregar información adicional al resultado
        result.update({
            "task_id": task_id,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "processing_duration_seconds": round(processing_duration, 2),
            "video_folder": video_folder,
            "config_used": final_config,
            "cached": False
        })
        
        # Guardar en cache si está disponible
        if CACHE_AVAILABLE:
            set_video_processing_cache(video_folder, final_config, result, ttl=1800)  # 30 minutos
            logger.info(f"💾 Resultado guardado en cache: {video_folder}")
        
        return result
        
    except Exception as e:
        error_msg = f"❌ Error en procesamiento: {str(e)}"
        logger.error(f"[TAREA CACHEADA {task_id}] {error_msg}")
        
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
            "error": str(e),
            "cached": False
        }

# Tarea process_unified_pipeline_cached eliminada

# ============================================================================
# PIPELINE COMPLETO OPTIMIZADO
# ============================================================================

# Tarea process_complete_pipeline_optimized eliminada (usaba unified pipeline)

if __name__ == '__main__':
    app.start()
