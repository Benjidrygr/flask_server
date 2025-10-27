"""
Cliente simplificado para ejecutar las 3 tareas en secuencia
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
from celery import Celery

# Importar configuración
from celery_config import CELERY_CONFIG, YOLO_CONFIG

# Crear instancia de Celery
app = Celery('yolo_processor')
app.config_from_object(CELERY_CONFIG)

# Importar tareas
from celery_tasks import (
    log_start_yolo,
    process_videos_yolo,
    log_end_yolo,
    process_videos_complete_pipeline
)

def execute_pipeline_sequence(video_folder: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Ejecutar las 3 tareas en secuencia desde el cliente
    """
    print("🔄 Iniciando pipeline completo de procesamiento YOLO")
    
    if config is None:
        config = YOLO_CONFIG.copy()
    
    pipeline_start_time = datetime.now()
    results = {}
    
    try:
        # Paso 1: Log de inicio
        print("📝 Paso 1/3: Ejecutando log de inicio...")
        start_task = log_start_yolo.delay(video_folder, config)
        start_log = start_task.get(timeout=60)
        results['start_log'] = start_log
        
        if not start_log.get("success", False):
            raise Exception(f"Error en log de inicio: {start_log.get('message', 'Error desconocido')}")
        
        print(f"✅ Log de inicio completado: {start_log.get('total_videos', 0)} videos encontrados")
        
        # Paso 2: Procesamiento de videos
        print("🎬 Paso 2/3: Ejecutando procesamiento de videos...")
        processing_task = process_videos_yolo.delay(video_folder, config)
        processing = processing_task.get(timeout=7200)  # 2 horas timeout
        results['processing_result'] = processing
        
        print("✅ Procesamiento de videos completado")
        
        # Paso 3: Log de finalización
        print("📝 Paso 3/3: Ejecutando log de finalización...")
        end_task = log_end_yolo.delay(processing)
        end_log = end_task.get(timeout=60)
        results['end_log'] = end_log
        
        print("✅ Log de finalización completado")
        
        # Resultado final
        pipeline_end_time = datetime.now()
        pipeline_duration = (pipeline_end_time - pipeline_start_time).total_seconds()
        
        final_result = {
            "success": True,
            "message": "Pipeline completo ejecutado exitosamente",
            "pipeline_duration_seconds": round(pipeline_duration, 2),
            "video_folder": video_folder,
            "config_used": config,
            "results": results
        }
        
        print(f"🎉 Pipeline completo finalizado en {pipeline_duration:.2f} segundos")
        return final_result
        
    except Exception as e:
        error_msg = f"❌ Error en pipeline: {str(e)}"
        print(error_msg)
        
        return {
            "success": False,
            "message": error_msg,
            "video_folder": video_folder,
            "config_used": config,
            "error": str(e),
            "partial_results": results
        }

def main():
    parser = argparse.ArgumentParser(description='Cliente simplificado para pipeline YOLO')
    parser.add_argument('--video-folder', type=str, required=True,
                       help='Carpeta que contiene los videos a procesar')
    parser.add_argument('--config', type=str,
                       help='Archivo JSON con configuración personalizada')
    parser.add_argument('--task', type=str, 
                       choices=['log_start', 'process_videos', 'log_end', 'complete_pipeline', 'optimized_pipeline'],
                       default='complete_pipeline',
                       help='Tipo de tarea a enviar')
    
    args = parser.parse_args()
    
    # Verificar que la carpeta existe
    if not os.path.exists(args.video_folder):
        print(f"❌ La carpeta {args.video_folder} no existe")
        return
    
    # Cargar configuración
    config = YOLO_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            config.update(custom_config)
        print(f"⚙️ Configuración cargada desde: {args.config}")
    
    # Verificar salud del servidor
    try:
        inspect = app.control.inspect()
        stats = inspect.stats()
        if not stats:
            print("❌ No hay workers activos")
            return
        print("✅ Servidor Celery está funcionando")
    except Exception as e:
        print(f"❌ Error conectando al servidor Celery: {e}")
        return
    
    # Ejecutar tarea
    if args.task == 'complete_pipeline':
        result = execute_pipeline_sequence(args.video_folder, config)
    elif args.task == 'optimized_pipeline':
        # Usar el pipeline completo (solo YOLO)
        task = process_videos_complete_pipeline.delay(args.video_folder, config)
        result = {"success": True, "result": task.get(timeout=7200)}
    elif args.task == 'log_start':
        task = log_start_yolo.delay(args.video_folder, config)
        result = {"success": True, "result": task.get(timeout=60)}
    elif args.task == 'process_videos':
        task = process_videos_yolo.delay(args.video_folder, config)
        result = {"success": True, "result": task.get(timeout=7200)}
    elif args.task == 'log_end':
        print("⚠️ log_end requiere el resultado de process_videos")
        return
    
    # Mostrar resultado
    if result.get("success", False):
        print("\n✅ Operación completada exitosamente")
        print("📊 Resultado:")
        print(json.dumps(result, indent=2, default=str))
        
        # Guardar resultado
        output_file = f"simple_result_{int(time.time())}.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"💾 Resultado guardado en: {output_file}")
    else:
        print(f"\n❌ Error: {result.get('message', 'Error desconocido')}")
        sys.exit(1)

if __name__ == '__main__':
    main()
