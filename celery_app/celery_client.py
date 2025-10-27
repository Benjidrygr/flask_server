"""
Cliente para enviar tareas al servidor Celery YOLO
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, Optional
from celery import Celery
from celery.result import AsyncResult

# Importar configuración
from .celery_config import CELERY_CONFIG, YOLO_CONFIG

# Crear instancia de Celery
app = Celery('yolo_processor')
app.config_from_object(CELERY_CONFIG)

# Importar tareas
from .celery_tasks import (
    log_start_yolo,
    process_videos_yolo,
    log_end_yolo,
    process_videos_complete_pipeline,
)

class YOLOCeleryClient:
    """Cliente para interactuar con el servidor Celery YOLO"""
    
    def __init__(self):
        self.app = app
        self.active_tasks = {}
    
    def check_server_health(self) -> bool:
        """Verificar que el servidor Celery esté funcionando"""
        try:
            # Intentar obtener estadísticas del broker
            inspect = self.app.control.inspect()
            stats = inspect.stats()
            
            if stats:
                print("✅ Servidor Celery está funcionando")
                for worker, worker_stats in stats.items():
                    print(f"   👷 Worker: {worker}")
                    print(f"      - Tareas activas: {worker_stats.get('total', {}).get('celery_tasks.process_videos_yolo', 0)}")
                return True
            else:
                print("❌ No hay workers activos")
                return False
                
        except Exception as e:
            print(f"❌ Error conectando al servidor Celery: {e}")
            return False
    
    def send_single_task(self, task_name: str, video_folder: str, config: Optional[Dict[str, Any]] = None, **kwargs) -> str:
        """Enviar una tarea individual"""
        print(f"📤 Enviando tarea: {task_name}")
        
        if task_name == "log_start":
            result = log_start_yolo.delay(video_folder, config)
        elif task_name == "process_videos":
            result = process_videos_yolo.delay(video_folder, config)
        elif task_name == "log_end":
            # Para log_end necesitamos el resultado del procesamiento
            print("⚠️ log_end requiere el resultado de process_videos")
            return None
        else:
            print(f"❌ Tarea desconocida: {task_name}")
            return None
        
        task_id = result.id
        self.active_tasks[task_id] = {
            'task_name': task_name,
            'result': result,
            'start_time': datetime.now()
        }
        
        print(f"✅ Tarea enviada. ID: {task_id}")
        return task_id
    
    def send_complete_pipeline(self, video_folder: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Enviar pipeline completo (las 3 tareas en secuencia)"""
        print("🔄 Enviando pipeline completo de procesamiento YOLO")
        
        result = process_videos_complete_pipeline.delay(video_folder, config)
        task_id = result.id
        
        self.active_tasks[task_id] = {
            'task_name': 'complete_pipeline',
            'result': result,
            'start_time': datetime.now()
        }
        
        print(f"✅ Pipeline enviado. ID: {task_id}")
        return task_id
    
    def send_complete_pipeline_with_unified(self, video_folder: str, 
                                          unified_config: Optional[Dict[str, Any]] = None,
                                          yolo_config: Optional[Dict[str, Any]] = None) -> str:
        """Enviar pipeline completo con unificado + YOLO"""
        print("🌐 Enviando pipeline completo con unificado + YOLO")
        
        result = process_complete_pipeline_with_unified.delay(
            video_folder, unified_config, yolo_config
        )
        task_id = result.id
        
        self.active_tasks[task_id] = {
            'task_name': 'complete_pipeline_with_unified',
            'result': result,
            'start_time': datetime.now()
        }
        
        print(f"✅ Pipeline completo enviado. ID: {task_id}")
        return task_id
    
    def monitor_task(self, task_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Monitorear el progreso de una tarea"""
        if task_id not in self.active_tasks:
            print(f"❌ Tarea {task_id} no encontrada")
            return None
        
        task_info = self.active_tasks[task_id]
        result = task_info['result']
        
        print(f"👀 Monitoreando tarea: {task_info['task_name']} (ID: {task_id})")
        print("⏳ Esperando resultado... (Ctrl+C para cancelar)")
        
        try:
            # Obtener resultado con timeout
            task_result = result.get(timeout=timeout)
            
            # Calcular duración
            duration = (datetime.now() - task_info['start_time']).total_seconds()
            
            print(f"✅ Tarea completada en {duration:.2f} segundos")
            return {
                'task_id': task_id,
                'task_name': task_info['task_name'],
                'duration_seconds': duration,
                'result': task_result,
                'success': task_result.get('success', False)
            }
            
        except KeyboardInterrupt:
            print("\n⚠️ Monitoreo cancelado por el usuario")
            return None
        except Exception as e:
            print(f"❌ Error obteniendo resultado: {e}")
            return None
    
    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Obtener estado de una tarea"""
        if task_id not in self.active_tasks:
            print(f"❌ Tarea {task_id} no encontrada")
            return None
        
        result = self.active_tasks[task_id]['result']
        state = result.state
        
        status_info = {
            'task_id': task_id,
            'state': state,
            'ready': result.ready(),
            'successful': result.successful() if result.ready() else False,
            'failed': result.failed() if result.ready() else False
        }
        
        if result.ready():
            if result.successful():
                status_info['result'] = result.result
            elif result.failed():
                status_info['error'] = str(result.result)
        else:
            # Tarea en progreso
            try:
                status_info['progress'] = result.info
            except:
                pass
        
        return status_info
    
    def list_active_tasks(self):
        """Listar todas las tareas activas"""
        if not self.active_tasks:
            print("📭 No hay tareas activas")
            return
        
        print(f"📋 Tareas activas ({len(self.active_tasks)}):")
        for task_id, task_info in self.active_tasks.items():
            duration = (datetime.now() - task_info['start_time']).total_seconds()
            status = self.get_task_status(task_id)
            
            print(f"   🆔 {task_id}")
            print(f"      - Tarea: {task_info['task_name']}")
            print(f"      - Estado: {status['state']}")
            print(f"      - Duración: {duration:.2f}s")
            if status.get('progress'):
                print(f"      - Progreso: {status['progress']}")
            print()

def main():
    parser = argparse.ArgumentParser(description='Cliente para servidor Celery YOLO')
    parser.add_argument('--video-folder', type=str, required=False,
                       help='Carpeta que contiene los videos a procesar')
    parser.add_argument('--task', type=str, 
                       choices=['log_start', 'process_videos', 'log_end', 'complete_pipeline'],
                       default='complete_pipeline',
                       help='Tipo de tarea a enviar')
    parser.add_argument('--config', type=str,
                       help='Archivo JSON con configuración personalizada para YOLO')
    parser.add_argument('--monitor', action='store_true',
                       help='Monitorear la tarea hasta completarse')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout en segundos para monitoreo (default: 3600)')
    parser.add_argument('--health-check', action='store_true',
                       help='Solo verificar salud del servidor')
    parser.add_argument('--list-tasks', action='store_true',
                       help='Listar tareas activas')
    parser.add_argument('--status', type=str,
                       help='Obtener estado de una tarea específica (ID)')
    
    args = parser.parse_args()
    
    # Crear cliente
    client = YOLOCeleryClient()
    
    # Verificar salud del servidor
    if args.health_check:
        client.check_server_health()
        return
    
    # Listar tareas activas
    if args.list_tasks:
        client.list_active_tasks()
        return
    
    # Obtener estado de tarea específica
    if args.status:
        status = client.get_task_status(args.status)
        if status:
            print(f"📊 Estado de tarea {args.status}:")
            print(json.dumps(status, indent=2, default=str))
        return
    
    # Verificar que video-folder esté proporcionado para tareas que lo requieren
    if not args.health_check and not args.list_tasks and not args.status and not args.video_folder:
        print("❌ --video-folder es requerido para esta operación")
        return
    
    # Verificar que el servidor esté funcionando
    if not client.check_server_health():
        print("❌ No se puede conectar al servidor Celery")
        print("💡 Asegúrate de que el servidor esté ejecutándose:")
        print("   python celery_server.py --worker")
        return
    
    # Cargar configuración YOLO
    yolo_config = YOLO_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            custom_config = json.load(f)
            yolo_config.update(custom_config)
        print(f"⚙️ Configuración YOLO cargada desde: {args.config}")
    
    # Cargar configuración unificada
    unified_config = None
    if args.unified_config and os.path.exists(args.unified_config):
        with open(args.unified_config, 'r') as f:
            unified_config = json.load(f)
        print(f"🌐 Configuración unificada cargada desde: {args.unified_config}")
    
    # Verificar que la carpeta de videos existe (solo si se proporciona)
    if args.video_folder and not os.path.exists(args.video_folder):
        print(f"❌ La carpeta {args.video_folder} no existe")
        return
    
    # Enviar tarea
    if args.task == 'complete_pipeline':
        task_id = client.send_complete_pipeline(args.video_folder, yolo_config)
    elif args.task == 'complete_pipeline_with_unified':
        task_id = client.send_complete_pipeline_with_unified(args.video_folder, unified_config, yolo_config)
    else:
        task_id = client.send_single_task(args.task, args.video_folder, yolo_config, unified_config)
    
    if not task_id:
        return
    
    # Monitorear si se solicita
    if args.monitor:
        result = client.monitor_task(task_id, args.timeout)
        if result:
            print("\n📊 Resultado final:")
            print(json.dumps(result['result'], indent=2, default=str))
            
            # Guardar resultado
            output_file = f"celery_result_{task_id}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"💾 Resultado guardado en: {output_file}")
    else:
        print(f"\n💡 Para monitorear esta tarea:")
        print(f"   python celery_client.py --status {task_id}")
        print(f"   python celery_client.py --list-tasks")

if __name__ == '__main__':
    main()
