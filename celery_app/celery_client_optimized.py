#!/usr/bin/env python3
"""
Cliente OPTIMIZADO para enviar tareas al servidor Celery YOLO
Incluye optimizaciones de overhead y cache
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
app = Celery('yolo_processor_optimized')
app.config_from_object(CELERY_CONFIG)

# Importar tareas optimizadas
from .celery_tasks_optimized import (
    process_videos_yolo_cached
)

# Importar sistema de cache
try:
    from .cache_manager import cache_manager, CACHE_AVAILABLE
except ImportError:
    CACHE_AVAILABLE = False
    print("⚠️ Sistema de cache no disponible")

class YOLOCeleryClientOptimized:
    """Cliente optimizado para interactuar con el servidor Celery YOLO"""
    
    def __init__(self):
        self.app = app
        self.active_tasks = {}
        self.cache_available = CACHE_AVAILABLE
    
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
                    print(f"      - Tareas activas: {worker_stats.get('total', {}).get('celery_tasks_optimized.process_videos_yolo_cached', 0)}")
                
                # Mostrar estadísticas de cache si está disponible
                if self.cache_available:
                    cache_stats = cache_manager.get_stats()
                    print(f"   💾 Cache Redis: {cache_stats.get('cache_keys', 0)} claves")
                    print(f"      - Memoria usada: {cache_stats.get('used_memory', 'N/A')}")
                else:
                    print("   💾 Cache: No disponible")
                
                return True
            else:
                print("❌ No hay workers activos")
                return False
                
        except Exception as e:
            print(f"❌ Error conectando al servidor Celery: {e}")
            return False
    
    # Función send_optimized_pipeline eliminada (usaba unified pipeline)
    
    def send_cached_yolo_task(self, video_folder: str, config: Optional[Dict[str, Any]] = None) -> str:
        """Enviar tarea YOLO con cache"""
        print("🎬 Enviando tarea YOLO con cache")
        
        result = process_videos_yolo_cached.delay(video_folder, config)
        task_id = result.id
        
        self.active_tasks[task_id] = {
            'task_name': 'yolo_cached',
            'result': result,
            'start_time': datetime.now()
        }
        
        print(f"✅ Tarea YOLO con cache enviada. ID: {task_id}")
        return task_id
    
    # Función send_cached_unified_task eliminada (usaba unified pipeline)
    
    def monitor_task(self, task_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Monitorear el progreso de una tarea con información de cache"""
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
            
            # Mostrar información de cache si está disponible
            if isinstance(task_result, dict) and 'cached' in task_result:
                if task_result['cached']:
                    print("🎯 Resultado obtenido del cache")
                else:
                    print("🔄 Resultado procesado (no en cache)")
            
            return {
                'task_id': task_id,
                'task_name': task_info['task_name'],
                'duration_seconds': duration,
                'result': task_result,
                'success': task_result.get('success', False),
                'cached': task_result.get('cached', False) if isinstance(task_result, dict) else False
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
                if isinstance(result.result, dict) and 'cached' in result.result:
                    status_info['cached'] = result.result['cached']
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
        """Listar todas las tareas activas con información de cache"""
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
            if status.get('cached'):
                print(f"      - Cache: 🎯 Hit")
            if status.get('progress'):
                print(f"      - Progreso: {status['progress']}")
            print()
    
    def clear_cache(self):
        """Limpiar cache si está disponible"""
        if not self.cache_available:
            print("❌ Sistema de cache no disponible")
            return False
        
        try:
            success = cache_manager.clear_all()
            if success:
                print("✅ Cache limpiado exitosamente")
            else:
                print("❌ Error limpiando cache")
            return success
        except Exception as e:
            print(f"❌ Error limpiando cache: {e}")
            return False
    
    def get_cache_stats(self):
        """Obtener estadísticas del cache"""
        if not self.cache_available:
            print("❌ Sistema de cache no disponible")
            return
        
        try:
            stats = cache_manager.get_stats()
            print("📊 Estadísticas del Cache:")
            print(f"   - Versión Redis: {stats.get('redis_version', 'N/A')}")
            print(f"   - Memoria usada: {stats.get('used_memory', 'N/A')}")
            print(f"   - Clientes conectados: {stats.get('connected_clients', 'N/A')}")
            print(f"   - Claves totales: {stats.get('total_keys', 'N/A')}")
            print(f"   - Claves de cache: {stats.get('cache_keys', 'N/A')}")
            print(f"   - Tiempo activo: {stats.get('uptime_seconds', 'N/A')} segundos")
        except Exception as e:
            print(f"❌ Error obteniendo estadísticas: {e}")

def main():
    parser = argparse.ArgumentParser(description='Cliente OPTIMIZADO para servidor Celery YOLO')
    parser.add_argument('--video-folder', type=str, required=False,
                       help='Carpeta que contiene los videos a procesar')
    parser.add_argument('--task', type=str, 
                       choices=['optimized_pipeline', 'yolo_cached', 'unified_cached'],
                       default='optimized_pipeline',
                       help='Tipo de tarea optimizada a enviar')
    parser.add_argument('--config', type=str,
                       help='Archivo JSON con configuración personalizada para YOLO')
    parser.add_argument('--unified-config', type=str,
                       help='Archivo JSON con configuración personalizada para pipeline unificado')
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
    parser.add_argument('--clear-cache', action='store_true',
                       help='Limpiar cache')
    parser.add_argument('--cache-stats', action='store_true',
                       help='Mostrar estadísticas del cache')
    
    args = parser.parse_args()
    
    # Crear cliente optimizado
    client = YOLOCeleryClientOptimized()
    
    # Verificar salud del servidor
    if args.health_check:
        client.check_server_health()
        return
    
    # Limpiar cache
    if args.clear_cache:
        client.clear_cache()
        return
    
    # Mostrar estadísticas del cache
    if args.cache_stats:
        client.get_cache_stats()
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
    if not args.health_check and not args.list_tasks and not args.status and not args.clear_cache and not args.cache_stats and not args.video_folder:
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
        with open(args.config, 'r', encoding='utf-8') as f:
            custom_config = json.load(f)
            yolo_config.update(custom_config)
        print(f"⚙️ Configuración YOLO cargada desde: {args.config}")
    
    # Cargar configuración unificada
    unified_config = None
    if args.unified_config and os.path.exists(args.unified_config):
        with open(args.unified_config, 'r', encoding='utf-8') as f:
            unified_config = json.load(f)
        print(f"🌐 Configuración unificada cargada desde: {args.unified_config}")
    
    # Verificar que la carpeta de videos existe (solo si se proporciona)
    if args.video_folder and not os.path.exists(args.video_folder):
        print(f"❌ La carpeta {args.video_folder} no existe")
        return
    
    # Enviar tarea optimizada
    if args.task == 'yolo_cached':
        task_id = client.send_cached_yolo_task(args.video_folder, yolo_config)
    # Opción unified_cached eliminada (usaba unified pipeline)
    else:
        print(f"❌ Tarea desconocida: {args.task}")
        return
    
    if not task_id:
        return
    
    # Monitorear si se solicita
    if args.monitor:
        result = client.monitor_task(task_id, args.timeout)
        if result:
            print("\n📊 Resultado final:")
            print(json.dumps(result['result'], indent=2, default=str))
            
            # Mostrar información de cache
            if result.get('cached'):
                print("\n🎯 Este resultado fue obtenido del cache (overhead reducido)")
            else:
                print("\n🔄 Este resultado fue procesado (guardado en cache para futuras consultas)")
            
            # Guardar resultado
            output_file = f"celery_optimized_result_{task_id}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str, ensure_ascii=False)
            print(f"💾 Resultado guardado en: {output_file}")
    else:
        print(f"\n💡 Para monitorear esta tarea:")
        print(f"   python celery_client_optimized.py --status {task_id}")
        print(f"   python celery_client_optimized.py --list-tasks")

if __name__ == '__main__':
    main()
