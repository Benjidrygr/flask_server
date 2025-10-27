#!/usr/bin/env python3
"""
Gestor Completo de Celery - Ejecuta todo el flujo automáticamente
Incluye: Redis, Worker, Cliente, y limpieza automática
"""

import os
import sys
import time
import signal
import subprocess
import threading
import json
import argparse
from datetime import datetime
from typing import Optional, List, Dict, Any
import psutil

class CeleryManager:
    """Gestor completo del flujo de Celery"""
    
    def __init__(self):
        self.redis_process = None
        self.celery_worker_process = None
        self.flower_process = None
        self.running = True
        
        # Configurar manejo de señales para limpieza
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Manejar señales de interrupción para limpieza"""
        print(f"\n🛑 Señal {signum} recibida. Cerrando servicios...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def check_redis_running(self) -> bool:
        """Verificar si Redis está corriendo"""
        try:
            result = subprocess.run(['redis-cli', 'ping'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0 and 'PONG' in result.stdout
        except:
            return False
    
    def start_redis(self) -> bool:
        """Iniciar Redis si no está corriendo"""
        if self.check_redis_running():
            print("✅ Redis ya está corriendo")
            return True
        
        print("📡 Iniciando Redis...")
        try:
            # Buscar Redis en diferentes ubicaciones
            redis_paths = [
                'redis-server',
                '/usr/local/bin/redis-server',
                '/opt/homebrew/bin/redis-server',
                '/usr/bin/redis-server'
            ]
            
            redis_cmd = None
            for path in redis_paths:
                try:
                    subprocess.run([path, '--version'], 
                                 capture_output=True, check=True)
                    redis_cmd = path
                    break
                except:
                    continue
            
            if not redis_cmd:
                print("❌ Redis no encontrado. Instálalo con:")
                print("   macOS: brew install redis")
                print("   Ubuntu: sudo apt install redis-server")
                return False
            
            # Iniciar Redis en segundo plano
            self.redis_process = subprocess.Popen(
                [redis_cmd],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Esperar a que Redis esté listo
            for i in range(10):
                if self.check_redis_running():
                    print("✅ Redis iniciado correctamente")
                    return True
                time.sleep(1)
            
            print("❌ Error: Redis no se pudo iniciar")
            return False
            
        except Exception as e:
            print(f"❌ Error iniciando Redis: {e}")
            return False
    
    def start_celery_worker(self, concurrency: int = 1) -> bool:
        """Iniciar worker de Celery"""
        print(f"👷 Iniciando worker de Celery (concurrency={concurrency})...")
        
        try:
            # Usar el servidor directamente con PYTHONPATH configurado
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()
            
            self.celery_worker_process = subprocess.Popen(
                [sys.executable, 'celery_app/celery_server.py', '--worker', f'--concurrency={concurrency}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )
            
            # Esperar a que el worker se inicialice
            print("⏳ Esperando a que el worker se inicialice...")
            for i in range(15):
                if not self.running:
                    return False
                
                # Verificar si el proceso sigue corriendo
                if self.celery_worker_process.poll() is not None:
                    print("❌ Error: Worker de Celery falló al iniciar")
                    return False
                
                # Verificar salud del servidor
                if self.check_celery_health():
                    print("✅ Worker de Celery iniciado correctamente")
                    return True
                
                time.sleep(2)
            
            print("❌ Error: Worker de Celery no respondió a tiempo")
            return False
            
        except Exception as e:
            print(f"❌ Error iniciando worker de Celery: {e}")
            return False
    
    def start_flower(self) -> bool:
        """Iniciar Flower (monitor web)"""
        print("🌸 Iniciando Flower (monitor web)...")
        
        try:
            self.flower_process = subprocess.Popen(
                [sys.executable, 'celery_app/celery_server.py', '--flower'],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            time.sleep(3)  # Dar tiempo a Flower para iniciar
            print("✅ Flower iniciado en http://localhost:5555")
            return True
            
        except Exception as e:
            print(f"⚠️ Error iniciando Flower: {e}")
            return False
    
    def check_celery_health(self) -> bool:
        """Verificar salud del servidor Celery"""
        try:
            # Usar el cliente simple para verificar salud
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()
            
            result = subprocess.run([
                sys.executable, 'celery_app/celery_simple_client.py', '--video-folder', 'dummy', '--task', 'log_start'
            ], capture_output=True, text=True, timeout=10, env=env)
            
            return result.returncode == 0
        except:
            return False
    
    def create_test_videos(self) -> str:
        """Crear carpeta de videos de prueba"""
        test_dir = "videos_test"
        os.makedirs(test_dir, exist_ok=True)
        
        # Crear videos dummy
        for i in range(3):
            with open(f"{test_dir}/test_video_{i+1}.mp4", "w") as f:
                f.write(f"dummy video content {i+1}")
        
        print(f"📁 Carpeta de videos de prueba creada: {test_dir}")
        return test_dir
    
    def run_unified_pipeline_direct(self, video_folder: str, config_file: str = None, 
                                   unified_params: Dict[str, Any] = None) -> bool:
        """Ejecutar unified pipeline directamente (sin Celery)"""
        print(f"🌐 Ejecutando unified pipeline directamente...")
        
        try:
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()
            
            cmd = [
                sys.executable, 'coordinates_processor/unified_pipeline_simple.py',
                '--video-dir', video_folder,
                '--output', './unified_output'
            ]
            
            # Agregar archivo de configuración si se especifica
            if config_file and os.path.exists(config_file):
                cmd.extend(['--config', config_file])
            
            # Agregar parámetros adicionales si se especifican
            if unified_params:
                if unified_params.get('image_dir'):
                    cmd.extend(['--image-dir', unified_params['image_dir']])
                if unified_params.get('coordinates'):
                    cmd.extend(['--coordinates', unified_params['coordinates']])
                if unified_params.get('api_camera_id'):
                    cmd.extend(['--api-camera-id', unified_params['api_camera_id']])
                if unified_params.get('api_begin_timestamp'):
                    cmd.extend(['--api-begin-timestamp', str(unified_params['api_begin_timestamp'])])
                if unified_params.get('api_end_timestamp'):
                    cmd.extend(['--api-end-timestamp', str(unified_params['api_end_timestamp'])])
                if unified_params.get('api_sid'):
                    cmd.extend(['--api-sid', unified_params['api_sid']])
                if unified_params.get('api_server'):
                    cmd.extend(['--api-server', unified_params['api_server']])
                if unified_params.get('api_url'):
                    cmd.extend(['--api-url', unified_params['api_url']])
                if unified_params.get('enable_aws_download'):
                    cmd.append('--enable-aws-download')
                if unified_params.get('aws_stream_name'):
                    cmd.extend(['--aws-stream-name', unified_params['aws_stream_name']])
                if unified_params.get('aws_trip_ids'):
                    cmd.extend(['--aws-trip-ids', unified_params['aws_trip_ids']])
                if unified_params.get('aws_session_id'):
                    cmd.extend(['--aws-session-id', unified_params['aws_session_id']])
                if unified_params.get('aws_max_downloads'):
                    cmd.extend(['--aws-max-downloads', str(unified_params['aws_max_downloads'])])
                if unified_params.get('aws_timeout'):
                    cmd.extend(['--aws-timeout', str(unified_params['aws_timeout'])])
                if unified_params.get('aws_download_only'):
                    cmd.append('--aws-download-only')
                if unified_params.get('verbose'):
                    cmd.append('--verbose')
            
            print(f"🚀 Comando: {' '.join(cmd)}")
            
            # Ejecutar pipeline
            result = subprocess.run(cmd, timeout=300, env=env)
            
            if result.returncode == 0:
                print("✅ Unified pipeline ejecutado exitosamente")
                return True
            else:
                print(f"❌ Unified pipeline falló con código: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Unified pipeline timeout después de 300 segundos")
            return False
        except Exception as e:
            print(f"❌ Error ejecutando unified pipeline: {e}")
            return False

    def run_pipeline(self, video_folder: str, task_type: str = "optimized_pipeline", 
                    monitor: bool = True, timeout: int = 300, config_file: str = None,
                    unified_params: Dict[str, Any] = None) -> bool:
        """Ejecutar pipeline de Celery"""
        print(f"🎬 Ejecutando pipeline: {task_type}")
        
        # Si es unified_pipeline, ejecutar directamente
        if task_type == "unified_pipeline_direct":
            return self.run_unified_pipeline_direct(video_folder, config_file, unified_params)
        
        try:
            # Usar el cliente simple que funciona mejor
            env = os.environ.copy()
            env['PYTHONPATH'] = os.getcwd()
            
            cmd = [
                sys.executable, 'celery_app/celery_simple_client.py',
                '--video-folder', video_folder,
                '--task', task_type
            ]
            
            # Agregar archivo de configuración si se especifica
            if config_file and os.path.exists(config_file):
                cmd.extend(['--config', config_file])
            
            # El cliente simple no soporta --monitor y --timeout
            # if monitor:
            #     cmd.append('--monitor')
            #     cmd.extend(['--timeout', str(timeout)])
            
            print(f"🚀 Comando: {' '.join(cmd)}")
            
            # Ejecutar pipeline
            result = subprocess.run(cmd, timeout=timeout, env=env)
            
            if result.returncode == 0:
                print("✅ Pipeline ejecutado exitosamente")
                return True
            else:
                print(f"❌ Pipeline falló con código: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Pipeline timeout después de {timeout} segundos")
            return False
        except Exception as e:
            print(f"❌ Error ejecutando pipeline: {e}")
            return False
    
    def cleanup(self):
        """Limpiar todos los procesos"""
        print("🧹 Limpiando procesos...")
        
        # Terminar Flower
        if self.flower_process and self.flower_process.poll() is None:
            print("🌸 Cerrando Flower...")
            self.flower_process.terminate()
            try:
                self.flower_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.flower_process.kill()
        
        # Terminar Worker de Celery
        if self.celery_worker_process and self.celery_worker_process.poll() is None:
            print("👷 Cerrando worker de Celery...")
            self.celery_worker_process.terminate()
            try:
                self.celery_worker_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.celery_worker_process.kill()
        
        # Terminar Redis
        if self.redis_process and self.redis_process.poll() is None:
            print("📡 Cerrando Redis...")
            self.redis_process.terminate()
            try:
                self.redis_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.redis_process.kill()
        
        # Limpiar archivos de prueba
        if os.path.exists("videos_test"):
            print("🗑️ Eliminando archivos de prueba...")
            import shutil
            shutil.rmtree("videos_test", ignore_errors=True)
        
        print("✅ Limpieza completada")
    
    def run_interactive_mode(self):
        """Modo interactivo para seleccionar tareas"""
        while self.running:
            print("\n" + "="*60)
            print("🎯 CELERY MANAGER - Modo Interactivo")
            print("="*60)
            print("1. Pipeline completo optimizado (unified + YOLO)")
            print("2. Solo pipeline unificado (Celery)")
            print("3. Solo YOLO")
            print("4. Datos mixtos")
            print("5. Unified Pipeline Directo (sin Celery)")
            print("6. Medir overhead")
            print("7. Verificar salud")
            print("8. Iniciar Flower")
            print("9. Salir")
            print("="*60)
            
            try:
                choice = input("Selecciona una opción (1-9): ").strip()
                
                if choice == "1":
                    video_folder = self.create_test_videos()
                    # Usar configuración unificada para pipeline optimizado
                    unified_config = "celery_app/unified_config_example.json"
                    self.run_pipeline(video_folder, "optimized_pipeline", config_file=unified_config)
                elif choice == "2":
                    video_folder = self.create_test_videos()
                    # Usar configuración unificada para pipeline unificado
                    unified_config = "celery_app/unified_config_example.json"
                    self.run_pipeline(video_folder, "unified_pipeline", config_file=unified_config)
                elif choice == "3":
                    video_folder = self.create_test_videos()
                    self.run_pipeline(video_folder, "process_videos")
                elif choice == "4":
                    video_folder = self.create_test_videos()
                    # Usar configuración unificada para datos mixtos
                    unified_config = "celery_app/unified_config_example.json"
                    self.run_pipeline(video_folder, "mixed_data", config_file=unified_config)
                elif choice == "5":
                    video_folder = self.create_test_videos()
                    # Unified Pipeline Directo (sin Celery)
                    unified_config = "celery_app/unified_pipeline_config.json"
                    self.run_pipeline(video_folder, "unified_pipeline_direct", config_file=unified_config)
                elif choice == "6":
                    print("📊 Ejecutando medición de overhead...")
                    subprocess.run([sys.executable, 'celery_app/measure_overhead.py'])
                elif choice == "7":
                    if self.check_celery_health():
                        print("✅ Servidor Celery funcionando correctamente")
                    else:
                        print("❌ Servidor Celery no responde")
                elif choice == "8":
                    if not self.flower_process or self.flower_process.poll() is not None:
                        self.start_flower()
                    else:
                        print("🌸 Flower ya está corriendo")
                elif choice == "9":
                    print("👋 Saliendo...")
                    break
                else:
                    print("❌ Opción inválida")
                    
            except KeyboardInterrupt:
                print("\n👋 Saliendo...")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def run_automatic_mode(self, task_type: str = "optimized_pipeline"):
        """Modo automático - ejecutar una tarea específica"""
        print(f"🤖 Modo automático: {task_type}")
        
        # Iniciar servicios
        if not self.start_redis():
            return False
        
        if not self.start_celery_worker():
            return False
        
        # Crear videos de prueba
        video_folder = self.create_test_videos()
        
        # Ejecutar pipeline con configuración apropiada
        config_file = None
        if task_type in ['optimized_pipeline', 'unified_pipeline', 'mixed_data']:
            config_file = "celery_app/unified_config_example.json"
        
        success = self.run_pipeline(video_folder, task_type, config_file=config_file)
        
        return success

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Gestor Completo de Celery")
    parser.add_argument('--mode', choices=['interactive', 'auto'], default='interactive',
                       help='Modo de ejecución')
    parser.add_argument('--task', choices=['optimized_pipeline', 'unified_pipeline', 'process_videos', 'mixed_data', 'unified_pipeline_direct'],
                       default='optimized_pipeline', help='Tarea a ejecutar en modo automático')
    parser.add_argument('--concurrency', type=int, default=1,
                       help='Número de workers concurrentes')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout para las tareas (segundos)')
    parser.add_argument('--flower', action='store_true',
                       help='Iniciar Flower automáticamente')
    
    args = parser.parse_args()
    
    print("🚀 CELERY MANAGER - Iniciando...")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    manager = CeleryManager()
    
    try:
        if args.mode == 'interactive':
            # Iniciar servicios básicos
            if not manager.start_redis():
                return 1
            
            if not manager.start_celery_worker(args.concurrency):
                return 1
            
            if args.flower:
                manager.start_flower()
            
            # Modo interactivo
            manager.run_interactive_mode()
            
        else:  # modo automático
            success = manager.run_automatic_mode(args.task)
            if not success:
                return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Interrumpido por el usuario")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        return 1
    finally:
        manager.cleanup()
    
    print("✅ CELERY MANAGER - Finalizado")
    return 0

if __name__ == "__main__":
    sys.exit(main())
