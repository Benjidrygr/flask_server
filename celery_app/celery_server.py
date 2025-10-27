"""
Servidor Celery para el procesador de videos YOLO
"""

import os
import sys
import logging
from celery import Celery
from celery_config import CELERY_CONFIG

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
)
logger = logging.getLogger(__name__)

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

@app.task
def health_check():
    """Tarea de verificación de salud del servidor"""
    return {
        "status": "healthy",
        "message": "Servidor YOLO Celery funcionando correctamente",
        "timestamp": str(datetime.now())
    }

if __name__ == '__main__':
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description='Servidor Celery para procesamiento YOLO')
    parser.add_argument('--worker', action='store_true', 
                       help='Iniciar como worker')
    parser.add_argument('--beat', action='store_true',
                       help='Iniciar como scheduler (beat)')
    parser.add_argument('--flower', action='store_true',
                       help='Iniciar Flower (monitor web)')
    parser.add_argument('--concurrency', type=int, default=1,
                       help='Número de workers concurrentes')
    parser.add_argument('--loglevel', type=str, default='info',
                       choices=['debug', 'info', 'warning', 'error'],
                       help='Nivel de logging')
    
    args = parser.parse_args()
    
    logger.info("🚀 Iniciando servidor Celery para procesamiento YOLO")
    logger.info(f"📊 Configuración:")
    logger.info(f"   - Broker: {CELERY_CONFIG['broker_url']}")
    logger.info(f"   - Backend: {CELERY_CONFIG['result_backend']}")
    logger.info(f"   - Concurrencia: {args.concurrency}")
    logger.info(f"   - Log level: {args.loglevel}")
    
    if args.worker:
        logger.info("👷 Iniciando worker...")
        app.worker_main([
            'worker',
            '--loglevel=' + args.loglevel,
            '--concurrency=' + str(args.concurrency),
            '--queues=yolo_queue,default',
            '--hostname=yolo-worker@%h'
        ])
    elif args.beat:
        logger.info("⏰ Iniciando scheduler (beat)...")
        app.control.purge()  # Limpiar colas
        app.start(['beat', '--loglevel=' + args.loglevel])
    elif args.flower:
        logger.info("🌸 Iniciando Flower (monitor web)...")
        try:
            from flower.command import FlowerCommand
            flower = FlowerCommand()
            flower.run_from_argv(['flower', '--port=5555', '--broker=' + CELERY_CONFIG['broker_url']])
        except ImportError:
            logger.error("❌ Flower no está instalado. Instálalo con: pip install flower")
            sys.exit(1)
    else:
        logger.info("ℹ️ Modo de ayuda. Usa --worker, --beat o --flower")
        logger.info("📖 Comandos disponibles:")
        logger.info("   python celery_server.py --worker --concurrency 2")
        logger.info("   python celery_server.py --beat")
        logger.info("   python celery_server.py --flower")
        logger.info("   python celery_server.py --worker --loglevel debug")
