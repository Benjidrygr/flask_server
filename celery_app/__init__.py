"""
Módulo Celery para procesamiento de videos YOLO
"""

# Solo importar configuraciones para evitar importaciones circulares
from .celery_config import CELERY_CONFIG, YOLO_CONFIG

__all__ = [
    'CELERY_CONFIG',
    'YOLO_CONFIG'
]
