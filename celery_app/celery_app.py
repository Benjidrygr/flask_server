#!/usr/bin/env python3
"""
Celery App Configuration
"""

from celery import Celery
from .celery_config import CELERY_CONFIG

# Crear la aplicación Celery
app = Celery('yolo_model')

# Configurar Celery
app.config_from_object(CELERY_CONFIG)

# Auto-descubrir tareas
app.autodiscover_tasks(['celery'])

if __name__ == '__main__':
    app.start()
