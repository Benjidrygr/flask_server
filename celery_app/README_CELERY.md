# Servidor Celery para Procesamiento YOLO

Este servidor Celery implementa un sistema de colas para procesar videos con YOLO de manera distribuida y escalable.

## 🏗️ Arquitectura

El sistema está compuesto por **3 tareas principales**:

1. **`log_start_yolo`** - Log de inicio del proceso YOLO
2. **`process_videos_yolo`** - Procesamiento de videos con YOLO
3. **`log_end_yolo`** - Log de finalización del proceso YOLO

## 📁 Archivos Creados

- `celery_config.py` - Configuración de Celery
- `celery_tasks.py` - Definición de las 3 tareas
- `celery_server.py` - Servidor principal de Celery
- `celery_client.py` - Cliente para enviar tareas
- `requirements_celery.txt` - Dependencias necesarias

## 🚀 Instalación

### 1. Instalar Redis (Broker)

**macOS:**
```bash
brew install redis
brew services start redis
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install redis-server
sudo systemctl start redis
```

**Windows:**
- Descargar desde https://redis.io/download
- Ejecutar `redis-server.exe`

### 2. Verificar Redis
```bash
redis-cli ping
# Debe responder: PONG
```

### 3. Instalar Dependencias
```bash
pip install -r requirements_celery.txt
```

## 🎯 Uso del Sistema

### Paso 1: Iniciar el Servidor Celery

```bash
# Iniciar worker (procesador de tareas)
python celery_server.py --worker --concurrency 2

# En otra terminal, iniciar monitor web (opcional)
python celery_server.py --flower
# Luego abrir: http://localhost:5555
```

### Paso 2: Enviar Tareas

```bash
# Pipeline completo (recomendado)
python celery_client.py --video-folder ./videos --task complete_pipeline --monitor

# Tareas individuales
python celery_client.py --video-folder ./videos --task log_start
python celery_client.py --video-folder ./videos --task process_videos
```

## 📊 Comandos del Cliente

### Verificar Salud del Servidor
```bash
python celery_client.py --health-check
```

### Enviar Pipeline Completo
```bash
python celery_client.py --video-folder ./videos --task complete_pipeline --monitor
```

### Enviar Tarea Individual
```bash
python celery_client.py --video-folder ./videos --task process_videos --monitor
```

### Monitorear Tarea Específica
```bash
python celery_client.py --status <TASK_ID>
```

### Listar Tareas Activas
```bash
python celery_client.py --list-tasks
```

### Usar Configuración Personalizada
```bash
# Crear archivo config.json
echo '{"skip_frames": true, "visualize": true, "max_age": 15}' > config.json

# Usar configuración personalizada
python celery_client.py --video-folder ./videos --config config.json --task complete_pipeline
```

## ⚙️ Configuración

### Variables de Entorno
```bash
export CELERY_BROKER_URL="redis://localhost:6379/0"
export CELERY_RESULT_BACKEND="redis://localhost:6379/0"
```

### Configuración YOLO (en celery_config.py)
```python
YOLO_CONFIG = {
    'max_age': 10,           # Parámetro SORT
    'min_hits': 3,           # Parámetro SORT
    'iou_threshold': 0.3,    # Parámetro SORT
    'skip_frames': False,    # Procesar todos los frames
    'visualize': True,       # Generar videos visualizados
    'output_folder': 'output_videos',
    'max_detections_in_memory': 10000,
    'save_intermediate_results': True,
}
```

## 🔄 Flujo de Trabajo

### Pipeline Completo
1. **Log de Inicio** → Registra inicio, verifica recursos, cuenta videos
2. **Procesamiento** → Ejecuta YOLO en todos los videos
3. **Log de Finalización** → Registra resultados, estadísticas finales

### Tareas Individuales
- Cada tarea puede ejecutarse por separado
- Útil para debugging o procesamiento específico

## 📈 Monitoreo

### Flower (Monitor Web)
```bash
python celery_server.py --flower
# Abrir: http://localhost:5555
```

### Logs del Worker
```bash
python celery_server.py --worker --loglevel debug
```

### Estado de Tareas
```bash
python celery_client.py --list-tasks
python celery_client.py --status <TASK_ID>
```

## 🛠️ Comandos Avanzados

### Múltiples Workers
```bash
# Terminal 1
python celery_server.py --worker --concurrency 2

# Terminal 2
python celery_server.py --worker --concurrency 2

# Terminal 3
python celery_server.py --worker --concurrency 2
```

### Scheduler (Beat)
```bash
python celery_server.py --beat
```

### Debug Mode
```bash
python celery_server.py --worker --loglevel debug
```

## 📋 Ejemplos de Uso

### Ejemplo 1: Procesamiento Básico
```bash
# Terminal 1: Iniciar worker
python celery_server.py --worker

# Terminal 2: Enviar tarea
python celery_client.py --video-folder ./videos --task complete_pipeline --monitor
```

### Ejemplo 2: Con Configuración Personalizada
```bash
# Crear configuración
cat > my_config.json << EOF
{
    "skip_frames": true,
    "visualize": true,
    "max_age": 15,
    "min_hits": 2,
    "iou_threshold": 0.4
}
EOF

# Enviar con configuración
python celery_client.py --video-folder ./videos --config my_config.json --task complete_pipeline --monitor
```

### Ejemplo 3: Solo Procesamiento (sin logs)
```bash
python celery_client.py --video-folder ./videos --task process_videos --monitor
```

## 🚨 Solución de Problemas

### Error: "No se puede conectar al servidor Celery"
```bash
# Verificar Redis
redis-cli ping

# Verificar workers
python celery_client.py --health-check
```

### Error: "No hay workers activos"
```bash
# Iniciar worker
python celery_server.py --worker
```

### Error: "FFmpeg no encontrado"
```bash
# macOS
brew install ffmpeg

# Ubuntu
sudo apt install ffmpeg
```

### Limpiar Colas
```bash
# En Python
from celery import Celery
app = Celery('yolo_processor')
app.control.purge()
```

## 📊 Estructura de Resultados

### Resultado del Pipeline Completo
```json
{
    "success": true,
    "message": "Pipeline completo ejecutado exitosamente",
    "pipeline_task_id": "uuid-here",
    "start_log": { ... },
    "processing_result": { ... },
    "end_log": { ... },
    "video_folder": "./videos",
    "config_used": { ... }
}
```

### Archivos Generados
- `celery_result_<TASK_ID>.json` - Resultado completo
- `output_videos/` - Videos visualizados (si visualize=true)

## 🔧 Personalización

### Agregar Nueva Tarea
1. Editar `celery_tasks.py`
2. Agregar función con decorador `@app.task`
3. Actualizar `celery_client.py` si es necesario

### Cambiar Broker
1. Editar `celery_config.py`
2. Cambiar `BROKER_URL` y `RESULT_BACKEND`
3. Instalar dependencias del nuevo broker

### Configurar Colas Personalizadas
1. Editar `CELERY_CONFIG` en `celery_config.py`
2. Agregar nuevas colas en `task_queues`
3. Configurar routing en `task_routes`
