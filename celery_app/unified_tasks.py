#!/usr/bin/env python3
"""
Tareas de Celery para el procesamiento unificado de datos
Incluye pipeline completo y pipeline con YOLO
"""

import logging
from celery import current_app as celery_app
from data_extraction.unified_pipeline import UnifiedDataProcessingPipeline, ProcessingConfig

# Configurar logging
logger = logging.getLogger(__name__)

@celery_app.task(bind=True, name='unified_tasks.unified_pipeline_complete_task')
def unified_pipeline_complete_task(self, camera_id, begin_timestamp, end_timestamp, sid, config_dict, stream_name=None, trip_id=None):
    """
    Ejecuta TODO el unified_pipeline:
    1. Descarga videos de AWS Kinesis
    2. Procesa coordenadas desde API
    3. Procesa videos descargados
    4. Integra datos y sube a API
    
    Args:
        camera_id: ID de la cámara (trip_id)
        begin_timestamp: Timestamp de inicio
        end_timestamp: Timestamp de fin
        sid: Session ID
        config_dict: Diccionario de configuración para ProcessingConfig
        
    Returns:
        dict: Resultado del procesamiento completo
    """
    try:
        logger.info("🚀 Iniciando unified_pipeline_complete_task para camera_id: %s", camera_id)
        
        # Crear configuración desde diccionario
        config = ProcessingConfig(**config_dict)
        pipeline = UnifiedDataProcessingPipeline(config)
        
        # 1. Procesar coordenadas desde API
        logger.info("📍 Procesando coordenadas desde API...")
        api_result = pipeline.process_coordinates_from_api(camera_id, begin_timestamp, end_timestamp, sid)
        
        # 2. Descargar videos de AWS (usando trip_id real)
        logger.info("📥 Descargando videos de AWS Kinesis...")
        # Usar trip_id si se proporciona (array con 1 elemento), sino usar camera_id como fallback
        if trip_id and len(trip_id) > 0:
            trip_ids = trip_id  # Ya es un array con 1 elemento
        else:
            trip_ids = [camera_id]  # Fallback al camera_id
        logger.info(f"🎯 Usando trip_ids: {trip_ids}")
        aws_result = pipeline.download_aws_videos(
            stream_name=stream_name or 'default',
            trip_ids=trip_ids,
            session_id=sid,
            begin_timestamp=begin_timestamp,
            end_timestamp=end_timestamp
        )
        
        # 3. Procesar videos descargados
        video_result = None
        if aws_result.success:
            logger.info("🎬 Procesando videos descargados...")
            video_dir = aws_result.aws_download_results['download_directory']
            video_result = pipeline.process_video_directory(video_dir)
        else:
            logger.warning("⚠️ No se pudieron descargar videos de AWS, saltando procesamiento de video")
        
        # 4. Guardar datos procesados para generar coordenadas procesadas
        logger.info("💾 Guardando datos procesados...")
        try:
            pipeline.results = [api_result, aws_result, video_result] if video_result else [api_result, aws_result]
            pipeline.save_results()
            logger.info("✅ Datos procesados guardados exitosamente")
            
            # Sleep de 2 segundos para asegurar que el archivo se escriba completamente
            import time
            logger.info("⏳ Esperando 2 segundos para asegurar escritura completa del archivo...")
            time.sleep(2)
            
        except Exception as e:
            logger.warning(f"⚠️ Error guardando datos procesados: {e}")
        
        # 5. Limpiar videos basándose en coordenadas procesadas
        cleaned_video_result = None
        if video_result and video_result.success and api_result.success:
            logger.info("🧹 Limpiando videos basándose en coordenadas procesadas...")
            try:
                import os
                from data_extraction.video_cleaner import VideoCleaner
                
                # Crear directorio para videos limpios FUERA del directorio del video
                # para evitar que YOLO procese el mismo video dos veces
                base_dir = os.path.dirname(video_dir)  # Subir un nivel
                cleaned_video_dir = os.path.join(base_dir, "cleaned_videos", os.path.basename(video_dir))
                os.makedirs(cleaned_video_dir, exist_ok=True)
                
                # Buscar el archivo de coordenadas procesadas más reciente
                import glob
                pattern = os.path.join('results', 'coordenadas_procesadas_*.json')
                processed_files = glob.glob(pattern)
                
                if processed_files:
                    # Usar el archivo más reciente
                    latest_file = max(processed_files, key=os.path.getctime)
                    logger.info(f"📄 Usando archivo de coordenadas procesadas: {latest_file}")
                    
                    # Limpiar videos usando el archivo JSON
                    cleaner = VideoCleaner(fps=30, frames_per_second=3)
                    cleaned_video_result = cleaner.clean_video_directory(
                        input_dir=video_dir,
                        output_dir=cleaned_video_dir,
                        json_path=latest_file
                    )
                    
                    logger.info(f"✅ Videos limpios guardados en: {cleaned_video_dir}")
                else:
                    logger.warning("⚠️ No se encontraron archivos de coordenadas procesadas")
                    cleaned_video_result = {'success': False, 'error': 'No processed coordinates found'}
                    
            except Exception as e:
                logger.error(f"❌ Error limpiando videos: {e}")
                cleaned_video_result = {'success': False, 'error': str(e)}
        else:
            logger.warning("⚠️ No se puede limpiar videos: falta video_result o api_result")
        
        # 6. Integrar datos
        logger.info("🔗 Integrando datos...")
        # Usar método público si existe, o crear integración básica
        try:
            integrated_data = pipeline._create_integrated_data_from_api_and_video(api_result, video_result)
        except AttributeError:
            # Fallback: crear integración básica
            integrated_data = {
                'api_data': api_result.__dict__,
                'video_data': video_result.__dict__ if video_result else None,
                'cleaned_video_data': cleaned_video_result if cleaned_video_result else None,
                'timestamp': begin_timestamp,
                'camera_id': camera_id
            }
        
        logger.info("✅ unified_pipeline_complete_task completado exitosamente")
        
        return {
            'success': True,
            'api_result': api_result.__dict__,
            'aws_result': aws_result.__dict__,
            'video_result': video_result.__dict__ if video_result else None,
            'cleaned_video_result': cleaned_video_result if cleaned_video_result else None,
            'integrated_data': integrated_data
        }
        
    except Exception as e:
        logger.error("❌ Error en unified_pipeline_complete_task: %s", e)
        raise


@celery_app.task(bind=True, name='unified_tasks.unified_pipeline_yolo_complete_task')
def unified_pipeline_yolo_complete_task(self, camera_id, begin_timestamp, end_timestamp, sid, config_dict, yolo_config_dict, stream_name=None, trip_id=None):
    """
    Ejecuta TODO el unified_pipeline + YOLO processor:
    1. Descarga videos de AWS Kinesis
    2. Procesa coordenadas desde API
    3. Procesa videos descargados (unified)
    4. Procesa videos con YOLO
    5. Integra datos y sube a API
    
    Args:
        camera_id: ID de la cámara (trip_id)
        begin_timestamp: Timestamp de inicio
        end_timestamp: Timestamp de fin
        sid: Session ID
        config_dict: Diccionario de configuración para ProcessingConfig
        yolo_config_dict: Diccionario de configuración para YOLO
        
    Returns:
        dict: Resultado del procesamiento completo con YOLO
    """
    try:
        logger.info("🚀 Iniciando unified_pipeline_yolo_complete_task para camera_id: %s", camera_id)
        
        # 1. Ejecutar unified_pipeline completo
        logger.info("🔄 Ejecutando unified_pipeline completo...")
        unified_result = unified_pipeline_complete_task.delay(camera_id, begin_timestamp, end_timestamp, sid, config_dict, stream_name, trip_id).get()
        
        if not unified_result['success']:
            logger.error("❌ unified_pipeline falló, abortando YOLO processing")
            return unified_result
        
        # 2. Ejecutar YOLO processor en la carpeta de videos limpios
        logger.info("🤖 Ejecutando procesamiento YOLO...")
        aws_result = unified_result.get('aws_result', {})
        aws_download_results = aws_result.get('aws_download_results', {})
        original_video_dir = aws_download_results.get('download_directory')
        
        # Usar la carpeta de videos limpios para YOLO
        import os
        base_dir = os.path.dirname(original_video_dir)
        cleaned_video_dir = os.path.join(base_dir, "cleaned_videos", os.path.basename(original_video_dir))
        
        # Verificar que la carpeta de videos limpios existe
        if not os.path.exists(cleaned_video_dir):
            logger.warning(f"⚠️ Carpeta de videos limpios no existe: {cleaned_video_dir}")
            logger.info("🔄 Usando directorio original de videos para YOLO")
            yolo_video_dir = original_video_dir
        else:
            logger.info(f"✅ Usando carpeta de videos limpios para YOLO: {cleaned_video_dir}")
            yolo_video_dir = cleaned_video_dir
        
        # Configurar directorio de salida de YOLO directamente en results/
        yolo_config_with_output = yolo_config_dict.copy()
        yolo_config_with_output['output_dir'] = 'results'  # Usar results/ directamente
        yolo_config_with_output['visualize'] = False  # Deshabilitar visualización
        
        # Importar y ejecutar YOLO
        from yolo.video_processor_celery import process_videos_celery_compatible
        yolo_result = process_videos_celery_compatible(yolo_video_dir, yolo_config_with_output)
        
        # 2.1. Verificar que los resultados de YOLO estén en results/
        logger.info("📁 Verificando resultados de YOLO en results/...")
        try:
            import os
            import glob
            import time
            
            # Crear directorio results si no existe
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            
            # Buscar archivos de YOLO que ya deberían estar en results/
            yolo_files = []
            
            # Buscar archivos JSON de detecciones trackeadas en results/
            tracked_pattern = os.path.join(results_dir, 'tracked_detections_*.json')
            tracked_files = glob.glob(tracked_pattern)
            
            # Filtrar archivos de prueba (que contienen solo {"test": "..."})
            real_tracked_files = []
            for file_path in tracked_files:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        # Si contiene "test" y es muy pequeño, es un archivo de prueba
                        if '"test"' in content and len(content) < 100:
                            logger.info(f"  ⚠️ Ignorando archivo de prueba: {os.path.basename(file_path)}")
                            continue
                        real_tracked_files.append(file_path)
                except Exception:
                    # Si no se puede leer, incluirlo por si acaso
                    real_tracked_files.append(file_path)
            
            yolo_files.extend(real_tracked_files)
            
            # Buscar videos visualizados en results/
            video_pattern = os.path.join(results_dir, 'visualized_*.mp4')
            video_files = glob.glob(video_pattern)
            yolo_files.extend(video_files)
            
            # Buscar archivos JSON adicionales en results/
            json_pattern = os.path.join(results_dir, 'processing_stats.json')
            json_files = glob.glob(json_pattern)
            yolo_files.extend(json_files)
            
            if yolo_files:
                logger.info(f"📁 {len(yolo_files)} archivos de YOLO encontrados en results/")
                for file_path in yolo_files:
                    logger.info(f"  ✅ {os.path.basename(file_path)}")
                yolo_result['yolo_files_in_results'] = yolo_files
            else:
                logger.warning("⚠️ No se encontraron archivos de YOLO en results/")
                
        except Exception as e:
            logger.error(f"❌ Error verificando archivos de YOLO: {e}")
            yolo_result['verification_error'] = str(e)
        
        # 3. Integrar resultados YOLO con unified_pipeline
        logger.info("🔗 Integrando resultados YOLO...")
        integrated_data = unified_result.get('integrated_data', {})
        
        # Agregar datos YOLO a integrated_data si es necesario
        if yolo_result and 'success' in yolo_result and yolo_result['success']:
            integrated_data['yolo_analysis'] = yolo_result
            logger.info("✅ Datos YOLO integrados exitosamente")
        else:
            logger.warning("⚠️ No se pudieron integrar datos YOLO")
            integrated_data['yolo_analysis'] = None
        
        logger.info("✅ unified_pipeline_yolo_complete_task completado exitosamente")
        
        return {
            'success': True,
            'unified_result': unified_result,
            'yolo_result': yolo_result,
            'final_integrated_data': integrated_data
        }
        
    except Exception as e:
        logger.error("❌ Error en unified_pipeline_yolo_complete_task: %s", e)
        raise


# Funciones auxiliares para facilitar el uso
def execute_unified_pipeline_async(camera_id, begin_timestamp, end_timestamp, sid, config_dict, stream_name=None, trip_id=None):
    """
    Función auxiliar para ejecutar unified_pipeline_complete_task de forma asíncrona
    
    Args:
        camera_id: ID de la cámara
        begin_timestamp: Timestamp de inicio
        end_timestamp: Timestamp de fin
        sid: Session ID
        config_dict: Configuración del pipeline
        stream_name: Nombre del stream de AWS Kinesis
        trip_id: ID del trip (opcional, usa camera_id si no se proporciona)
        
    Returns:
        AsyncResult: Resultado asíncrono de la tarea
    """
    return unified_pipeline_complete_task.delay(
        camera_id=camera_id,
        begin_timestamp=begin_timestamp,
        end_timestamp=end_timestamp,
        sid=sid,
        config_dict=config_dict,
        stream_name=stream_name,
        trip_id=trip_id
    )


def execute_unified_pipeline_yolo_async(camera_id, begin_timestamp, end_timestamp, sid, config_dict, yolo_config_dict, stream_name=None, trip_id=None):
    """
    Función auxiliar para ejecutar unified_pipeline_yolo_complete_task de forma asíncrona
    
    Args:
        camera_id: ID de la cámara
        begin_timestamp: Timestamp de inicio
        end_timestamp: Timestamp de fin
        sid: Session ID
        config_dict: Configuración del pipeline
        yolo_config_dict: Configuración de YOLO
        stream_name: Nombre del stream de AWS Kinesis
        trip_id: ID del trip (opcional, usa camera_id si no se proporciona)
        
    Returns:
        AsyncResult: Resultado asíncrono de la tarea
    """
    return unified_pipeline_yolo_complete_task.delay(
        camera_id=camera_id,
        begin_timestamp=begin_timestamp,
        end_timestamp=end_timestamp,
        sid=sid,
        config_dict=config_dict,
        yolo_config_dict=yolo_config_dict,
        stream_name=stream_name,
        trip_id=trip_id
    )
