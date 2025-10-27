#!/usr/bin/env python3
"""
Script de prueba directa para el flujo completo de YOLO (sin Celery)
"""

import os
import time
import json
import logging
from datetime import datetime

# Asegúrate de que el directorio raíz del proyecto esté en PYTHONPATH
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if project_root not in os.sys.path:
    os.sys.path.insert(0, project_root)

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_direct_yolo_flow():
    print("\n" + "=" * 80)
    print("🚀 FLUJO DIRECTO: Pipeline completo + YOLO")
    print("=" * 80)
    
    # Parámetros de prueba (reales)
    camera_id = "6732cf1db60957bbe7144988"
    begin_timestamp = 1748031360
    end_timestamp = 1748031420
    sid = "68b72e4515dcc305c9219ae6"
    trip_id = ["6834ede0aef38316c518f2f9"]
    
    print(f"📋 Camera ID: {camera_id}")
    print(f"📋 Trip ID: {trip_id}")
    print(f"📋 Stream: d83add525bd7")
    print(f"📋 Timestamps: {begin_timestamp} - {end_timestamp}")
    print()
    
    # Configuración del pipeline
    config_dict = {
        'enable_geographic_classification': True,
        'enable_distance_calculation': True,
        'enable_speed_analysis': True,
        'enable_video_processing': True,
        'verbose': True,
        'api_server': 'us',
        'enable_data_upload': True,
        'keep_videos': True,
        'keep_results': True
    }
    
    # Configuración de YOLO
    yolo_config_dict = {
        'model_path': 'yolo/weigths/20251001-best.pt',
        'confidence_threshold': 0.5,
        'iou_threshold': 0.4,
        'visualize': True,  # Habilitar visualización para generar videos
        'output_dir': 'yolo_results'  # Este será sobrescrito por 'yolo_output'
    }
    
    # Parámetros adicionales para AWS
    stream_name = 'd83add525bd7'
    
    try:
        print("🚀 Ejecutando pipeline completo + YOLO...")
        start_time = time.time()
        
        # Importar y ejecutar directamente la lógica del pipeline
        from data_extraction.unified_pipeline import UnifiedDataProcessingPipeline, ProcessingConfig
        
        # 1. Crear configuración y pipeline
        config = ProcessingConfig(**config_dict)
        pipeline = UnifiedDataProcessingPipeline(config)
        
        # 2. Procesar coordenadas desde API
        print("📍 Paso 1: Procesando coordenadas desde API...")
        api_result = pipeline.process_coordinates_from_api(camera_id, begin_timestamp, end_timestamp, sid)
        print(f"  ✅ API Result Success: {api_result.success}")
        
        # 3. Descargar videos de AWS
        print("📥 Paso 2: Descargando videos de AWS Kinesis...")
        aws_result = pipeline.download_aws_videos(
            stream_name=stream_name,
            trip_ids=trip_id,
            session_id=sid,
            begin_timestamp=begin_timestamp,
            end_timestamp=end_timestamp
        )
        print(f"  ✅ AWS Result Success: {aws_result.success}")
        if aws_result.success:
            download_dir = aws_result.aws_download_results['download_directory']
            print(f"  📂 Directorio de descarga: {download_dir}")
        
        # 4. Procesar videos descargados
        video_result = None
        if aws_result.success:
            print("🎬 Paso 3: Procesando videos descargados...")
            video_dir = aws_result.aws_download_results['download_directory']
            video_result = pipeline.process_video_directory(video_dir)
            print(f"  ✅ Video Result Success: {video_result.success if video_result else False}")
        else:
            print("  ⚠️ No se pudieron descargar videos de AWS, saltando procesamiento de video")
        
        # 5. Guardar datos procesados para generar coordenadas procesadas
        print("💾 Paso 4: Guardando datos procesados...")
        try:
            pipeline.results = [api_result, aws_result, video_result] if video_result else [api_result, aws_result]
            pipeline.save_results()
            print("  ✅ Datos procesados guardados exitosamente")
            
            # Sleep de 2 segundos para asegurar que el archivo se escriba completamente
            print("  ⏳ Esperando 2 segundos para asegurar escritura completa del archivo...")
            time.sleep(2)
            
        except Exception as e:
            print(f"  ⚠️ Error guardando datos procesados: {e}")
        
        # 6. Limpiar videos basándose en coordenadas procesadas
        cleaned_video_result = None
        if video_result and video_result.success and api_result.success:
            print("🧹 Paso 5: Limpiando videos basándose en coordenadas procesadas...")
            try:
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
                    print(f"  📄 Usando archivo de coordenadas procesadas: {latest_file}")
                    
                    # Limpiar videos usando el archivo JSON
                    cleaner = VideoCleaner(fps=30, frames_per_second=3)
                    cleaned_video_result = cleaner.clean_video_directory(
                        input_dir=video_dir,
                        output_dir=cleaned_video_dir,
                        json_path=latest_file
                    )
                    
                    print(f"  ✅ Videos limpios guardados en: {cleaned_video_dir}")
                    if cleaned_video_result.get('success'):
                        frames_removed = cleaned_video_result.get('total_frames_removed', 'N/A')
                        print(f"  📊 Frames removidos: {frames_removed}")
                else:
                    print("  ⚠️ No se encontraron archivos de coordenadas procesadas")
                    cleaned_video_result = {'success': False, 'error': 'No processed coordinates found'}
                    
            except Exception as e:
                print(f"  ❌ Error limpiando videos: {e}")
                cleaned_video_result = {'success': False, 'error': str(e)}
        else:
            print("  ⚠️ No se puede limpiar videos: falta video_result o api_result")
        
        # 7. Ejecutar YOLO processor en la carpeta de videos limpios
        yolo_result = None
        if aws_result.success:
            print("🤖 Paso 6: Ejecutando procesamiento YOLO...")
            try:
                from yolo.video_processor_celery import process_videos_celery_compatible
                
                # Usar la carpeta de videos limpios para YOLO
                base_dir = os.path.dirname(video_dir)
                cleaned_video_dir = os.path.join(base_dir, "cleaned_videos", os.path.basename(video_dir))
                
                # Verificar que la carpeta de videos limpios existe
                if not os.path.exists(cleaned_video_dir):
                    print(f"  ⚠️ Carpeta de videos limpios no existe: {cleaned_video_dir}")
                    print("  🔄 Usando directorio original de videos para YOLO")
                    yolo_video_dir = video_dir
                else:
                    print(f"  ✅ Usando carpeta de videos limpios para YOLO: {cleaned_video_dir}")
                    yolo_video_dir = cleaned_video_dir
                
                # Configurar directorio de salida de YOLO directamente en results/
                yolo_config_with_output = yolo_config_dict.copy()
                yolo_config_with_output['output_dir'] = 'results'  # Usar results/ directamente
                yolo_config_with_output['visualize'] = False  # Deshabilitar visualización
                
                yolo_result = process_videos_celery_compatible(yolo_video_dir, yolo_config_with_output)
                print(f"  ✅ YOLO Result Success: {yolo_result.get('success', False)}")
                
                if yolo_result.get('success'):
                    detections = yolo_result.get('detections', [])
                    tracked_detections = yolo_result.get('tracked_detections', [])
                    print(f"  🔍 Detecciones encontradas: {len(detections)}")
                    print(f"  🎯 Detecciones trackeadas: {len(tracked_detections)}")
                
            except Exception as e:
                print(f"  ❌ Error ejecutando YOLO: {e}")
                yolo_result = {'success': False, 'error': str(e)}
        else:
            print("  ⚠️ No se puede ejecutar YOLO: falta aws_result")
        
        # 8. Verificar que los resultados de YOLO estén en results/
        if yolo_result and yolo_result.get('success'):
            print("📁 Paso 7: Verificando resultados de YOLO en results/...")
            try:
                import glob
                
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
                                print(f"  ⚠️ Ignorando archivo de prueba: {os.path.basename(file_path)}")
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
                    print(f"  📁 {len(yolo_files)} archivos de YOLO encontrados en results/")
                    for file_path in yolo_files:
                        print(f"    ✅ {os.path.basename(file_path)}")
                    yolo_result['yolo_files_in_results'] = yolo_files
                else:
                    print("  ⚠️ No se encontraron archivos de YOLO en results/")
                    
            except Exception as e:
                print(f"  ❌ Error verificando archivos de YOLO: {e}")
                yolo_result['verification_error'] = str(e)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⏱️  Tiempo total de ejecución: {duration:.2f} segundos")
        print("\n" + "=" * 80)
        print("📊 RESULTADOS DEL FLUJO COMPLETO")
        print("=" * 80)
        
        # Verificar resultado general
        overall_success = (api_result.success and aws_result.success and 
                          (video_result.success if video_result else True) and
                          (cleaned_video_result.get('success', True) if cleaned_video_result else True) and
                          (yolo_result.get('success', True) if yolo_result else True))
        
        print(f"✅ Success General: {overall_success}")
        print(f"📍 API Result Success: {api_result.success}")
        print(f"📥 AWS Result Success: {aws_result.success}")
        print(f"🎬 Video Result Success: {video_result.success if video_result else 'N/A'}")
        print(f"🧹 Cleaned Video Success: {cleaned_video_result.get('success', 'N/A') if cleaned_video_result else 'N/A'}")
        print(f"🤖 YOLO Result Success: {yolo_result.get('success', 'N/A') if yolo_result else 'N/A'}")
        
        # Verificar directorio results/
        print(f"\n📂 CONTENIDO DEL DIRECTORIO results/:")
        results_dir = 'results'
        if os.path.exists(results_dir):
            results_files = os.listdir(results_dir)
            if results_files:
                for file_name in sorted(results_files):
                    file_path = os.path.join(results_dir, file_name)
                    if os.path.isfile(file_path):
                        file_size = os.path.getsize(file_path)
                        file_time = os.path.getmtime(file_path)
                        file_time_str = datetime.fromtimestamp(file_time).strftime('%Y-%m-%d %H:%M:%S')
                        print(f"  📄 {file_name} ({file_size} bytes, {file_time_str})")
            else:
                print("  ⚠️ Directorio results/ está vacío")
        else:
            print("  ❌ Directorio results/ no existe")
        
        # Verificar archivos movidos por YOLO
        if yolo_result and yolo_result.get('moved_files'):
            print(f"\n📁 ARCHIVOS DE YOLO MOVIDOS:")
            for file_path in yolo_result['moved_files']:
                print(f"  ✅ {os.path.basename(file_path)}")
        
        print(f"\n🎉 FLUJO COMPLETO EJECUTADO EXITOSAMENTE!")
        
        return {
            'success': overall_success,
            'api_result': api_result,
            'aws_result': aws_result,
            'video_result': video_result,
            'cleaned_video_result': cleaned_video_result,
            'yolo_result': yolo_result
        }
        
    except Exception as e:
        print(f"❌ Error en el flujo completo: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_direct_yolo_flow()
    
    if result and result.get('success'):
        print(f"\n✅ Prueba completada exitosamente")
    else:
        print(f"\n❌ La prueba falló")
