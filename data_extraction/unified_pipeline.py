#!/usr/bin/env python3

import sys
import os
import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import requests
from collections import defaultdict
import uuid

def load_env_file(env_path=".env"):
    """Cargar variables de entorno desde archivo .env"""
    env_file = Path(env_path)
    if not env_file.exists():
        return False
    
    with open(env_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Saltar líneas vacías y comentarios
            if not line or line.startswith('#'):
                continue
            
            # Verificar formato KEY=VALUE
            if '=' not in line:
                continue
            
            key, value = line.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remover comillas si existen
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            
            # Establecer variable de entorno
            os.environ[key] = value
    
    return True

# Cargar variables de entorno al inicio
load_env_file()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from binary_clasifier.coordinate_pipeline import CoordinateClassificationPipeline
from binary_clasifier.distance_calculator import DistanceCalculator
from location_speed.pipeline_angle_speed import PipelineAngleSpeed
from location_speed.speed_calculator import Coordenada as CoordenadaVelocidad
from location_speed.pipeline_api import CameraLocationAPI, LocationSignal
from image_proccesing.video_pipeline import VideoProcessingPipeline
from image_proccesing.dark_frames_detector import DarkFrameDetector
from image_proccesing.motion_detector import VideoMotionAnalyzer

# Import para AWS Video Downloader
import sys
import asyncio
import time
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'video_donwload'))
from aws_video_downloader import main as aws_download_main
from video_donwload.video_config import get_video_download_path, API_SERVERS, set_api_server


# Configuración de servidores API disponibles
API_SERVERS = {
    'us': {
        'name': 'Main Server - US',
        'url': 'https://hardware-server.shellcatch.com/',
        's3_region': 'us-east-1'
    },
    'oceania': {
        'name': 'Oceania',
        'url': 'https://oceania.shellcatch.com/',
        's3_region': 'ap-southeast-2'
    },
    'europe': {
        'name': 'Europe',
        'url': 'https://norway.shellcatch.com/',
        's3_region': 'eu-central-1'
    },
    'beta': {
        'name': 'Beta Server',
        'url': 'https://beta.shellcatch.com/',
        's3_region': 'us-east-1'
    }
}

DEFAULT_API_SERVER = 'us'  # Servidor por defecto


class TimestampGrouper:
    """Agrupa resultados por timestamp y aplica votación por mayoría para valores booleanos"""
    
    def __init__(self, grouping_seconds: int = 1):
        self.grouping_seconds = grouping_seconds
        self.groups = defaultdict(list)
    
    def add_frame_result(self, timestamp: float, frame_data: dict):
        """Agregar resultado de frame al grupo correspondiente"""
        group_key = int(timestamp // self.grouping_seconds) * self.grouping_seconds
        self.groups[group_key].append(frame_data)
    
    def get_grouped_results(self) -> List[dict]:
        """Obtener resultados agrupados por timestamp con votación por mayoría"""
        grouped_results = []
        
        for timestamp, frames in sorted(self.groups.items()):
            if not frames:
                continue
            
            # Inicializar resultado del grupo
            group_result = {
                'timestamp': timestamp,
                'frame_count': len(frames),
                'frames': frames
            }
            
            # Procesar valores booleanos con votación por mayoría
            boolean_fields = ['is_dark', 'has_motion']
            for field in boolean_fields:
                if any(field in frame for frame in frames):
                    values = [frame.get(field, False) for frame in frames]
                    true_count = sum(values)
                    false_count = len(values) - true_count
                    
                    # Votación por mayoría
                    group_result[f'{field}_majority'] = true_count > false_count
                    group_result[f'{field}_true_count'] = true_count
                    group_result[f'{field}_false_count'] = false_count
                    group_result[f'{field}_ratio'] = true_count / len(values)
            
            # Procesar valores numéricos con promedio
            numeric_fields = ['motion_score', 'average_score', 'dark_ratio', 'mean_brightness']
            for field in numeric_fields:
                values = [frame.get(field) for frame in frames if frame.get(field) is not None]
                if values:
                    group_result[f'{field}_average'] = sum(values) / len(values)
                    group_result[f'{field}_min'] = min(values)
                    group_result[f'{field}_max'] = max(values)
            
            grouped_results.append(group_result)
        
        return grouped_results
    
    def clear(self):
        """Limpiar grupos"""
        self.groups.clear()


@dataclass
class ProcessingConfig:
    enable_geographic_classification: bool = True
    enable_distance_calculation: bool = True
    enable_speed_analysis: bool = True
    angle_window_size: int = 9
    angle_average_size: int = 3
    enable_video_processing: bool = True
    motion_window: int = 15
    motion_threshold: float = 30.0
    brightness_threshold: int = 30
    dark_pixel_ratio: float = 0.5
    skip_frames_after_motion: int = 15
    api_timeout: int = 30
    api_retry_attempts: int = 3
    geographic_threshold: float = 40.0
    cache_enabled: bool = True
    output_format: str = 'json'
    verbose: bool = False
    group_by_timestamp: bool = False
    timestamp_grouping_seconds: int = 1
    enable_data_upload: bool = False
    api_server: str = DEFAULT_API_SERVER  # Servidor para descarga de videos
    upload_server: Optional[str] = None  # Servidor para upload de datos (independiente)
    upload_api_url: Optional[str] = None  # Se calculará automáticamente basado en upload_server
    camera_location_signal_id: Optional[str] = None
    # CONFIGURACIÓN SIMPLE: Por defecto SÍ guardar todo
    keep_videos: bool = True  # Por defecto conservar videos después del procesamiento
    keep_results: bool = True  # Por defecto guardar archivos de resultados
    
    def __post_init__(self):
        """Inicializar URL de API basada en el servidor seleccionado"""
        if self.upload_api_url is None:
            # Usar upload_server si está especificado, sino usar api_server
            server_to_use = self.upload_server if self.upload_server else self.api_server
            self.upload_api_url = API_SERVERS[server_to_use]['url']
    
    def get_api_url(self) -> str:
        """Obtener URL de la API basada en el servidor seleccionado"""
        return API_SERVERS[self.api_server]['url']
    
    def get_upload_api_url(self) -> str:
        """Obtener URL de la API de upload basada en el servidor de upload"""
        server_to_use = self.upload_server if self.upload_server else self.api_server
        return API_SERVERS[server_to_use]['url']
    
    def get_s3_region(self) -> str:
        """Obtener región S3 basada en el servidor seleccionado"""
        return API_SERVERS[self.api_server]['s3_region']
    
    def get_server_info(self) -> dict:
        """Obtener información completa del servidor seleccionado"""
        return API_SERVERS[self.api_server]


@dataclass
class ProcessingResult:
    timestamp: str
    input_type: str
    input_path: str
    success: bool
    error_message: Optional[str] = None
    geographic_results: Optional[Dict] = None
    speed_results: Optional[Dict] = None
    video_results: Optional[Dict] = None
    image_results: Optional[Dict] = None
    upload_results: Optional[Dict] = None
    aws_download_results: Optional[Dict] = None
    processing_time_seconds: float = 0.0
    total_coordinates_processed: int = 0
    total_frames_processed: int = 0


class UnifiedDataProcessingPipeline:
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.results = []
        self._initialize_components()
    
    def _initialize_components(self):
        if self.config.enable_geographic_classification:
            # Verificar y regenerar cache de distancias si es necesario
            self._ensure_distance_cache()
            
            self.geographic_pipeline = CoordinateClassificationPipeline(
                enable_distance_calculation=self.config.enable_distance_calculation
            )
        
        if self.config.enable_speed_analysis:
            self.speed_pipeline = PipelineAngleSpeed()
        
        if self.config.enable_video_processing:
            self.video_pipeline = VideoProcessingPipeline(
                motion_window=self.config.motion_window,
                motion_threshold=self.config.motion_threshold,
                brightness_threshold=self.config.brightness_threshold,
                dark_pixel_ratio=self.config.dark_pixel_ratio,
                skip_frames_after_motion=self.config.skip_frames_after_motion
            )
        
        self.dark_detector = DarkFrameDetector(
            brightness_threshold=self.config.brightness_threshold,
            dark_pixel_ratio=self.config.dark_pixel_ratio
        )
    
    def _ensure_distance_cache(self):
        """Verificar y regenerar cache de distancias si es necesario (THREAD-SAFE)"""
        if not self.config.enable_distance_calculation:
            return
        
        cache_file = os.path.join(
            os.path.dirname(__file__), 
            'binary_clasifier', 
            'distance_cache_global.npy'
        )
        
        # Usar file locking para evitar regeneración simultánea
        self._ensure_distance_cache_safe(cache_file)
    
    def _ensure_distance_cache_safe(self, cache_file):
        """Implementación thread-safe del cache de distancias con file locking"""
        import fcntl
        
        lock_file = cache_file + '.lock'
        
        # Intentar adquirir lock exclusivo
        try:
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                
                # Verificar si el cache existe y es válido
                cache_valid = False
                if os.path.exists(cache_file):
                    try:
                        import numpy as np
                        cache_data = np.load(cache_file)
                        # Verificar que el cache tiene las dimensiones correctas
                        if cache_data.shape == (4008, 8017):
                            cache_valid = True
                            if self.config.verbose:
                                print(f"✅ Cache de distancias válido encontrado: {cache_file}")
                    except Exception as e:
                        if self.config.verbose:
                            print(f"⚠️  Error verificando cache: {e}")
                
                # Regenerar cache si no es válido
                if not cache_valid:
                    if self.config.verbose:
                        print("🔄 Regenerando cache de distancias...")
                    
                    try:
                        # Importar y crear instancia del DistanceCalculator para regenerar cache
                        from binary_clasifier.distance_calculator import DistanceCalculator
                        
                        # Eliminar cache corrupto si existe
                        if os.path.exists(cache_file):
                            os.remove(cache_file)
                        
                        # Crear nueva instancia (esto regenerará el cache)
                        calculator = DistanceCalculator()
                        
                        if self.config.verbose:
                            print("✅ Cache de distancias regenerado exitosamente")
                            
                    except Exception as e:
                        if self.config.verbose:
                            print(f"⚠️  Error regenerando cache: {e}")
                            
        except IOError:
            # No se pudo adquirir el lock, esperar y reintentar
            if self.config.verbose:
                print("⏳ Esperando que otro worker termine de regenerar el cache...")
            import time
            time.sleep(1)
            return self._ensure_distance_cache_safe(cache_file)
    
    def upload_data_to_api(self, camera_location_signal_id: str, data: Dict, sid: str) -> Dict:
        """
        Subir datos procesados a la API usando PUT
        
        Args:
            camera_location_signal_id: ID del signal de ubicación de la cámara
            data: Datos a subir en formato JSON
            sid: Session ID
            
        Returns:
            Dict con el resultado de la subida
        """
        try:
            # Usar URL de API de upload (puede ser diferente al servidor de descarga)
            upload_api_url = self.config.get_upload_api_url()
            url = f"{upload_api_url.rstrip('/')}/api/camera-signal-location-by-batch"
            query_params = {'_sid_': sid}
            
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            if self.config.verbose:
                print(f"Subiendo datos a: {url}")
                print(f"Datos a subir: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            response = requests.put(
                url, 
                json=data, 
                params=query_params, 
                headers=headers,
                timeout=self.config.api_timeout
            )
            
            response.raise_for_status()
            
            result = {
                'success': True,
                'status_code': response.status_code,
                'response_data': response.json() if response.content else None,
                'upload_url': url,
                'camera_location_signal_id': camera_location_signal_id
            }
            
            if self.config.verbose:
                print(f"Subida exitosa. Status: {response.status_code}")
                if result['response_data']:
                    print(f"Respuesta: {json.dumps(result['response_data'], indent=2, ensure_ascii=False)}")
            
            return result
            
        except requests.exceptions.Timeout:
            return {
                'success': False,
                'error': 'Timeout en la petición',
                'status_code': None,
                'upload_url': url,
                'camera_location_signal_id': camera_location_signal_id
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Error en la petición: {str(e)}',
                'status_code': getattr(e.response, 'status_code', None) if hasattr(e, 'response') else None,
                'upload_url': url,
                'camera_location_signal_id': camera_location_signal_id
            }
        except Exception as e:
            return {
                'success': False,
                'error': f'Error inesperado: {str(e)}',
                'status_code': None,
                'upload_url': url,
                'camera_location_signal_id': camera_location_signal_id
            }
    
    def _apply_timestamp_grouping_to_video_results(self, video_results: dict) -> dict:
        """Aplicar agrupamiento por timestamp a los resultados de video"""
        grouper = TimestampGrouper(self.config.timestamp_grouping_seconds)
        
        # Procesar cada video
        for video_result in video_results.get('video_results', []):
            if 'error' in video_result:
                continue
            
            # Agrupar frames oscuros por timestamp
            if 'dark_analysis' in video_result and 'dark_frames' in video_result['dark_analysis']:
                for frame in video_result['dark_analysis']['dark_frames']:
                    grouper.add_frame_result(frame['timestamp'], {
                        'is_dark': True,
                        'dark_ratio': frame['dark_ratio'],
                        'mean_brightness': frame['mean_brightness'],
                        'frame_number': frame['frame_number']
                    })
            
            # Agrupar frames sin movimiento por timestamp
            if 'no_motion_analysis' in video_result and 'no_motion_frames' in video_result['no_motion_analysis']:
                for frame in video_result['no_motion_analysis']['no_motion_frames']:
                    grouper.add_frame_result(frame['timestamp'], {
                        'has_motion': False,
                        'motion_score': frame['motion_score'],
                        'average_score': frame['average_score'],
                        'frame_number': frame['frame_number']
                    })
            
            # Agrupar frames problemáticos por timestamp
            if 'problematic_frames' in video_result and 'frames' in video_result['problematic_frames']:
                for frame in video_result['problematic_frames']['frames']:
                    grouper.add_frame_result(frame['timestamp'], {
                        'is_dark': True,
                        'has_motion': False,
                        'dark_ratio': frame['dark_ratio'],
                        'motion_score': frame['motion_score'],
                        'frame_number': frame['frame_number']
                    })
            
            # Obtener resultados agrupados
            grouped_results = grouper.get_grouped_results()
            
            # Agregar resultados agrupados al video_result
            video_result['timestamp_grouped_results'] = grouped_results
            
            # Limpiar grouper para el siguiente video
            grouper.clear()
        
        return video_results
    
    def process_coordinates_from_api(self, camera_id: str, begin_timestamp: int, 
                                   end_timestamp: int, sid: str, api_url: str = None) -> ProcessingResult:
        start_time = datetime.now()
        
        print(f"\n📍 === OBTENCIÓN DE COORDENADAS ===")
        print(f"📡 Cámara ID: {camera_id}")
        print(f"⏰ Rango: {begin_timestamp} - {end_timestamp}")
        print(f"🔗 API URL: {api_url or self.config.get_api_url()}")
        
        try:
            # Usar URL de API de la configuración si no se proporciona una específica
            if api_url is None:
                api_url = self.config.get_api_url()
            
            print(f"🔄 Obteniendo señales de la API...")
            api_client = CameraLocationAPI(base_url=api_url)
            signals = api_client.get_camera_location_signals(
                camera_id, begin_timestamp, end_timestamp, sid
            )
            
            if not signals:
                print(f"❌ No se obtuvieron señales de la API")
                return ProcessingResult(
                    timestamp=start_time.isoformat(),
                    input_type='api',
                    input_path=f"{api_url}/api/cameralocationsignal/{camera_id}",
                    success=False,
                    error_message="No se obtuvieron señales de la API"
                )
            
            print(f"✅ Obtenidas {len(signals)} coordenadas de la API")
            
            # Guardar coordenadas originales con sus _id
            self._save_original_coordinates(signals, camera_id)
            
            # Convertir a coordenadas de velocidad
            coordenadas_velocidad = [
                CoordenadaVelocidad(signal.lat, signal.lon, signal.timestamp) 
                for signal in signals
            ]
            
            result = ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='api',
                input_path=f"{api_url}/api/cameralocationsignal/{camera_id}",
                success=True,
                total_coordinates_processed=len(coordenadas_velocidad)
            )
            
            if self.config.enable_geographic_classification:
                print(f"🌍 === PROCESAMIENTO DE COORDENADAS ===")
                print(f"🔄 Clasificando geográficamente {len(signals)} coordenadas...")
                
                geographic_results = []
                for i, signal in enumerate(signals):
                    if i % 10 == 0:  # Log cada 10 coordenadas
                        print(f"📍 Procesando coordenada {i+1}/{len(signals)}")
                    
                    geo_result = self.geographic_pipeline.classify_with_distance(
                        signal.lat, signal.lon
                    )
                    # Agregar información original del signal
                    geo_result['locationTimestamp'] = signal.timestamp
                    geo_result['latitude'] = signal.lat
                    geo_result['longitude'] = signal.lon
                    geo_result['_id'] = signal._id
                    geographic_results.append(geo_result)
                
                print(f"✅ Clasificación geográfica completada: {len(geographic_results)} coordenadas procesadas")
                
                result.geographic_results = {
                    'total_classifications': len(geographic_results),
                    'land_count': sum(1 for r in geographic_results if r['classification'] == 'land'),
                    'water_count': sum(1 for r in geographic_results if r['classification'] == 'water'),
                    'classifications': geographic_results
                }
            
            # Procesar con análisis de velocidad
            if self.config.enable_speed_analysis and len(coordenadas_velocidad) >= 9:
                speed_result = self.speed_pipeline.procesar_pipeline(coordenadas_velocidad)
                result.speed_results = {
                    'angle': speed_result['angulo'].angulo,
                    'direction_basic': speed_result['angulo'].direccion_basica,
                    'direction_precise': speed_result['angulo'].direccion_precisa,
                    'velocity_original_kmh': speed_result['velocidad_original'].velocidad_kmh,
                    'velocity_original_knots': speed_result['velocidad_original'].velocidad_nudos,
                    'velocity_adjusted_kmh': speed_result['velocidad_ajustada']['velocidad_promediada_filtrada'].velocidad_kmh if speed_result['velocidad_ajustada'] else None,
                    'distance_meters': speed_result['velocidad_original'].distancia_metros,
                    'time_seconds': speed_result['velocidad_original'].tiempo_segundos,
                    'segments': speed_result.get('segmentos_dinamicos', [])
                }
            
            # Subir datos a la API si está habilitado
            if self.config.enable_data_upload:
                # Crear array de datos procesados (sin metadatos)
                processed_data_array = []
                
                # Crear mapeo de timestamps para datos de velocidad (si hay segmentos dinámicos)
                speed_data_map = {}
                if result.speed_results and 'segments' in result.speed_results:
                    for segment in result.speed_results['segments']:
                        # Asignar datos de velocidad a cada timestamp en el segmento
                        for i in range(segment.get('indice_inicio', 0), segment.get('indice_fin', 0) + 1):
                            if i < len(signals):
                                timestamp = signals[i].timestamp
                                speed_data_map[timestamp] = {
                                    'angle': segment.get('angulo', result.speed_results['angle']),
                                    'speed': segment.get('velocidad_kmh', result.speed_results['velocity_original_kmh'])
                                }
                
                # Procesar cada coordenada y crear el formato específico sincronizado
                for i, signal in enumerate(signals):
                    data_point = {}
                    
                    # Agregar _id como primer campo si existe en la API
                    if signal._id:
                        data_point['_id'] = signal._id
                    
                    data_point.update({
                        'timestamp': int(signal.timestamp),  # Convertir a entero
                        'darkFrame': False,  # Nuevo nombre del campo
                        'motion': True,   # Nuevo nombre del campo
                        'location': {
                            'latitude': signal.lat,
                            'longitude': signal.lon
                        }
                    })
                    
                    # Agregar datos geográficos si están disponibles
                    if result.geographic_results and i < len(result.geographic_results['classifications']):
                        geo_classification = result.geographic_results['classifications'][i]
                        data_point['geography'] = geo_classification['classification']
                        # Formatear distancia con 2 decimales
                        distance = geo_classification.get('distance_to_coast_km', 0)
                        if distance != 'N/A' and distance is not None:
                            data_point['landDistance'] = f"{float(distance):.2f}"
                        else:
                            data_point['landDistance'] = "0.00"
                    
                    # Agregar datos de velocidad específicos por timestamp
                    if signal.timestamp in speed_data_map:
                        speed_data = speed_data_map[signal.timestamp]
                        data_point['angle'] = round(float(speed_data['angle']), 2)  # 2 decimales
                        data_point['speed'] = round(float(speed_data['speed']), 2)  # 2 decimales
                    elif result.speed_results:
                        # Fallback a datos generales si no hay segmentos específicos
                        data_point['angle'] = round(float(result.speed_results['angle']), 2)  # 2 decimales
                        data_point['speed'] = round(float(result.speed_results['velocity_original_kmh']), 2)  # 2 decimales
                    
                    processed_data_array.append(data_point)
                
                # Usar el camera_location_signal_id si se proporciona, sino usar camera_id
                signal_id = self.config.camera_location_signal_id or camera_id
                upload_result = self.upload_data_to_api(signal_id, processed_data_array, sid)
                upload_result['upload_data'] = processed_data_array  # Guardar los datos que se subieron
                result.upload_results = upload_result
            
            result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            return ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='api',
                input_path=f"{api_url}/api/cameralocationsignal/{camera_id}",
                success=False,
                error_message=str(e)
            )
    
    def process_video_directory(self, input_dir: str, output_dir: str = None) -> ProcessingResult:
        start_time = datetime.now()
        
        try:
            # Procesar videos sin guardar archivos, solo para extraer datos
            video_results = self.video_pipeline.process_directory(input_dir, None)
            
            # Aplicar agrupamiento por timestamp si está habilitado
            if self.config.group_by_timestamp and 'error' not in video_results:
                video_results = self._apply_timestamp_grouping_to_video_results(video_results)
            
            result = ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='video',
                input_path=input_dir,
                success='error' not in video_results,
                error_message=video_results.get('error'),
                video_results=video_results,
                total_frames_processed=sum(
                    vr.get('total_frames', 0) for vr in video_results.get('video_results', [])
                    if 'error' not in vr
                )
            )
            
            result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            return ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='video',
                input_path=input_dir,
                success=False,
                error_message=str(e)
            )
    
    def download_aws_videos(self, stream_name: str, trip_ids: List[str], session_id: str, 
                           max_downloads: int = 3, max_retries: int = 5, timeout: int = 300, 
                           groups: int = 5, sleep_seconds: int = 10, max_attempts: int = 3,
                           begin_timestamp: int = None, end_timestamp: int = None) -> ProcessingResult:
        start_time = datetime.now()
        
        try:
            # SOLUCIÓN MEJORADA: Ejecutar el downloader con rutas absolutas y PYTHONPATH correcto
            import subprocess
            import os
            import sys
            
            # Obtener rutas absolutas
            current_dir = os.path.dirname(os.path.abspath(__file__))
            downloader_dir = os.path.join(current_dir, 'video_donwload')
            downloader_script = os.path.join(downloader_dir, 'aws_video_downloader.py')
            
            # Verificar que el script existe
            if not os.path.exists(downloader_script):
                raise FileNotFoundError(f"Script de descarga no encontrado: {downloader_script}")
            
            # Configurar PYTHONPATH para incluir el directorio padre
            env = os.environ.copy()
            pythonpath = env.get('PYTHONPATH', '')
            if pythonpath:
                env['PYTHONPATH'] = f"{current_dir}:{pythonpath}"
            else:
                env['PYTHONPATH'] = current_dir
            
            # Crear comando con ruta absoluta al script
            cmd = [
                sys.executable,  # Usar el mismo intérprete de Python
                downloader_script,
                '--api-server', self.config.api_server,
                '--stream', stream_name,
                '--session', session_id,
                '--max-downloads', str(max_downloads),
                '--max-retries', str(max_retries),
                '--timeout', str(timeout),
                '--groups', str(groups)
            ]
            
            # Solo agregar --tripid si hay trip IDs válidos
            if trip_ids and len(trip_ids) > 0 and any(trip_id.strip() for trip_id in trip_ids):
                cmd.extend(['--tripid', ','.join(trip_ids)])
                if self.config.verbose:
                    print(f"🎯 Usando trip IDs específicos: {trip_ids}")
            else:
                if self.config.verbose:
                    print("🔄 No hay trip IDs específicos, usando descarga general...")
            
            # Agregar timestamps si se proporcionan
            if begin_timestamp:
                cmd.extend(['--begin-timestamp', str(begin_timestamp)])
            if end_timestamp:
                cmd.extend(['--end-timestamp', str(end_timestamp)])
            
            if self.config.verbose:
                print(f"Ejecutando comando: {' '.join(cmd)}")
                print(f"Directorio de trabajo: {downloader_dir}")
                print(f"PYTHONPATH: {env['PYTHONPATH']}")
            
            # Ejecutar con timeout extendido y variables de entorno
            result = subprocess.run(
                cmd, 
                cwd=downloader_dir, 
                capture_output=True, 
                text=True, 
                timeout=timeout + 120,  # Timeout más generoso
                env=env
            )
            
            if self.config.verbose:
                print(f"Return code: {result.returncode}")
                print(f"STDOUT: {result.stdout}")
                if result.stderr:
                    print(f"STDERR: {result.stderr}")
            
            # Buscar archivos descargados
            base_download_dir = get_video_download_path()
            import glob
            pattern = os.path.join(base_download_dir, '**', '*.mp4')
            downloaded_files = glob.glob(pattern, recursive=True)
            
            # Buscar directorios específicos de trip IDs
            trip_dirs = glob.glob(os.path.join(base_download_dir, '*'))
            trip_dirs = [d for d in trip_dirs if os.path.isdir(d)]
            
            # Usar directorio específico del tripid
            download_dir = base_download_dir  # Inicializar con directorio base
            
            # CORREGIDO: Usar el directorio específico del tripid si hay trip IDs
            if trip_ids and len(trip_ids) > 0 and trip_ids[0].strip():
                trip_id = trip_ids[0].strip()
                specific_dir = os.path.join(base_download_dir, trip_id)
                general_dir = os.path.join(base_download_dir, "general_download")
                
                # Crear directorio específico si no existe
                os.makedirs(specific_dir, exist_ok=True)
                
                # Si hay archivos en general_download, moverlos al directorio específico
                if os.path.exists(general_dir):
                    general_files = glob.glob(os.path.join(general_dir, '*.mp4'))
                    if general_files:
                        print(f"🔄 Moviendo {len(general_files)} archivos de general_download a directorio específico del trip...")
                        for file_path in general_files:
                            filename = os.path.basename(file_path)
                            dest_path = os.path.join(specific_dir, filename)
                            try:
                                import shutil
                                shutil.move(file_path, dest_path)
                                print(f"✅ Movido: {filename}")
                            except Exception as e:
                                print(f"⚠️ Error moviendo {filename}: {e}")
                
                download_dir = specific_dir
                print(f"🔍 CORREGIDO: Usando directorio específico del trip: {download_dir}")
            else:
                # Para descarga general, usar directorio "general_download" SOLO si no hay trip IDs
                general_dir = os.path.join(base_download_dir, "general_download")
                download_dir = general_dir
                print(f"🔍 CORREGIDO: Usando directorio general: {download_dir}")
            
            # CORREGIDO: Si hay trip_dirs disponibles, usar el primero (esto es un fallback)
            # SOLO si no se encontró un directorio específico del trip
            if trip_dirs and len(trip_dirs) > 0 and not (trip_ids and len(trip_ids) > 0 and trip_ids[0].strip()):
                download_dir = trip_dirs[0]
                print(f"🔍 CORREGIDO: Usando primer directorio disponible como fallback: {download_dir}")
            
            if self.config.verbose:
                print(f"📁 Directorio base de descarga: {base_download_dir}")
                print(f"📁 Directorio específico para procesamiento: {download_dir}")
                print(f"📁 Directorios de trips encontrados: {trip_dirs}")
                print(f"📁 Trip IDs proporcionados: {trip_ids}")
                print(f"📁 Archivos descargados encontrados: {len(downloaded_files)}")
                print(f"🔍 DEBUG: download_dir final que se retornará: {download_dir}")
                print(f"🔍 DEBUG: ¿Es el directorio base? {download_dir == base_download_dir}")
            
            # Determinar éxito basado en archivos descargados y return code
            success = len(downloaded_files) > 0 and result.returncode == 0
            
            # Si no hay archivos pero el return code es 0, podría ser que no hay videos en el rango
            if result.returncode != 0 and not success:
                error_msg = f"Error en descarga (return code: {result.returncode}): {result.stderr}"
            elif len(downloaded_files) == 0:
                error_msg = "No se descargaron archivos. Verificar que existan videos en el rango de timestamps especificado."
            else:
                error_msg = None
            
            # DEBUG: Verificar el directorio que se va a retornar
            print(f"🔍 DEBUG FINAL: download_dir que se retornará: {download_dir}")
            print(f"🔍 DEBUG FINAL: ¿Es diferente del base? {download_dir != base_download_dir}")
            
            return ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='aws_download',
                input_path=download_dir,
                success=success,
                error_message=error_msg,
                total_frames_processed=len(downloaded_files),
                aws_download_results={
                    'stream_name': stream_name,
                    'trip_ids': trip_ids,
                    'session_id': session_id,
                    'downloaded_files': downloaded_files,
                    'download_directory': download_dir,
                    'trip_directories': trip_dirs,
                    'total_files_downloaded': len(downloaded_files),
                    'stdout': result.stdout,
                    'stderr': result.stderr,
                    'return_code': result.returncode,
                    'command_executed': ' '.join(cmd),
                    'working_directory': downloader_dir
                }
            )
            
        except subprocess.TimeoutExpired:
            return ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='aws_download',
                input_path=get_video_download_path(),
                success=False,
                error_message=f"Timeout: La descarga excedió el tiempo límite de {timeout + 120} segundos"
            )
        except Exception as e:
            return ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='aws_download',
                input_path=get_video_download_path(),
                success=False,
                error_message=f"Error ejecutando descarga: {str(e)}"
            )
    
    def process_image_directory(self, input_dir: str, output_dir: str = None) -> ProcessingResult:
        start_time = datetime.now()
        
        try:
            # Analizar directorio de imágenes sin guardar archivos, solo para extraer datos
            image_results = self.dark_detector.analyze_directory(
                input_dir, 
                save_dark_images=None  # No guardar imágenes
            )
            
            result = ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='images',
                input_path=input_dir,
                success='error' not in image_results,
                error_message=image_results.get('error'),
                image_results=image_results,
                total_frames_processed=image_results.get('total_images', 0)
            )
            
            result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            return ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='images',
                input_path=input_dir,
                success=False,
                error_message=str(e)
            )
    
    def process_coordinates_list(self, coordinates: List[Tuple[float, float]], 
                               timestamps: Optional[List[int]] = None) -> ProcessingResult:
        start_time = datetime.now()
        
        try:
            
            # Crear coordenadas de velocidad si se proporcionan timestamps
            if timestamps and len(timestamps) == len(coordinates):
                coordenadas_velocidad = [
                    CoordenadaVelocidad(lat, lon, ts) 
                    for (lat, lon), ts in zip(coordinates, timestamps)
                ]
            else:
                # Usar timestamps simulados (1 segundo entre cada punto)
                coordenadas_velocidad = [
                    CoordenadaVelocidad(lat, lon, i) 
                    for i, (lat, lon) in enumerate(coordinates)
                ]
            
            result = ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='coordinates',
                input_path='coordinate_list',
                success=True,
                total_coordinates_processed=len(coordinates)
            )
            
            # Procesar con clasificación geográfica
            if self.config.enable_geographic_classification:
                geographic_results = []
                for lat, lon in coordinates:
                    geo_result = self.geographic_pipeline.classify_with_distance(lat, lon)
                    geographic_results.append(geo_result)
                
                result.geographic_results = {
                    'total_classifications': len(geographic_results),
                    'land_count': sum(1 for r in geographic_results if r['classification'] == 'land'),
                    'water_count': sum(1 for r in geographic_results if r['classification'] == 'water'),
                    'classifications': geographic_results
                }
            
            # Procesar con análisis de velocidad
            if self.config.enable_speed_analysis and len(coordenadas_velocidad) >= 9:
                speed_result = self.speed_pipeline.procesar_pipeline(coordenadas_velocidad)
                result.speed_results = {
                    'angle': speed_result['angulo'].angulo,
                    'direction_basic': speed_result['angulo'].direccion_basica,
                    'direction_precise': speed_result['angulo'].direccion_precisa,
                    'velocity_original_kmh': speed_result['velocidad_original'].velocidad_kmh,
                    'velocity_original_knots': speed_result['velocidad_original'].velocidad_nudos,
                    'velocity_adjusted_kmh': speed_result['velocidad_ajustada']['velocidad_promediada_filtrada'].velocidad_kmh if speed_result['velocidad_ajustada'] else None,
                    'distance_meters': speed_result['velocidad_original'].distancia_metros,
                    'time_seconds': speed_result['velocidad_original'].tiempo_segundos,
                    'segments': speed_result.get('segmentos_dinamicos', [])
                }
            
            result.processing_time_seconds = (datetime.now() - start_time).total_seconds()
            return result
            
        except Exception as e:
            return ProcessingResult(
                timestamp=start_time.isoformat(),
                input_type='coordinates',
                input_path='coordinate_list',
                success=False,
                error_message=str(e)
            )
    
    def _integrate_video_data_with_coordinates(self, upload_data: Dict, video_results: Dict) -> Dict:
        """
        Integrar datos de video (dark_frame, has_motion) con los datos de coordenadas
        Aplica votación por mayoría para múltiples frames por segundo
        
        Args:
            upload_data: Datos de subida con coordenadas
            video_results: Resultados del procesamiento de video
            
        Returns:
            Datos actualizados con información de video sincronizada
        """
        if not video_results or 'error' in video_results:
            return upload_data
        
        # Crear mapeo de timestamps a datos de video con votación por mayoría
        video_data_map = {}
        
        for video_result in video_results.get('video_results', []):
            if 'error' in video_result:
                continue
            
            # Usar datos agrupados por timestamp si están disponibles (votación por mayoría)
            if 'timestamp_grouped_results' in video_result:
                for grouped_result in video_result['timestamp_grouped_results']:
                    timestamp = grouped_result['timestamp']
                    if timestamp not in video_data_map:
                        video_data_map[timestamp] = {}
                    
                    # Usar votación por mayoría para frames oscuros
                    if 'is_dark_majority' in grouped_result:
                        video_data_map[timestamp]['dark_frame'] = grouped_result['is_dark_majority']
                    
                    # Usar votación por mayoría para movimiento
                    if 'has_motion_majority' in grouped_result:
                        video_data_map[timestamp]['has_motion'] = grouped_result['has_motion_majority']  # CORREGIDO: Booleano
            
            # Fallback: mapear frames individuales si no hay datos agrupados
            else:
                # Mapear frames oscuros
                if 'dark_analysis' in video_result and 'dark_frames' in video_result['dark_analysis']:
                    dark_frames = video_result['dark_analysis']['dark_frames']
                    if dark_frames:
                        for frame in dark_frames:
                            timestamp = frame['timestamp']
                            if timestamp not in video_data_map:
                                video_data_map[timestamp] = {}
                            video_data_map[timestamp]['dark_frame'] = True
                
                # Mapear frames sin movimiento
                if 'no_motion_analysis' in video_result and 'no_motion_frames' in video_result['no_motion_analysis']:
                    no_motion_frames = video_result['no_motion_analysis']['no_motion_frames']
                    if no_motion_frames:
                        for frame in no_motion_frames:
                            timestamp = frame['timestamp']
                            if timestamp not in video_data_map:
                                video_data_map[timestamp] = {}
                            video_data_map[timestamp]['has_motion'] = False  # CORREGIDO: Booleano
                
                # Mapear frames problemáticos (oscuros y sin movimiento)
                if 'problematic_frames' in video_result and 'frames' in video_result['problematic_frames']:
                    problematic_frames = video_result['problematic_frames']['frames']
                    if problematic_frames:
                        for frame in problematic_frames:
                            timestamp = frame['timestamp']
                            if timestamp not in video_data_map:
                                video_data_map[timestamp] = {}
                            video_data_map[timestamp]['dark_frame'] = True
                            video_data_map[timestamp]['has_motion'] = False  # CORREGIDO: Booleano
        
        # Actualizar datos de coordenadas con información de video sincronizada
        for data_point in upload_data:  # upload_data ahora es directamente un array
            timestamp = data_point['timestamp']
            if timestamp in video_data_map:
                video_info = video_data_map[timestamp]
                if 'dark_frame' in video_info:
                    data_point['darkFrame'] = video_info['dark_frame']  # Nuevo nombre del campo
                if 'has_motion' in video_info:
                    data_point['motion'] = video_info['has_motion']  # Nuevo nombre del campo
        
        return upload_data
    
    def _create_integrated_data_from_api_and_video(self, api_result, video_result):
        """
        Crear datos integrados desde resultado de API y video
        Sincroniza datos de coordenadas con datos de video por timestamp
        """
        integrated_data = []
        
        # Crear mapeo de timestamps a datos de video
        video_data_map = {}
        
        for video_detail in video_result.video_results.get('video_results', []):
            if 'error' in video_detail:
                continue
            
            # Usar datos agrupados por timestamp si están disponibles
            if 'timestamp_grouped_results' in video_detail:
                for grouped_result in video_detail['timestamp_grouped_results']:
                    timestamp = grouped_result['timestamp']
                    if timestamp not in video_data_map:
                        video_data_map[timestamp] = {}
                    
                    # Usar votación por mayoría para frames oscuros
                    if 'is_dark_majority' in grouped_result:
                        video_data_map[timestamp]['dark_frame'] = grouped_result['is_dark_majority']
                    
                    # Usar votación por mayoría para movimiento
                    if 'has_motion_majority' in grouped_result:
                        video_data_map[timestamp]['has_motion'] = grouped_result['has_motion_majority']
        
        # Procesar cada coordenada y sincronizar con datos de video
        if api_result.geographic_results and 'classifications' in api_result.geographic_results:
            for i, geo_classification in enumerate(api_result.geographic_results['classifications']):
                timestamp = geo_classification.get('locationTimestamp')
                
                data_point = {
                    'timestamp': int(timestamp) if timestamp else 0,  # Convertir a entero
                    'darkFrame': False,  # Nuevo nombre del campo
                    'motion': True,   # Nuevo nombre del campo
                    'location': {
                        'latitude': geo_classification.get('latitude', 0.0),
                        'longitude': geo_classification.get('longitude', 0.0)
                    },
                    'geography': geo_classification.get('classification', 'unknown'),
                    'landDistance': f"{float(geo_classification.get('distance_to_coast_km', 0)):.2f}"  # Nuevo nombre y formato
                }
                
                # Solo agregar _id si existe en la API
                if geo_classification.get('_id'):
                    data_point['_id'] = geo_classification.get('_id')
                
                # Sincronizar con datos de video si están disponibles
                if timestamp in video_data_map:
                    video_info = video_data_map[timestamp]
                    if 'dark_frame' in video_info:
                        data_point['darkFrame'] = video_info['dark_frame']  # Nuevo nombre del campo
                    if 'has_motion' in video_info:
                        data_point['motion'] = video_info['has_motion']  # Nuevo nombre del campo
                
                # Agregar datos de velocidad si están disponibles
                if api_result.speed_results:
                    data_point['angle'] = round(float(api_result.speed_results.get('angle', 0)), 2)  # 2 decimales
                    data_point['speed'] = round(float(api_result.speed_results.get('velocity_original_kmh', 0.0)), 2)  # 2 decimales
                
                integrated_data.append(data_point)
        
        return integrated_data
    
    def process_mixed_data(self, video_dir: Optional[str] = None, 
                          image_dir: Optional[str] = None,
                          coordinates: Optional[List[Tuple[float, float]]] = None,
                          api_config: Optional[Dict] = None) -> List[ProcessingResult]:
        results = []
        video_result = None
        
        # Procesar videos si se proporciona directorio (sin guardar archivos)
        if video_dir:
            video_result = self.process_video_directory(video_dir)
            results.append(video_result)
        
        # Procesar imágenes si se proporciona directorio (sin guardar archivos)
        if image_dir:
            result = self.process_image_directory(image_dir)
            results.append(result)
        
        # Procesar coordenadas si se proporcionan
        if coordinates:
            result = self.process_coordinates_list(coordinates)
            results.append(result)
        
        # Procesar desde API si se proporciona configuración
        if api_config:
            result = self.process_coordinates_from_api(
                api_config['camera_id'],
                api_config['begin_timestamp'],
                api_config['end_timestamp'],
                api_config['sid'],
                api_config['api_url']
            )
            
            # CORREGIDO: Integrar datos de video con coordenadas siempre que sea posible
            if video_result and video_result.success:
                # Si hay upload_results, integrar con ellos
                if result.upload_results and 'upload_data' in result.upload_results:
                    updated_upload_data = self._integrate_video_data_with_coordinates(
                        result.upload_results.get('upload_data', []),  # Ahora es un array
                        video_result.video_results
                    )
                    result.upload_results['upload_data'] = updated_upload_data
                # Si no hay upload_results pero hay datos de API, crear estructura básica integrada
                elif result.input_type == 'api' and result.geographic_results:
                    # Crear datos básicos integrados con video
                    integrated_data = self._create_integrated_data_from_api_and_video(result, video_result)
                    # Agregar los datos integrados al resultado
                    result.integrated_data = integrated_data
            
            results.append(result)
        
        self.results.extend(results)
        return results
    
    def save_results(self, output_path: str = None, output_format: str = 'json'):
        """Guardar solo los datos estructurados en archivo JSON/CSV"""
        
        # POR DEFECTO: No guardar archivos si keep_results es False
        if not self.config.keep_results:
            if self.config.verbose:
                print("ℹ️  Archivos de resultados no guardados (keep_results=False)")
            return
        
        # Si no se especifica output_path, usar results/ por defecto
        if output_path is None:
            output_path = "results"
        
        output_file = Path(output_path)
        
        # Si es un directorio, crear un archivo con timestamp
        if output_file.is_dir() or (not output_file.suffix and not output_file.exists()):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_file / f"coordenadas_procesadas_{timestamp}.{output_format}"
        
        # Crear directorio padre si no existe
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Extraer datos estructurados de los resultados
        structured_data = []
        
        for result in self.results:
            if result.success:
                # CORREGIDO: Priorizar datos integrados si están disponibles
                if hasattr(result, 'integrated_data') and result.integrated_data:
                    structured_data.extend(result.integrated_data)
                # Si hay upload_results, usar esos datos
                elif result.upload_results and 'upload_data' in result.upload_results:
                    upload_data = result.upload_results['upload_data']
                    # upload_data ahora es directamente un array
                    structured_data.extend(upload_data)
                # Si no hay upload_results pero hay datos de API, crear estructura básica
                elif result.input_type == 'api' and result.total_coordinates_processed > 0:
                    # Crear datos básicos para coordenadas de API sin subida
                    basic_data = self._create_basic_data_from_api_result(result)
                    structured_data.extend(basic_data)
        
        # Guardar en el formato especificado
        if output_format.lower() == 'csv':
            self._save_csv(structured_data, output_file)
        else:  # JSON por defecto
            self._save_json(structured_data, output_file)
        
        print(f"Datos estructurados guardados en: {output_file}")
        print(f"Total de registros: {len(structured_data)}")
    
    def _create_basic_data_from_api_result(self, result: ProcessingResult) -> List[Dict]:
        """Crear datos básicos desde resultado de API sin subida"""
        basic_data = []
        
        # Crear datos básicos con información disponible
        if result.geographic_results and 'classifications' in result.geographic_results:
            for i, geo_classification in enumerate(result.geographic_results['classifications']):
                data_point = {
                    'timestamp': int(geo_classification.get('locationTimestamp', 0)) if geo_classification.get('locationTimestamp') else 0,  # Convertir a entero
                    'darkFrame': False,  # Nuevo nombre del campo
                    'motion': True,   # Nuevo nombre del campo
                    'location': {
                        'latitude': geo_classification.get('latitude', 0.0),
                        'longitude': geo_classification.get('longitude', 0.0)
                    },
                    'geography': geo_classification.get('classification', 'unknown'),
                    'landDistance': f"{float(geo_classification.get('distance_to_coast_km', 0)):.2f}"  # Nuevo nombre y formato
                }
                
                # Solo agregar _id si existe en la API
                if geo_classification.get('_id'):
                    data_point['_id'] = geo_classification.get('_id')
                
                # Agregar datos de velocidad si están disponibles
                if result.speed_results:
                    data_point['angle'] = round(float(result.speed_results.get('angle', 0)), 2)  # 2 decimales
                    data_point['speed'] = round(float(result.speed_results.get('velocity_original_kmh', 0.0)), 2)  # 2 decimales
                
                basic_data.append(data_point)
        
        return basic_data
    
    def _save_original_coordinates(self, signals, camera_id: str):
        """Guardar coordenadas originales con sus _id en JSON"""
        try:
            original_data = []
            for signal in signals:
                coord_data = {
                    'timestamp': signal.timestamp,
                    'latitude': signal.lat,
                    'longitude': signal.lon,
                    'camera_id': signal.camera_id,
                    'geography': signal.geography
                }
                # Solo agregar _id si existe en la API
                if signal._id:
                    coord_data['_id'] = signal._id
                
                original_data.append(coord_data)
            
            # Crear directorio de resultados si no existe
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            # Guardar archivo con timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = results_dir / f"coordenadas_originales_{camera_id}_{timestamp}.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(original_data, f, indent=2, ensure_ascii=False)
            
            # Contar cuántas tienen _id de la API
            coords_with_id = sum(1 for coord in original_data if '_id' in coord)
            
            print(f"💾 Coordenadas originales guardadas en: {filename}")
            print(f"📊 Total de coordenadas: {len(original_data)}")
            print(f"🆔 Coordenadas con _id de la API: {coords_with_id}")
            if coords_with_id < len(original_data):
                print(f"⚠️  {len(original_data) - coords_with_id} coordenadas sin _id de la API")
            
        except Exception as e:
            print(f"⚠️ Error guardando coordenadas originales: {e}")
    
    def _cleanup_videos_simple(self, video_directories: List[str]) -> Dict:
        """Limpieza simple de videos - elimina archivos y directorios"""
        if not video_directories:
            return {'cleaned': False, 'reason': 'no_directories'}
        
        import shutil
        import glob
        
        cleaned_dirs = 0
        deleted_files = 0
        space_freed = 0
        errors = []
        
        for video_dir in video_directories:
            if not os.path.exists(video_dir):
                continue
            
            try:
                # Buscar archivos de video
                video_files = []
                for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    pattern = os.path.join(video_dir, f'**/*{ext}')
                    video_files.extend(glob.glob(pattern, recursive=True))
                
                # Eliminar archivos
                for file_path in video_files:
                    try:
                        if os.path.isfile(file_path):
                            space_freed += os.path.getsize(file_path)
                            os.remove(file_path)
                            deleted_files += 1
                    except Exception as e:
                        errors.append(f"Error eliminando {file_path}: {e}")
                
                # Eliminar directorio si está vacío
                try:
                    if os.path.exists(video_dir) and not os.listdir(video_dir):
                        os.rmdir(video_dir)
                        cleaned_dirs += 1
                except Exception as e:
                    errors.append(f"Error eliminando directorio {video_dir}: {e}")
                
                if self.config.verbose:
                    print(f"🧹 Limpiado: {video_dir} ({deleted_files} archivos, {space_freed / (1024*1024):.1f} MB)")
                    
            except Exception as e:
                errors.append(f"Error procesando {video_dir}: {e}")
        
        return {
            'cleaned': True,
            'directories_cleaned': cleaned_dirs,
            'files_deleted': deleted_files,
            'space_freed_mb': space_freed / (1024 * 1024),
            'errors': errors
        }
    
    def _extract_metadata(self) -> Dict:
        """Extraer metadatos importantes de todos los resultados"""
        metadata = {
            "processing_info": {
                "generated_at": datetime.now().isoformat(),
                "total_processing_results": len(self.results),
                "successful_results": sum(1 for r in self.results if r.success),
                "failed_results": sum(1 for r in self.results if not r.success),
                "total_processing_time_seconds": sum(r.processing_time_seconds for r in self.results)
            },
            "api_info": {},
            "video_info": {},
            "coordinates_info": {},
            "upload_info": {}
        }
        
        # Extraer información de API
        api_results = [r for r in self.results if r.input_type == 'api' and r.success]
        if api_results:
            api_result = api_results[0]  # Tomar el primer resultado de API exitoso
            if api_result.upload_results and 'upload_data' in api_result.upload_results:
                upload_data = api_result.upload_results['upload_data']
                if 'metadata' in upload_data:
                    api_metadata = upload_data['metadata']
                    metadata["api_info"] = {
                        "sid": api_metadata.get('sid'),
                        "camera_id": api_metadata.get('camera_id'),
                        "begin_timestamp": api_metadata.get('begin_timestamp'),
                        "end_timestamp": api_metadata.get('end_timestamp'),
                        "upload_enabled": api_metadata.get('upload_enabled'),
                        "upload_url": api_metadata.get('upload_url'),
                        "processing_timestamp": api_metadata.get('processing_timestamp'),
                        "processing_time_seconds": api_metadata.get('processing_time_seconds')
                    }
            
            metadata["coordinates_info"] = {
                "total_coordinates_processed": api_result.total_coordinates_processed,
                "geographic_classifications": api_result.geographic_results.get('total_classifications', 0) if api_result.geographic_results else 0,
                "land_count": api_result.geographic_results.get('land_count', 0) if api_result.geographic_results else 0,
                "water_count": api_result.geographic_results.get('water_count', 0) if api_result.geographic_results else 0,
                "speed_analysis_available": api_result.speed_results is not None,
                "angle": api_result.speed_results.get('angle') if api_result.speed_results else None,
                "velocity_kmh": api_result.speed_results.get('velocity_original_kmh') if api_result.speed_results else None,
                "direction": api_result.speed_results.get('direction_basic') if api_result.speed_results else None
            }
            
            # Información de subida
            if api_result.upload_results:
                metadata["upload_info"] = {
                    "upload_successful": api_result.upload_results.get('success', False),
                    "upload_status_code": api_result.upload_results.get('status_code'),
                    "upload_url": api_result.upload_results.get('upload_url'),
                    "camera_location_signal_id": api_result.upload_results.get('camera_location_signal_id'),
                    "upload_error": api_result.upload_results.get('error') if not api_result.upload_results.get('success', False) else None
                }
        
        # Extraer información de video
        video_results = [r for r in self.results if r.input_type == 'video' and r.success]
        if video_results:
            video_result = video_results[0]  # Tomar el primer resultado de video exitoso
            metadata["video_info"] = {
                "total_frames_processed": video_result.total_frames_processed,
                "video_input_path": video_result.input_path,
                "video_processing_time_seconds": video_result.processing_time_seconds,
                "timestamp_grouping_enabled": self.config.group_by_timestamp,
                "timestamp_grouping_seconds": self.config.timestamp_grouping_seconds
            }
        
        # Extraer información de descarga de AWS
        aws_download_results = [r for r in self.results if r.input_type == 'aws_download' and r.success]
        if aws_download_results:
            aws_result = aws_download_results[0]  # Tomar el primer resultado de descarga exitoso
            if aws_result.aws_download_results:
                metadata["aws_download_info"] = {
                    "stream_name": aws_result.aws_download_results.get('stream_name'),
                    "trip_ids": aws_result.aws_download_results.get('trip_ids'),
                    "total_files_downloaded": aws_result.aws_download_results.get('total_files_downloaded'),
                    "download_directory": aws_result.aws_download_results.get('download_directory'),
                    "download_time_seconds": aws_result.processing_time_seconds,
                    "max_downloads": aws_result.aws_download_results.get('max_downloads'),
                    "max_retries": aws_result.aws_download_results.get('max_retries'),
                    "timeout": aws_result.aws_download_results.get('timeout'),
                    "groups": aws_result.aws_download_results.get('groups')
                }
            
            # Información detallada de videos si está disponible
            if video_result.video_results and 'video_results' in video_result.video_results:
                video_details = video_result.video_results['video_results']
                metadata["video_info"]["total_videos_processed"] = len(video_details)
                metadata["video_info"]["successful_videos"] = sum(1 for v in video_details if 'error' not in v)
                metadata["video_info"]["failed_videos"] = sum(1 for v in video_details if 'error' in v)
                
                # Estadísticas de frames
                total_dark_frames = 0
                total_no_motion_frames = 0
                total_problematic_frames = 0
                
                for video_detail in video_details:
                    if 'error' not in video_detail:
                        if 'dark_analysis' in video_detail:
                            total_dark_frames += video_detail['dark_analysis'].get('dark_frames_count', 0)
                        if 'no_motion_analysis' in video_detail:
                            total_no_motion_frames += video_detail['no_motion_analysis'].get('frames_without_motion', 0)
                        if 'problematic_frames' in video_detail:
                            total_problematic_frames += video_detail['problematic_frames'].get('count', 0)
                
                metadata["video_info"]["frame_statistics"] = {
                    "total_dark_frames": total_dark_frames,
                    "total_no_motion_frames": total_no_motion_frames,
                    "total_problematic_frames": total_problematic_frames
                }
        
        # Información de configuración
        metadata["configuration"] = {
            "enable_geographic_classification": self.config.enable_geographic_classification,
            "enable_distance_calculation": self.config.enable_distance_calculation,
            "enable_speed_analysis": self.config.enable_speed_analysis,
            "enable_video_processing": self.config.enable_video_processing,
            "enable_data_upload": self.config.enable_data_upload,
            "motion_window": self.config.motion_window,
            "motion_threshold": self.config.motion_threshold,
            "brightness_threshold": self.config.brightness_threshold,
            "dark_pixel_ratio": self.config.dark_pixel_ratio,
            "geographic_threshold": self.config.geographic_threshold,
            "angle_window_size": self.config.angle_window_size,
            "angle_average_size": self.config.angle_average_size
        }
        
        return metadata
    
    def _save_json(self, data: List[Dict], output_file: Path):
        """Guardar datos en formato JSON limpio - solo array de datos"""
        # Guardar directamente el array sin metadatos
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def _save_csv(self, data: List[Dict], output_file: Path):
        """Guardar datos en formato CSV"""
        if not data:
            return
        
        # Obtener todas las claves posibles
        all_keys = set()
        for item in data:
            all_keys.update(item.keys())
        
        # Aplanar datos anidados
        flattened_data = []
        for item in data:
            flattened_item = {}
            for key, value in item.items():
                if isinstance(value, dict):
                    # Aplanar objetos anidados como location
                    for nested_key, nested_value in value.items():
                        flattened_item[f"{key}_{nested_key}"] = nested_value
                else:
                    flattened_item[key] = value
            flattened_data.append(flattened_item)
        
        # Escribir CSV
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            if flattened_data:
                writer = csv.DictWriter(f, fieldnames=flattened_data[0].keys())
                writer.writeheader()
                writer.writerows(flattened_data)
        
    
    def print_summary(self):
        """Imprimir resumen de todos los resultados"""
        print("\n" + "="*80)
        print("RESUMEN DEL PIPELINE UNIFICADO")
        print("="*80)
        
        total_results = len(self.results)
        successful_results = sum(1 for r in self.results if r.success)
        failed_results = total_results - successful_results
        
        print(f"Total de procesamientos: {total_results}")
        print(f"Exitosos: {successful_results}")
        print(f"Fallidos: {failed_results}")
        
        # Mostrar información del servidor API
        server_info = self.config.get_server_info()
        print(f"Servidor API: {server_info['name']} ({self.config.api_server})")
        print(f"URL API: {server_info['url']}")
        print(f"Región S3: {server_info['s3_region']}")
        
        if successful_results > 0:
            total_coordinates = sum(r.total_coordinates_processed for r in self.results)
            total_frames = sum(r.total_frames_processed for r in self.results)
            total_time = sum(r.processing_time_seconds for r in self.results)
            
            print(f"Total coordenadas procesadas: {total_coordinates}")
            print(f"Total frames procesados: {total_frames}")
            print(f"Tiempo total de procesamiento: {total_time:.2f} segundos")
            
            # Mostrar información de descargas de AWS
            aws_download_results = [r for r in self.results if r.input_type == 'aws_download']
            if aws_download_results:
                successful_downloads = sum(1 for r in aws_download_results if r.success)
                failed_downloads = len(aws_download_results) - successful_downloads
                print(f"Descargas de AWS: {successful_downloads} exitosas, {failed_downloads} fallidas")
                
                if successful_downloads > 0:
                    total_aws_files = sum(r.aws_download_results.get('total_files_downloaded', 0) for r in aws_download_results if r.success)
                    print(f"Total archivos descargados de AWS: {total_aws_files}")
                
                # Mostrar información de reintentos
                for result in aws_download_results:
                    if result.aws_download_results:
                        attempts_made = result.aws_download_results.get('attempts_made', 1)
                        max_attempts = result.aws_download_results.get('max_attempts', 1)
                        sleep_seconds = result.aws_download_results.get('sleep_seconds', 0)
                        print(f"Intentos de descarga: {attempts_made}/{max_attempts}, Sleep: {sleep_seconds}s")
                
                if failed_downloads > 0:
                    print("Errores de descarga de AWS:")
                    for result in aws_download_results:
                        if not result.success:
                            last_error = result.aws_download_results.get('last_error') if result.aws_download_results else result.error_message
                            print(f"  - {last_error}")
            
            # Mostrar información de subidas
            upload_results = [r for r in self.results if r.upload_results is not None]
            if upload_results:
                successful_uploads = sum(1 for r in upload_results if r.upload_results.get('success', False))
                failed_uploads = len(upload_results) - successful_uploads
                print(f"Subidas a API: {successful_uploads} exitosas, {failed_uploads} fallidas")
                
                if failed_uploads > 0:
                    print("Errores de subida:")
                    for result in upload_results:
                        if not result.upload_results.get('success', False):
                            print(f"  - {result.upload_results.get('error', 'Error desconocido')}")
        
        print("="*80)


def main():
    """
    Pipeline Unificado de Procesamiento de Datos
    
    Integra todas las tecnologías de los scripts:
    - binary_clasifier/: Clasificación geográfica y cálculo de distancias
    - location_speed/: Análisis de velocidad y ángulos de dirección  
    - image_proccesing/: Procesamiento de video e imágenes
    """
    parser = argparse.ArgumentParser(description="Pipeline Unificado de Procesamiento de Datos")
    
    # === ARGUMENTOS DE ENTRADA ===
    parser.add_argument('--video-dir', type=str, help='Directorio con videos para procesar (image_proccesing/)')
    parser.add_argument('--image-dir', type=str, help='Directorio con imágenes para procesar (image_proccesing/)')
    parser.add_argument('--coordinates', type=str, help='Archivo JSON con coordenadas para procesar (location_speed/)')
    parser.add_argument('--coordinate-format', type=str, choices=['json', 'csv', 'txt'], default='json', help='Formato del archivo de coordenadas')
    parser.add_argument('--coordinate-delimiter', type=str, default=',', help='Delimitador para archivos CSV de coordenadas')
    
    # === ARGUMENTOS DE API (location_speed/pipeline_api.py) ===
    parser.add_argument('--api-camera-id', type=str, help='ID de cámara para API (pipeline_api.py)')
    parser.add_argument('--api-begin-timestamp', type=int, help='Timestamp de inicio para API (pipeline_api.py)')
    parser.add_argument('--api-end-timestamp', type=int, help='Timestamp de fin para API (pipeline_api.py)')
    parser.add_argument('--api-sid', type=str, help='Session ID para API (pipeline_api.py)')
    parser.add_argument('--api-server', type=str, choices=['us', 'oceania', 'europe', 'beta'], default=DEFAULT_API_SERVER, 
                      help=f'Servidor API a utilizar: us (Main Server - US), oceania (Oceania), europe (Europe), beta (Beta Server). Default: {DEFAULT_API_SERVER}')
    parser.add_argument('--api-url', type=str, help='URL base personalizada de la API (sobrescribe --api-server)')
    parser.add_argument('--api-timeout', type=int, default=30, help='Timeout para peticiones API en segundos (pipeline_api.py)')
    parser.add_argument('--api-retry-attempts', type=int, default=3, help='Número de reintentos para API (pipeline_api.py)')
    
    # === ARGUMENTOS DE AWS VIDEO DOWNLOADER ===
    parser.add_argument('--enable-aws-download', action='store_true', help='Habilitar descarga de videos desde AWS Kinesis')
    parser.add_argument('--aws-stream-name', type=str, help='Nombre del stream de AWS Kinesis para descarga')
    parser.add_argument('--aws-trip-ids', type=str, help='IDs de los trips separados por comas para descarga de videos')
    parser.add_argument('--aws-session-id', type=str, help='Session ID para AWS Kinesis (si es diferente del API)')
    parser.add_argument('--aws-max-downloads', type=int, default=3, help='Número máximo de descargas simultáneas de AWS')
    parser.add_argument('--aws-max-retries', type=int, default=5, help='Número máximo de reintentos por clip de AWS')
    parser.add_argument('--aws-timeout', type=int, default=300, help='Timeout de lectura en segundos para descargas de AWS')
    parser.add_argument('--aws-groups', type=int, default=5, help='Número de grupos de descargas paralelas de AWS')
    parser.add_argument('--aws-download-only', action='store_true', help='Solo descargar videos, no procesarlos')
    parser.add_argument('--aws-sleep-seconds', type=int, default=10, help='Segundos de espera entre intentos de descarga de AWS')
    parser.add_argument('--aws-max-attempts', type=int, default=3, help='Número máximo de intentos para descarga de AWS')
    
    # === ARGUMENTOS DE SUBIDA DE DATOS ===
    parser.add_argument('--enable-upload', action='store_true', help='Habilitar subida de datos procesados a la API')
    parser.add_argument('--upload-api-url', type=str, help='URL base personalizada para subida de datos a la API (sobrescribe --api-server)')
    parser.add_argument('--camera-location-signal-id', type=str, help='ID del signal de ubicación de la cámara para subida (si es diferente del camera_id)')
    
    # === ARGUMENTOS DE CLASIFICACIÓN GEOGRÁFICA (binary_clasifier/) ===
    parser.add_argument('--geographic-threshold', type=float, default=40.0, help='Umbral para clasificación tierra/agua (binary_classifier.py)')
    parser.add_argument('--disable-distance-calculation', action='store_true', help='Deshabilitar cálculo de distancias a costa (distance_calculator.py)')
    parser.add_argument('--disable-cache', action='store_true', help='Deshabilitar cache de clasificaciones (binary_classifier.py)')
    
    # === ARGUMENTOS DE ANÁLISIS DE VELOCIDAD (location_speed/) ===
    parser.add_argument('--angle-window-size', type=int, default=9, help='Tamaño de ventana para cálculo de ángulos (angle_calculator.py)')
    parser.add_argument('--angle-average-size', type=int, default=3, help='Tamaño de promedio para ángulos (angle_calculator.py)')
    parser.add_argument('--speed-tolerance', type=float, default=15.0, help='Tolerancia en grados para filtrado de velocidad (pipeline_angle_speed.py)')
    
    # === ARGUMENTOS DE PROCESAMIENTO DE VIDEO (image_proccesing/) ===
    parser.add_argument('--motion-window', type=int, default=15, help='Ventana temporal para detección de movimiento (motion_detector.py)')
    parser.add_argument('--motion-threshold', type=float, default=30.0, help='Umbral de movimiento (motion_detector.py)')
    parser.add_argument('--skip-frames-after-motion', type=int, default=15, help='Frames a saltar después de detectar movimiento (video_pipeline.py)')
    parser.add_argument('--video-extensions', type=str, nargs='+', default=['.mp4', '.avi', '.mov', '.mkv'], help='Extensiones de video soportadas (video_pipeline.py)')
    parser.add_argument('--save-motion-frames', action='store_true', help='Guardar frames con movimiento (video_pipeline.py)')
    parser.add_argument('--save-no-motion-frames', action='store_true', help='Guardar frames sin movimiento (video_pipeline.py)')
    
    # === ARGUMENTOS DE DETECCIÓN DE FRAMES OSCUROS (image_proccesing/) ===
    parser.add_argument('--brightness-threshold', type=int, default=15, help='Umbral de brillo para detección de frames oscuros (dark_frames_detector.py)')
    parser.add_argument('--dark-ratio', type=float, default=0.8, help='Proporción mínima de píxeles oscuros (dark_frames_detector.py)')
    parser.add_argument('--image-extensions', type=str, nargs='+', default=['.jpg', '.jpeg', '.png', '.bmp'], help='Extensiones de imagen soportadas (dark_frames_detector.py)')
    parser.add_argument('--save-dark-images', action='store_true', help='Guardar imágenes oscuras detectadas (dark_frames_detector.py)')
    
    # === ARGUMENTOS DE CONFIGURACIÓN GENERAL ===
    parser.add_argument('--output', type=str, required=True, help='Archivo o directorio de salida para resultados (si es directorio, se creará un archivo con timestamp)')
    parser.add_argument('--output-format', type=str, choices=['json', 'csv'], default='json', help='Formato de salida')
    parser.add_argument('--verbose', action='store_true', help='Mostrar información detallada de procesamiento')
    parser.add_argument('--create-output-dirs', action='store_true', help='Crear directorios de salida automáticamente')
    parser.add_argument('--max-concurrent', type=int, default=4, help='Número máximo de procesos concurrentes')
    parser.add_argument('--save-intermediate', action='store_true', help='Guardar resultados intermedios')
    
    # === ARGUMENTOS DE AGRUPAMIENTO POR TIMESTAMP ===
    parser.add_argument('--enable-timestamp-grouping', action='store_true', help='Habilitar agrupamiento por timestamp (por defecto deshabilitado)')
    parser.add_argument('--timestamp-grouping-seconds', type=int, default=1, help='Intervalo en segundos para agrupamiento por timestamp')
    
    # === ARGUMENTOS DE DESHABILITACIÓN ===
    parser.add_argument('--disable-geographic', action='store_true', help='Deshabilitar clasificación geográfica (binary_clasifier/)')
    parser.add_argument('--disable-speed', action='store_true', help='Deshabilitar análisis de velocidad (location_speed/)')
    parser.add_argument('--disable-video', action='store_true', help='Deshabilitar procesamiento de video (image_proccesing/)')
    
    # === ARGUMENTOS SIMPLES DE CONSERVACIÓN ===
    parser.add_argument('--no-keep-videos', action='store_true', help='Eliminar videos después del procesamiento (por defecto: se conservan)')
    parser.add_argument('--no-keep-results', action='store_true', help='No guardar archivos de resultados (por defecto: se guardan)')
    
    args = parser.parse_args()
    
    # Validar que se proporcione al menos un tipo de entrada
    if not any([args.video_dir, args.image_dir, args.coordinates, args.api_camera_id, args.enable_aws_download]):
        parser.error("Debe especificar al menos un tipo de entrada: --video-dir, --image-dir, --coordinates, --api-camera-id, o --enable-aws-download")
    
    # Validar argumentos de AWS download
    if args.enable_aws_download:
        if not args.aws_stream_name:
            parser.error("Para habilitar descarga de AWS, debe proporcionar --aws-stream-name")
        if not args.aws_trip_ids:
            parser.error("Para habilitar descarga de AWS, debe proporcionar --aws-trip-ids")
        if not args.aws_session_id and not args.api_sid:
            parser.error("Para habilitar descarga de AWS, debe proporcionar --aws-session-id o --api-sid")
    
    # Crear configuración
    config = ProcessingConfig(
        enable_geographic_classification=not args.disable_geographic,
        enable_distance_calculation=not args.disable_distance_calculation,
        enable_speed_analysis=not args.disable_speed,
        enable_video_processing=not args.disable_video,
        angle_window_size=args.angle_window_size,
        angle_average_size=args.angle_average_size,
        motion_window=args.motion_window,
        motion_threshold=args.motion_threshold,
        brightness_threshold=args.brightness_threshold,
        dark_pixel_ratio=args.dark_ratio,
        skip_frames_after_motion=args.skip_frames_after_motion,
        api_timeout=args.api_timeout,
        api_retry_attempts=args.api_retry_attempts,
        geographic_threshold=args.geographic_threshold,
        cache_enabled=not args.disable_cache,
        output_format=args.output_format,
        verbose=args.verbose,
        group_by_timestamp=args.enable_timestamp_grouping,
        timestamp_grouping_seconds=args.timestamp_grouping_seconds,
        enable_data_upload=args.enable_upload,
        api_server=args.api_server,
        upload_api_url=args.upload_api_url,  # Si se proporciona, sobrescribe la del servidor
        camera_location_signal_id=args.camera_location_signal_id,
        # CONFIGURACIÓN SIMPLE: Por defecto SÍ guardar todo
        keep_videos=not args.no_keep_videos,  # Por defecto True (conservar videos)
        keep_results=not args.no_keep_results  # Por defecto True (guardar resultados)
    )
    
    # Crear pipeline
    pipeline = UnifiedDataProcessingPipeline(config)
    
    # Procesar según los argumentos proporcionados
    results = []
    
    # Procesar videos (sin guardar archivos)
    if args.video_dir:
        result = pipeline.process_video_directory(args.video_dir)
        results.append(result)
    
    # Procesar imágenes (sin guardar archivos)
    if args.image_dir:
        result = pipeline.process_image_directory(args.image_dir)
        results.append(result)
    
    # Procesar coordenadas desde archivo
    if args.coordinates:
        with open(args.coordinates, 'r') as f:
            coord_data = json.load(f)
        
        coordinates = [(item['lat'], item['lon']) for item in coord_data.get('coordinates', [])]
        timestamps = [item.get('timestamp') for item in coord_data.get('coordinates', [])]
        
        result = pipeline.process_coordinates_list(coordinates, timestamps)
        results.append(result)
    
    # Procesar desde API
    if args.api_camera_id:
        if not all([args.api_begin_timestamp, args.api_end_timestamp, args.api_sid]):
            parser.error("Para usar API, debe proporcionar --api-begin-timestamp, --api-end-timestamp, y --api-sid")
        
        result = pipeline.process_coordinates_from_api(
            args.api_camera_id,
            args.api_begin_timestamp,
            args.api_end_timestamp,
            args.api_sid,
            args.api_url  # Si se proporciona, sobrescribe la del servidor
        )
        results.append(result)
    
    # Descargar videos desde AWS Kinesis (después de obtener coordenadas si es necesario)
    aws_download_result = None
    if args.enable_aws_download:
        # Si se procesaron coordenadas desde API, esperar antes de descargar videos
        api_processed = any(r.input_type == 'api' and r.success for r in results)
        if api_processed and args.verbose:
            print(f"\n⏳ Esperando {args.aws_sleep_seconds} segundos después del procesamiento de coordenadas antes de descargar videos...")
            time.sleep(args.aws_sleep_seconds)
        
        # Usar session_id de AWS o API según esté disponible
        session_id = args.aws_session_id or args.api_sid
        
        print(f"\n🎬 === DESCARGA DE VIDEOS AWS KINESIS ===")
        print(f"📡 Stream: {args.aws_stream_name}")
        print(f"🆔 Trip IDs: {args.aws_trip_ids}")
        print(f"⏰ Rango de timestamps: {args.api_begin_timestamp} - {args.api_end_timestamp}")
        print(f"🔄 Iniciando descarga...")
        
        aws_download_result = pipeline.download_aws_videos(
            stream_name=args.aws_stream_name,
            trip_ids=args.aws_trip_ids.split(','),
            session_id=session_id,
            max_downloads=args.aws_max_downloads,
            max_retries=args.aws_max_retries,
            timeout=args.aws_timeout,
            groups=args.aws_groups,
            sleep_seconds=args.aws_sleep_seconds,
            max_attempts=args.aws_max_attempts,
            begin_timestamp=args.api_begin_timestamp,
            end_timestamp=args.api_end_timestamp
        )
        results.append(aws_download_result)
        
        if aws_download_result.success:
            print(f"✅ Descarga de videos completada exitosamente")
            if hasattr(aws_download_result, 'aws_download_results'):
                download_info = aws_download_result.aws_download_results
                print(f"📁 Directorio de descarga: {download_info.get('download_directory', 'N/A')}")
                print(f"📊 Videos descargados: {download_info.get('total_downloaded', 0)}")
        else:
            print(f"❌ Error en la descarga de videos: {aws_download_result.error_message}")
        
        # Si no es solo descarga, procesar los videos descargados
        if not args.aws_download_only and aws_download_result.success:
            # Obtener el directorio de descarga
            download_dir = aws_download_result.aws_download_results['download_directory']
            
            print(f"\n🎥 === PROCESAMIENTO DE VIDEO ===")
            print(f"📁 Directorio: {download_dir}")
            print(f"🔄 Iniciando análisis de frames oscuros y movimiento...")
            
            # Procesar los videos descargados
            video_result = pipeline.process_video_directory(download_dir)
            
            if video_result.success:
                print(f"✅ Procesamiento de video completado exitosamente")
                if hasattr(video_result, 'video_results'):
                    video_info = video_result.video_results
                    print(f"📊 Videos procesados: {video_info.get('total_videos_processed', 0)}")
                    print(f"🎞️ Frames analizados: {video_info.get('total_frames_processed', 0)}")
            else:
                print(f"❌ Error en el procesamiento de video: {video_result.error_message}")
            results.append(video_result)
            
            # LIMPIEZA AUTOMÁTICA: Eliminar videos después del procesamiento
            if not config.keep_videos and video_result.success:
                if args.verbose:
                    print(f"\n🧹 Eliminando videos después del procesamiento...")
                
                cleanup_result = pipeline._cleanup_videos_simple([download_dir])
                
                if args.verbose:
                    if cleanup_result['cleaned']:
                        print(f"✅ Videos eliminados: {cleanup_result['files_deleted']} archivos, {cleanup_result['space_freed_mb']:.1f} MB liberados")
                        if cleanup_result['errors']:
                            for error in cleanup_result['errors']:
                                print(f"⚠️  {error}")
                    else:
                        print(f"ℹ️  No se eliminaron videos: {cleanup_result.get('reason', 'desconocido')}")
    
    # Guardar resultados automáticamente en results/
    pipeline.results = results
    
    print(f"\n🔗 === UNIFICACIÓN DE DATOS ===")
    print(f"🔄 Integrando datos de coordenadas y video...")
    
    # Guardar automáticamente en results/ si no se especifica output
    if args.output == "results" or args.output.endswith("/results"):
        pipeline.save_results()  # Usar results/ por defecto
    else:
        pipeline.save_results(args.output, args.output_format)
    
    print(f"✅ Datos unificados guardados exitosamente")
    
    pipeline.print_summary()


if __name__ == "__main__":
    main()
