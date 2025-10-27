"""
Módulo para limpiar videos basándose en datos JSON de análisis
Elimina frames según condiciones específicas basadas en análisis de movimiento, brillo y geografía
"""

import os
import json
import cv2
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoCleaner:
    """
    Clase para limpiar videos eliminando frames basándose en análisis JSON
    """
    
    def __init__(self, fps: int = 30, frames_per_second: int = 3):
        """
        Inicializar el limpiador de videos
        
        Args:
            fps: Frames por segundo del video
            frames_per_second: Número de frames a eliminar por segundo (por defecto 3)
        """
        self.fps = fps
        self.frames_per_second = frames_per_second
        self.logger = logger
    
    def load_json_data(self, json_path: str) -> List[Dict]:
        """
        Cargar datos JSON de análisis
        
        Args:
            json_path: Ruta al archivo JSON
            
        Returns:
            Lista de diccionarios con datos de análisis
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.logger.info(f"✅ Cargados {len(data)} registros desde {json_path}")
            return data
        except Exception as e:
            self.logger.error(f"❌ Error cargando JSON: {str(e)}")
            raise
    
    def should_remove_frames(self, data_point: Dict) -> bool:
        """
        Determinar si los frames deben ser eliminados basándose en las condiciones
        
        Condiciones para eliminar:
        1. has_motion = false AND dark_frame = true
        2. geography != "water"
        
        Args:
            data_point: Punto de datos del JSON
            
        Returns:
            True si los frames deben ser eliminados
        """
        # Condición 1: Sin movimiento Y frame oscuro
        condition1 = (
            not data_point.get('has_motion', True) and 
            data_point.get('dark_frame', False)
        )
        
        # Condición 2: Geografía no es agua
        condition2 = data_point.get('geography', '') != 'water'
        
        should_remove = condition1 or condition2
        
        if should_remove:
            self.logger.debug(f"🗑️ Frame marcado para eliminación: {data_point.get('_id', 'unknown')}")
            self.logger.debug(f"   - Condición 1 (sin movimiento + oscuro): {condition1}")
            self.logger.debug(f"   - Condición 2 (no agua): {condition2}")
        
        return should_remove
    
    def get_frames_to_remove(self, json_data: List[Dict], video_start_timestamp: Optional[float] = None, total_video_frames: Optional[int] = None) -> List[Tuple[int, int]]:
        """
        Obtener rangos de frames a eliminar basándose en los datos JSON
        Cada timestamp mapea exactamente a 3 frames consecutivos
        
        Args:
            json_data: Lista de datos de análisis
            video_start_timestamp: Timestamp de inicio del video (opcional)
            total_video_frames: Total de frames del video (opcional)
            
        Returns:
            Lista de tuplas (inicio_frame, fin_frame) a eliminar
        """
        frames_to_remove = []
        
        # Si no se proporciona timestamp de inicio, usar el mínimo del JSON
        if video_start_timestamp is None:
            timestamps = [item.get('timestamp', 0) for item in json_data if 'timestamp' in item]
            if timestamps:
                video_start_timestamp = min(timestamps)
            else:
                self.logger.warning("⚠️ No se encontraron timestamps en los datos JSON")
                return []
        
        self.logger.info(f"🎬 Timestamp de inicio del video: {video_start_timestamp}")
        
        # Calcular frames residuales si se proporciona el total
        residual_frames = 0
        if total_video_frames is not None:
            # Calcular cuántos frames corresponden a los timestamps
            timestamps_count = len(json_data)
            frames_for_timestamps = timestamps_count * self.frames_per_second
            
            # Calcular frames residuales
            residual_frames = total_video_frames - frames_for_timestamps
            
            if residual_frames > 0:
                self.logger.info(f"📊 Frames residuales detectados: {residual_frames}")
                self.logger.info(f"   - Total frames del video: {total_video_frames}")
                self.logger.info(f"   - Frames para timestamps: {frames_for_timestamps}")
                self.logger.info(f"   - Frames residuales: {residual_frames} (NO se eliminan automáticamente)")
                self.logger.info(f"   ℹ️ Solo se eliminan frames basados en condiciones de timestamps")
            else:
                self.logger.info(f"✅ No hay frames residuales (video es múltiplo exacto de 3)")
        
        # Mapear cada timestamp a exactamente 3 frames consecutivos
        for i, data_point in enumerate(json_data):
            if self.should_remove_frames(data_point):
                # Calcular posición del frame basada en el índice del timestamp
                frame_start = i * self.frames_per_second
                frame_end = frame_start + self.frames_per_second
                
                # Solo agregar si no excede el total de frames (si se proporciona)
                if total_video_frames is None or frame_start < total_video_frames:
                    frames_to_remove.append((frame_start, frame_end))
                    
                    timestamp = data_point.get('timestamp', 0)
                    relative_time = timestamp - video_start_timestamp
                    self.logger.debug(f"📝 Timestamp {i}: frames {frame_start}-{frame_end} marcados para eliminación (tiempo: {relative_time:.2f}s)")
                else:
                    self.logger.debug(f"⚠️ Frame {frame_start} fuera del rango del video")
        
        # Consolidar rangos superpuestos
        frames_to_remove = self._consolidate_ranges(frames_to_remove)
        
        self.logger.info(f"🎯 Total de rangos de frames a eliminar: {len(frames_to_remove)}")
        return frames_to_remove
    
    def _consolidate_ranges(self, ranges: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Consolidar rangos superpuestos para optimizar el procesamiento
        
        Args:
            ranges: Lista de rangos (inicio, fin)
            
        Returns:
            Lista consolidada de rangos
        """
        if not ranges:
            return []
        
        # Ordenar por inicio
        ranges.sort(key=lambda x: x[0])
        consolidated = [ranges[0]]
        
        for current in ranges[1:]:
            last = consolidated[-1]
            
            # Si hay superposición, extender el rango
            if current[0] <= last[1]:
                consolidated[-1] = (last[0], max(last[1], current[1]))
            else:
                consolidated.append(current)
        
        return consolidated
    
    def clean_video(self, input_video_path: str, output_video_path: str, 
                   json_data: List[Dict], progress_callback: Optional[callable] = None) -> Dict:
        """
        Limpiar video eliminando frames según las condiciones especificadas
        
        Args:
            input_video_path: Ruta al video de entrada
            output_video_path: Ruta al video de salida
            json_data: Datos JSON de análisis
            progress_callback: Función de callback para progreso
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        try:
            self.logger.info(f"🎬 Iniciando limpieza de video: {input_video_path}")
            
            # Abrir video para obtener propiedades
            cap = cv2.VideoCapture(input_video_path)
            if not cap.isOpened():
                raise Exception(f"No se pudo abrir el video: {input_video_path}")
            
            # Obtener propiedades del video
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            self.logger.info(f"📊 Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
            
            # Obtener frames a eliminar
            frames_to_remove = self.get_frames_to_remove(json_data, total_video_frames=total_frames)
            
            if not frames_to_remove:
                self.logger.info("✅ No hay frames para eliminar, copiando video original")
                import shutil
                shutil.copy2(input_video_path, output_video_path)
                return {
                    'success': True,
                    'frames_removed': 0,
                    'total_frames_processed': 0,
                    'output_path': output_video_path,
                    'message': 'No frames to remove'
                }
            
            # Configurar writer de salida con codec H264 (más compatible)
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            frames_removed = 0
            current_frame = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Verificar si este frame debe ser eliminado
                should_remove = False
                for start_frame, end_frame in frames_to_remove:
                    if start_frame <= current_frame < end_frame:
                        should_remove = True
                        break
                
                if not should_remove:
                    out.write(frame)
                else:
                    frames_removed += 1
                
                current_frame += 1
                
                # Callback de progreso
                if progress_callback and current_frame % 100 == 0:
                    progress = (current_frame / total_frames) * 100
                    progress_callback(progress, current_frame, total_frames)
            
            # Liberar recursos
            cap.release()
            out.release()
            
            result = {
                'success': True,
                'frames_removed': frames_removed,
                'total_frames_processed': current_frame,
                'output_path': output_video_path,
                'removal_ranges': len(frames_to_remove),
                'fps': fps,
                'resolution': f"{width}x{height}"
            }
            
            self.logger.info(f"✅ Video limpiado exitosamente: {frames_removed} frames eliminados")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error limpiando video: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'frames_removed': 0,
                'total_frames_processed': 0
            }
    
    def clean_video_directory(self, input_dir: str, output_dir: str, 
                           json_path: str, progress_callback: Optional[callable] = None) -> Dict:
        """
        Limpiar todos los videos en un directorio basándose en datos JSON
        
        Args:
            input_dir: Directorio con videos de entrada
            output_dir: Directorio para videos de salida
            json_path: Ruta al archivo JSON de análisis
            progress_callback: Función de callback para progreso
            
        Returns:
            Diccionario con resultados del procesamiento
        """
        try:
            # Crear directorio de salida
            os.makedirs(output_dir, exist_ok=True)
            
            # Cargar datos JSON
            json_data = self.load_json_data(json_path)
            
            # Buscar videos en el directorio
            video_files = []
            for file in os.listdir(input_dir):
                if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_files.append(file)
            
            if not video_files:
                return {
                    'success': False,
                    'error': 'No se encontraron archivos de video',
                    'processed_videos': 0
                }
            
            self.logger.info(f"🎬 Encontrados {len(video_files)} videos para procesar")
            
            results = []
            total_frames_removed = 0
            
            for i, video_file in enumerate(video_files):
                input_path = os.path.join(input_dir, video_file)
                # FORZAR CONSERVACIÓN DEL NOMBRE ORIGINAL - NO AGREGAR PREFIJOS
                output_path = os.path.join(output_dir, video_file)  # Conservar nombre original
                
                self.logger.info(f"🔄 Procesando video {i+1}/{len(video_files)}: {video_file}")
                self.logger.info(f"📁 Ruta de entrada: {input_path}")
                self.logger.info(f"📁 Ruta de salida: {output_path}")
                
                # Limpiar video individual
                result = self.clean_video(input_path, output_path, json_data, progress_callback)
                results.append({
                    'input_file': video_file,
                    'output_file': video_file,  # Conservar nombre original
                    'result': result
                })
                
                if result['success']:
                    total_frames_removed += result.get('frames_removed', 0)
            
            return {
                'success': True,
                'processed_videos': len(video_files),
                'total_frames_removed': total_frames_removed,
                'results': results,
                'output_directory': output_dir
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error procesando directorio: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processed_videos': 0
            }

def clean_video_with_json(video_path: str, json_path: str, output_path: str, 
                         fps: int = 30, frames_per_second: int = 3) -> Dict:
    """
    Función de conveniencia para limpiar un video con datos JSON
    
    Args:
        video_path: Ruta al video de entrada
        json_path: Ruta al archivo JSON de análisis
        output_path: Ruta al video de salida
        fps: Frames por segundo del video
        frames_per_second: Frames a eliminar por segundo
        
    Returns:
        Diccionario con resultados
    """
    cleaner = VideoCleaner(fps=fps, frames_per_second=frames_per_second)
    
    # Cargar datos JSON
    json_data = cleaner.load_json_data(json_path)
    
    # Limpiar video
    return cleaner.clean_video(video_path, output_path, json_data)

if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    
    if len(sys.argv) < 4:
        print("Uso: python video_cleaner.py <video_path> <json_path> <output_path>")
        sys.exit(1)
    
    video_path = sys.argv[1]
    json_path = sys.argv[2]
    output_path = sys.argv[3]
    
    result = clean_video_with_json(video_path, json_path, output_path)
    print(f"Resultado: {result}")
