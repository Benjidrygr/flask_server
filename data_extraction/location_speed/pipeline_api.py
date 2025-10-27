import sys
import os
import argparse
import requests
from typing import List, Optional
from dataclasses import dataclass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .pipeline_angle_speed import PipelineAngleSpeed
from .speed_calculator import Coordenada as CoordenadaVelocidad


@dataclass
class LocationSignal:
    lat: float
    lon: float
    timestamp: int
    geography: Optional[str] = None
    camera_id: Optional[str] = None
    _id: Optional[str] = None


class CameraLocationAPI:
    def __init__(self, base_url: str = "http://localhost:3000"):
        self.base_url = base_url.rstrip('/')
    
    def get_camera_location_signals(self, camera_id: str, begin_timestamp: int, end_timestamp: int, sid: str) -> List[LocationSignal]:
        url = f"{self.base_url}/api/cameralocationsignal/{camera_id}"
        query_params = {
            'beginTimestamp': begin_timestamp,
            'endTimestamp': end_timestamp,
            '_sid_': sid
        }
        
        try:

            response = requests.get(url, params=query_params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not isinstance(data, list):
                raise ValueError("La respuesta de la API debe ser una lista")
            
            signals = []
            for item in data:
                if not all(key in item for key in ['location', 'locationTimestamp']):
                    raise ValueError("Cada elemento debe tener 'location' y 'locationTimestamp'")
                
                location = item['location']
                if not all(key in location for key in ['latitude', 'longitude']):
                    raise ValueError("El objeto 'location' debe tener 'latitude' y 'longitude'")
                
                signal = LocationSignal(
                    lat=float(location['latitude']),
                    lon=float(location['longitude']),
                    timestamp=int(item['locationTimestamp']),
                    geography=item.get('geography'),
                    camera_id=item.get('cameraId'),
                    _id=item.get('_id')
                )
                signals.append(signal)
            
            signals.sort(key=lambda x: x.timestamp)
            return signals
            
        except requests.RequestException as e:
            raise requests.RequestException(f"Error en la petición a la API: {e}")
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Error al procesar la respuesta de la API: {e}")


def procesar_con_api(camera_id: str, begin_timestamp: int, end_timestamp: int, sid: str, api_url: str = "http://localhost:3000"):
    """Procesa el pipeline usando datos de la API"""
    # Obtener datos de la API
    api_client = CameraLocationAPI(base_url=api_url)
    signals = api_client.get_camera_location_signals(camera_id, begin_timestamp, end_timestamp, sid)
    
    # Convertir a coordenadas de velocidad
    coordenadas_velocidad = [CoordenadaVelocidad(signal.lat, signal.lon, signal.timestamp) for signal in signals]
    
    # Usar el pipeline existente
    pipeline = PipelineAngleSpeed()
    resultado = pipeline.procesar_pipeline(coordenadas_velocidad)
    
    return {
        "camera_id": camera_id,
        "begin_timestamp": begin_timestamp,
        "end_timestamp": end_timestamp,
        "sid": sid,
        "total_coordenadas": len(signals),
        "angulo": resultado['angulo'].angulo,
        "direccion_basica": resultado['angulo'].direccion_basica,
        "direccion_precisa": resultado['angulo'].direccion_precisa,
        "segmentos_dinamicos": resultado.get('segmentos_dinamicos', []),
        "velocidad_original_kmh": resultado['velocidad_original'].velocidad_kmh,
        "velocidad_original_nudos": resultado['velocidad_original'].velocidad_nudos,
        "distancia_metros": resultado['velocidad_original'].distancia_metros,
        "tiempo_segundos": resultado['velocidad_original'].tiempo_segundos,
        "velocidad_ajustada_kmh": resultado['velocidad_ajustada']['velocidad_promediada_filtrada'].velocidad_kmh if resultado['velocidad_ajustada'] else None,
        "velocidad_ajustada_nudos": resultado['velocidad_ajustada']['velocidad_promediada_filtrada'].velocidad_nudos if resultado['velocidad_ajustada'] else None,
        "mejora_kmh": (resultado['velocidad_ajustada']['velocidad_promediada_filtrada'].velocidad_kmh - resultado['velocidad_original'].velocidad_kmh) if resultado['velocidad_ajustada'] else None,
        "coordenadas_filtradas": len(resultado['velocidad_ajustada']['coordenadas_filtradas']) if resultado['velocidad_ajustada'] else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Pipeline de Ángulo y Velocidad API')
    parser.add_argument('camera_id', help='ID de la cámara')
    parser.add_argument('begin_timestamp', type=int, help='Timestamp de inicio')
    parser.add_argument('end_timestamp', type=int, help='Timestamp de fin')
    parser.add_argument('sid', type=str, help='Session ID')
    parser.add_argument('--url', default='https://hardware-server.shellcatch.com/', help='URL base de la API')
    
    args = parser.parse_args()
    
    if args.begin_timestamp >= args.end_timestamp:
        print("Error: begin_timestamp debe ser menor que end_timestamp")
        sys.exit(1)
    
    try:
        resultado = procesar_con_api(args.camera_id, args.begin_timestamp, args.end_timestamp, args.sid, args.url)
        
        print(f"Camera ID: {resultado['camera_id']}")
        print(f"Período: {resultado['begin_timestamp']} - {resultado['end_timestamp']}")
        print(f"Coordenadas totales: {resultado['total_coordenadas']}")
        print(f"Coordenadas filtradas: {resultado['coordenadas_filtradas']}")
        print(f"Ángulo promedio: {resultado['angulo']:.1f}° ({resultado['direccion_precisa']})")
        print(f"Distancia total: {resultado['distancia_metros']:.1f} metros")
        print(f"Tiempo total: {resultado['tiempo_segundos']:.1f} segundos")
        print(f"Velocidad original: {resultado['velocidad_original_kmh']:.2f} km/h ({resultado['velocidad_original_nudos']:.2f} nudos)")
        
        if resultado['velocidad_ajustada_kmh']:
            print(f"Velocidad ajustada: {resultado['velocidad_ajustada_kmh']:.2f} km/h ({resultado['velocidad_ajustada_nudos']:.2f} nudos)")
            print(f"Mejora: {resultado['mejora_kmh']:+.2f} km/h")
        else:
            print("No se pudo calcular velocidad ajustada (insuficientes coordenadas filtradas)")
        
        # Mostrar segmentos dinámicos con ángulos y velocidades
        if 'segmentos_dinamicos' in resultado and resultado['segmentos_dinamicos']:
            print(f"\nSegmentos calculados cada 9 coordenadas:")
            for i, segmento in enumerate(resultado['segmentos_dinamicos']):
                print(f"  Segmento {i+1}: Coordenadas {segmento['indice_inicio']}-{segmento['indice_fin']}")
                print(f"    Ángulo: {segmento['angulo']:.1f}° ({segmento['direccion_precisa']})")
                print(f"    Velocidad: {segmento['velocidad_kmh']:.2f} km/h ({segmento['velocidad_nudos']:.2f} nudos)")
                print(f"    Distancia: {segmento['distancia_metros']:.1f}m, Tiempo: {segmento['tiempo_segundos']:.1f}s")
                print()
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()