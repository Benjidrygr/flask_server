"""
Módulo temporal para server_requests
"""

import requests
from video_config import API_URL, SESSION_ID

def get_trip_data(trip_id: str):
    """
    Obtener datos reales del trip desde la API
    """
    try:
        # Construir URL de la API
        api_endpoint = f"{API_URL}/api/fishing-trip/{trip_id}/geolocation-data-v2?_sid_={SESSION_ID}"
        
        # Hacer petición a la API
        response = requests.get(api_endpoint, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extraer información de cámaras del formato de respuesta
            cameras = []
            if 'cameras' in data:
                for camera_data in data['cameras']:
                    camera_id = camera_data.get('cameraId')
                    if camera_id:
                        # Los eventos están en la clave 'signals' de cada cámara
                        signals = camera_data.get('signals', [])
                        
                        cameras.append({
                            'cameraId': camera_id,
                            'signals': signals
                        })
            
            return {
                "trip_id": trip_id,
                "status": "active",
                "cameras": cameras,
                "raw_data": data
            }
        else:
            print(f"Error al obtener datos del trip {trip_id}: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Error al obtener datos del trip {trip_id}: {str(e)}")
        return None
