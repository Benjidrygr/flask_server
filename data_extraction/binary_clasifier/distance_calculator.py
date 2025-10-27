#!/usr/bin/env python3
"""
DistanceCalculator: Calculadora de distancias desde coordenadas marítimas hasta la costa más cercana
"""

import numpy as np
import os
import time
import psutil
from scipy.ndimage import distance_transform_edt
try:
    from .binary_classifier import BinaryClassifier
except ImportError:
    from binary_classifier import BinaryClassifier


class DistanceCalculator(BinaryClassifier):
    """Calculadora de distancias desde coordenadas marítimas hasta la costa más cercana"""
    
    def __init__(self, image_path="global_map/global_planet_map_v1.0.tif", threshold=None, 
                 cache_enabled=True, cache_file="distance_cache_global.npy"):
        super().__init__(image_path=image_path, threshold=threshold)
        self.cache_enabled = cache_enabled
        self.cache_file = cache_file
        self.distance_map = None
        self.scale_factor = 100
        self._compute_distance_map()
    
    def _compute_distance_map(self):
        if self.cache_enabled and os.path.exists(self.cache_file):
            try:
                print(f"Cargando cache de distancias: {self.cache_file}")
                self.distance_map = np.load(self.cache_file)
                print("✅ Cache de distancias cargado")
                return
            except Exception as e:
                print(f"⚠️  Error cargando cache: {e}")
        
        try:
            print("Calculando mapa de distancias...")
            # Crear máscara de agua (donde mask == 0, según convención: 0=agua, 1=tierra)
            water_mask = (self.mask == 0).astype(np.uint8)
            
            # Calcular transformada de distancia euclidiana
            distance_pixels = distance_transform_edt(water_mask)
            
            # Escalar para almacenamiento eficiente
            distance_scaled = (distance_pixels * self.scale_factor).astype(np.uint16)
            
            # Poner 0 en tierra (donde mask == 1)
            distance_scaled[self.mask == 1] = 0
            
            self.distance_map = distance_scaled
            
            if self.cache_enabled:
                try:
                    np.save(self.cache_file, self.distance_map)
                    print(f"✅ Cache de distancias guardado: {self.cache_file}")
                except Exception as e:
                    print(f"⚠️  Error guardando cache: {e}")
                    
        except Exception as e:
            print(f"⚠️  Error calculando mapa de distancias: {e}")
            self.distance_map = np.zeros_like(self.mask, dtype=np.uint16)
            # No lanzar excepción, permitir que funcione con valores por defecto
    
    def distance_to_coast(self, lat, lon):
        if self.distance_map is None:
            print("⚠️  Mapa de distancias no disponible, retornando None")
            return None
        
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            raise ValueError(f"Coordenadas fuera de rango: lat={lat}, lon={lon}")
        
        # Convertir coordenadas a píxeles usando la resolución exacta
        if lon < 0:
            lon_mapped = lon + 360
        else:
            lon_mapped = lon
        
        pixel_x = int(lon_mapped / self.resolution)
        pixel_y = int((90 - lat) / self.resolution)
        pixel_x = max(0, min(self.shape[1] - 1, pixel_x))
        pixel_y = max(0, min(self.shape[0] - 1, pixel_y))
        
        # Si está en tierra (valor 1), distancia es 0
        if self.mask[pixel_y, pixel_x] == 1:
            return 0.0
        
        # Si está en agua, calcular distancia
        distance_pixels_scaled = self.distance_map[pixel_y, pixel_x]
        
        # Si el mapa de distancias tiene valores por defecto (0), retornar None
        if distance_pixels_scaled == 0:
            return None
        
        distance_pixels = distance_pixels_scaled / self.scale_factor
        distance_km = self._pixels_to_km(distance_pixels, lat)
        
        return distance_km
    
    def classify_binary_only(self, lat, lon):
        if lon < 0:
            lon_mapped = lon + 360
        else:
            lon_mapped = lon
        
        lon_mapped = max(0, min(360, lon_mapped))
        pixel_x = int(lon_mapped / self.resolution)
        pixel_y = int((90 - lat) / self.resolution)
        pixel_x = max(0, min(self.shape[1] - 1, pixel_x))
        pixel_y = max(0, min(self.shape[0] - 1, pixel_y))
        
        return "water" if self.mask[pixel_y, pixel_x] == 0 else "land"
    
    def _pixels_to_km(self, pixels, lat):
        lat_correction = np.cos(np.radians(lat))
        km = pixels * self.resolution * 111.0 * lat_correction
        return float(km)
    
    
    def distance_batch(self, coordinates):
        distances = []
        for lat, lon in coordinates:
            try:
                distance = self.distance_to_coast(lat, lon)
                distances.append(distance)
            except Exception:
                distances.append(None)
        return distances
    
    def get_distance_stats(self):
        if self.distance_map is None:
            return {"error": "Mapa de distancias no disponible"}
        
        water_pixels = self.distance_map[self.mask == 0]
        if len(water_pixels) == 0:
            return {"error": "No hay píxeles de agua en el mapa"}
        
        distances_km = water_pixels.astype(np.float32) / self.scale_factor * self.resolution * 111.0
        
        return {
            "min_distance_km": float(np.min(distances_km)),
            "max_distance_km": float(np.max(distances_km)),
            "mean_distance_km": float(np.mean(distances_km)),
            "median_distance_km": float(np.median(distances_km)),
            "std_distance_km": float(np.std(distances_km)),
            "total_water_pixels": int(len(water_pixels)),
            "total_land_pixels": int(np.sum(self.mask == 1)),
            "memory_usage_mb": self._get_memory_usage(),
            "cache_enabled": self.cache_enabled,
            "scale_factor": self.scale_factor,
            "resolution_degrees": self.resolution
        }
    
    def _get_memory_usage(self):
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def print_distance_info(self, lat, lon, location_name=""):
        try:
            distance = self.distance_to_coast(lat, lon)
            classification_binary = self.classify_binary_only(lat, lon)
            classification_original = self.classify(lat, lon)
            
            location_str = f" ({location_name})" if location_name else ""
            classification_emoji = "🏝️" if classification_binary == "land" else "🌊"
            
            print(f"{classification_emoji} {lat:>7.2f}, {lon:>8.2f}{location_str}")
            print(f"   Clasificación (binaria): {classification_binary}")
            print(f"   Clasificación (original): {classification_original}")
            print(f"   Distancia a costa: {distance:.2f} km")
            
        except Exception as e:
            print(f"❌ Error procesando ({lat}, {lon}): {e}")


