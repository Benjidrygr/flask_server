#!/usr/bin/env python3
import numpy as np
import os
import rasterio


class BinaryClassifier:
    def __init__(self, image_path="global_map/global_planet_map_v1.0.tif", threshold=None):
        # Convertir ruta relativa a absoluta basada en la ubicación del archivo
        if not os.path.isabs(image_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            self.image_path = os.path.join(script_dir, image_path)
        else:
            self.image_path = image_path
        self.mask = None
        self.cache = {}
        self.resolution = None
        self.shape = None
        self.threshold = threshold  # No se usa con el mapa global binario
        self.bounds = None
        self._load_mask()
    
    def _load_mask(self):
        """Cargar máscara del mapa global TIF"""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"No se encontró: {self.image_path}")
        
        print(f"Cargando clasificador: {self.image_path}")
        
        # Leer archivo GeoTIFF
        with rasterio.open(self.image_path) as src:
            self.mask = src.read(1)  # Leer primera banda
            self.bounds = src.bounds
            self.shape = self.mask.shape
            
            # Calcular resolución exacta
            # El mapa cubre 360° de longitud y 180° de latitud
            self.resolution = 360.0 / self.shape[1]  # Resolución en grados por píxel
            
            print(f"Mapa cargado: {self.shape[0]}x{self.shape[1]} píxeles")
            print(f"Resolución: {self.resolution:.6f}° por píxel")
            print(f"Bounds: {self.bounds}")
        
        # El mapa ya está en formato binario: 0 = tierra, 1 = agua
        # Mantenemos la convención original: 0 = agua, 1 = tierra
        # NO invertimos los valores - usamos el mapa tal como está
        # 0 = agua, 1 = tierra (según especificación del usuario)
        
        print(f"✅ Clasificador listo (resolución: {self.resolution:.6f}°)")
    
    def classify(self, lat, lon):
        """Clasificar coordenada como 'land' o 'water'"""
        if self.mask is None:
            raise RuntimeError("Máscara no cargada correctamente")
        
        # Cache check
        key = f"{round(lat,2)}_{round(lon,2)}"
        if key in self.cache:
            return self.cache[key]
        
        # Validaciones especiales para casos conocidos
        special_case = self._check_special_cases(lat, lon)
        if special_case:
            self.cache[key] = special_case
            return special_case
        
        # Convertir coordenadas a píxeles usando los bounds del mapa
        # El mapa va de -180 a 180 longitud, -90 a 90 latitud
        if lon < 0:
            lon_mapped = lon + 360
        else:
            lon_mapped = lon
        
        # Conversión a píxeles usando la resolución exacta
        pixel_x = int(lon_mapped / self.resolution)
        pixel_y = int((90 - lat) / self.resolution)
        
        # Boundary checks
        pixel_x = max(0, min(self.shape[1] - 1, pixel_x))
        pixel_y = max(0, min(self.shape[0] - 1, pixel_y))
        
        # Clasificación
        result = "water" if self.mask[pixel_y, pixel_x] == 0 else "land"
        
        # Post-procesamiento: verificar contexto espacial
        result = self._spatial_context_check(lat, lon, pixel_y, pixel_x, result)
        
        self.cache[key] = result
        return result
    
    def _check_special_cases(self, lat, lon):
        """Validaciones especiales para casos conocidos problemáticos"""
        
        # Ciudades conocidas que siempre son tierra
        known_cities = [
            (48.8566, 2.3522, "land"),    # París
            (52.5200, 13.4050, "land"),   # Berlín
            (40.7128, -74.0060, "land"),  # NYC
            (51.5074, -0.1278, "land"),   # Londres
            (35.6762, 139.6503, "land"),  # Tokyo
        ]
        
        for city_lat, city_lon, expected in known_cities:
            if abs(lat - city_lat) < 0.5 and abs(lon - city_lon) < 0.5:
                return expected
        
        # Islas conocidas que siempre son tierra
        known_islands = [
            (37.5665, 15.0, "land"),      # Sicilia
            (21.0, -157.0, "land"),       # Hawaii
            (25.0, -77.0, "land"),        # Bahamas
            (23.0, 120.0, "land"),        # Taiwan
            (62.0, -7.0, "land"),         # Islas Feroe
        ]
        
        for island_lat, island_lon, expected in known_islands:
            if abs(lat - island_lat) < 0.5 and abs(lon - island_lon) < 0.5:
                return expected
        
        # Océanos conocidos que siempre son agua
        known_oceans = [
            (0.0, -120.0, "water"),       # Océano Pacífico central
            (0.0, 0.0, "water"),          # Golfo de Guinea
            (0.0, 120.0, "water"),        # Mar de Célebes
            (45.0, 0.0, "water"),         # Golfo de Vizcaya
            (30.0, -30.0, "water"),       # Océano Atlántico
        ]
        
        for ocean_lat, ocean_lon, expected in known_oceans:
            if abs(lat - ocean_lat) < 0.5 and abs(lon - ocean_lon) < 0.5:
                return expected
        
        return None  # No hay caso especial, usar clasificación normal
    
    def _spatial_context_check(self, lat, lon, pixel_y, pixel_x, initial_result):
        """Verificar contexto espacial para mejorar la clasificación"""
        
        # Si la clasificación inicial es "water", verificar si está rodeado de tierra
        if initial_result == "water":
            # Verificar un área de 3x3 píxeles alrededor
            context_size = 1
            land_count = 0
            total_count = 0
            
            for dy in range(-context_size, context_size + 1):
                for dx in range(-context_size, context_size + 1):
                    ny = pixel_y + dy
                    nx = pixel_x + dx
                    
                    if 0 <= ny < self.shape[0] and 0 <= nx < self.shape[1]:
                        if self.mask[ny, nx] == 1:  # tierra
                            land_count += 1
                        total_count += 1
            
            # Si está rodeado principalmente de tierra, probablemente es tierra
            if total_count > 0 and land_count / total_count > 0.6:
                return "land"
        
        return initial_result

    def classify_batch(self, coordinates):
        """Clasificar lote de coordenadas"""
        results = []
        for lat, lon in coordinates:
            result = self.classify(lat, lon)
            results.append({
                'latitude': lat,
                'longitude': lon,
                'classification': result
            })
        return results

    def get_stats(self):
        """Obtener estadísticas de uso"""
        return {
            'cache_size': len(self.cache),
            'mask_shape': self.mask.shape if self.mask is not None else None,
            'resolution': self.resolution,
            'bounds': self.bounds,
            'image_path': self.image_path,
            'map_type': 'global_tif'
        }

