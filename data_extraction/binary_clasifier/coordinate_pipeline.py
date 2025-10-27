#!/usr/bin/env python3
"""
Pipeline de clasificación de coordenadas usando mapa global
- Principal: BinaryClassifier (mapa global TIF)
- Calculadora de distancias integrada
"""

try:
    from .binary_classifier import BinaryClassifier
    from .distance_calculator import DistanceCalculator
except ImportError:
    from binary_classifier import BinaryClassifier
    from distance_calculator import DistanceCalculator
import time

class CoordinateClassificationPipeline:
    def __init__(self, enable_distance_calculation=True):
        """
        Inicializar pipeline con clasificador global y calculadora de distancias
        
        Args:
            enable_distance_calculation (bool): Habilitar calculadora de distancias
        """
        print("Inicializando pipeline de clasificación...")
        
        # Clasificador principal (mapa global TIF)
        print("Cargando clasificador global...")
        self.binary_classifier = BinaryClassifier()
        
        # Calculadora de distancias (opcional)
        self.distance_calculator = None
        if enable_distance_calculation:
            print("Cargando calculadora de distancias...")
            self.distance_calculator = DistanceCalculator()
        
        # Estadísticas
        self.stats = {
            'global_classifications': 0,
            'distance_calculations': 0,
            'total_processed': 0,
            'errors': 0
        }
        
        print("✅ Pipeline listo!")
    
    def classify(self, lat, lon):
        """
        Clasificar coordenada usando mapa global
        
        Args:
            lat (float): Latitud
            lon (float): Longitud
            
        Returns:
            dict: Resultado con clasificación y metadatos
        """
        self.stats['total_processed'] += 1
        
        try:
            # Usar clasificador global
            start_time = time.time()
            global_result = self.binary_classifier.classify(lat, lon)
            processing_time = time.time() - start_time
            
            self.stats['global_classifications'] += 1
            
            return {
                'latitude': lat,
                'longitude': lon,
                'classification': global_result,
                'method': 'global_map',
                'processing_time': processing_time,
                'confidence': 'high'
            }
            
        except Exception as e:
            self.stats['errors'] += 1
            return {
                'latitude': lat,
                'longitude': lon,
                'classification': 'error',
                'method': 'failed',
                'processing_time': 0,
                'confidence': 'none',
                'error': str(e)
            }
    
    def classify_batch(self, coordinates):
        """Clasificar lote de coordenadas"""
        results = []
        for lat, lon in coordinates:
            result = self.classify(lat, lon)
            results.append(result)
        return results
    
    def distance_to_coast(self, lat, lon):
        """
        Calcular distancia desde coordenadas hasta la costa más cercana
        
        Args:
            lat (float): Latitud
            lon (float): Longitud
            
        Returns:
            float: Distancia en kilómetros, o None si no está disponible
        """
        if self.distance_calculator is None:
            print("⚠️  Calculadora de distancias no disponible")
            return None
        
        try:
            self.stats['distance_calculations'] += 1
            return self.distance_calculator.distance_to_coast(lat, lon)
        except Exception as e:
            print(f"⚠️  Error calculando distancia para ({lat}, {lon}): {e}")
            return None
    
    def distance_batch(self, coordinates):
        """
        Calcular distancias para un lote de coordenadas
        
        Args:
            coordinates (list): Lista de tuplas (lat, lon)
            
        Returns:
            list: Lista de distancias en kilómetros
        """
        if self.distance_calculator is None:
            print("⚠️  Calculadora de distancias no disponible")
            return [None] * len(coordinates)
        
        return self.distance_calculator.distance_batch(coordinates)
    
    def classify_with_distance(self, lat, lon):
        """
        Clasificar coordenada y calcular distancia a la costa
        
        Args:
            lat (float): Latitud
            lon (float): Longitud
            
        Returns:
            dict: Resultado con clasificación y distancia
        """
        # Obtener clasificación
        classification_result = self.classify(lat, lon)
        
        # Calcular distancia si está disponible
        distance = self.distance_to_coast(lat, lon)
        
        # Combinar resultados
        result = classification_result.copy()
        result['distance_to_coast_km'] = distance
        
        return result
    
    def classify_batch_with_distance(self, coordinates):
        """
        Clasificar lote de coordenadas y calcular distancias
        
        Args:
            coordinates (list): Lista de tuplas (lat, lon)
            
        Returns:
            list: Lista de resultados con clasificación y distancia
        """
        results = []
        for lat, lon in coordinates:
            result = self.classify_with_distance(lat, lon)
            results.append(result)
        return results
    
    def get_stats(self):
        """Obtener estadísticas del pipeline"""
        total = self.stats['total_processed']
        if total == 0:
            return self.stats
        
        stats = {
            **self.stats,
            'global_classification_rate': self.stats['global_classifications'] / total * 100,
            'error_rate': self.stats['errors'] / total * 100
        }
        
        # Agregar estadísticas de distancias si está disponible
        if self.distance_calculator is not None:
            distance_stats = self.distance_calculator.get_distance_stats()
            if "error" not in distance_stats:
                stats['distance_stats'] = distance_stats
        
        return stats
    
    def print_stats(self):
        """Imprimir estadísticas del pipeline"""
        stats = self.get_stats()
        print("\n" + "="*60)
        print("ESTADÍSTICAS DEL PIPELINE")
        print("="*60)
        print(f"Total procesadas: {stats['total_processed']}")
        print(f"Clasificaciones globales: {stats['global_classifications']} ({stats['global_classification_rate']:.1f}%)")
        print(f"Cálculos distancia: {stats['distance_calculations']}")
        print(f"Errores: {stats['errors']} ({stats['error_rate']:.1f}%)")
        
        # Mostrar estadísticas de distancias si están disponibles
        if 'distance_stats' in stats:
            dist_stats = stats['distance_stats']
            print(f"\nESTADÍSTICAS DE DISTANCIAS:")
            print(f"  Distancia mínima: {dist_stats['min_distance_km']:.2f} km")
            print(f"  Distancia máxima: {dist_stats['max_distance_km']:.2f} km")
            print(f"  Distancia promedio: {dist_stats['mean_distance_km']:.2f} km")
            print(f"  Uso memoria: {dist_stats['memory_usage_mb']:.0f} MB")
        
        print("="*60)

