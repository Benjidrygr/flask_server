"""
Calculador de Velocidad
Sistema para calcular velocidades basado en coordenadas GPS.
"""

import math
from typing import List
from dataclasses import dataclass


@dataclass
class Coordenada:
    lat: float
    lon: float
    timestamp: float = 0.0  # Timestamp en segundos


@dataclass
class ResultadoVelocidad:
    velocidad_kmh: float
    velocidad_nudos: float
    distancia_metros: float
    tiempo_segundos: float


class CalculadorVelocidad:
    def __init__(self):
        self.R = 6371000  # Radio de la Tierra en metros
    
    def calcular_distancia(self, coord1: Coordenada, coord2: Coordenada) -> float:
        """Calcula distancia entre dos coordenadas en metros."""
        lat1_rad = math.radians(coord1.lat)
        lat2_rad = math.radians(coord2.lat)
        dlat_rad = math.radians(coord2.lat - coord1.lat)
        dlon_rad = math.radians(coord2.lon - coord1.lon)
        
        a = (math.sin(dlat_rad/2) * math.sin(dlat_rad/2) +
             math.cos(lat1_rad) * math.cos(lat2_rad) *
             math.sin(dlon_rad/2) * math.sin(dlon_rad/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return self.R * c
    
    def calcular_velocidad(self, coord_inicial: Coordenada, coord_final: Coordenada) -> ResultadoVelocidad:
        """Calcula velocidad entre dos coordenadas."""
        distancia = self.calcular_distancia(coord_inicial, coord_final)
        tiempo = coord_final.timestamp - coord_inicial.timestamp
        
        if tiempo <= 0:
            raise ValueError("El tiempo debe ser mayor que 0")
        
        velocidad_ms = distancia / tiempo  # m/s
        velocidad_kmh = velocidad_ms * 3.6  # km/h
        velocidad_nudos = velocidad_ms * 1.94384  # nudos (1 m/s = 1.94384 nudos)
        
        return ResultadoVelocidad(
            velocidad_kmh=velocidad_kmh,
            velocidad_nudos=velocidad_nudos,
            distancia_metros=distancia,
            tiempo_segundos=tiempo
        )
    
    def promediar_coordenadas(self, coordenadas: List[Coordenada]) -> Coordenada:
        """Calcula el promedio de una lista de coordenadas."""
        if not coordenadas:
            raise ValueError("Lista de coordenadas vacía")
        
        lat_promedio = sum(coord.lat for coord in coordenadas) / len(coordenadas)
        lon_promedio = sum(coord.lon for coord in coordenadas) / len(coordenadas)
        timestamp_promedio = sum(coord.timestamp for coord in coordenadas) / len(coordenadas)
        
        return Coordenada(lat_promedio, lon_promedio, timestamp_promedio)
    
    def agrupar_y_promediar(self, coordenadas: List[Coordenada], grupo_size: int = 2) -> List[Coordenada]:
        """Agrupa coordenadas y calcula el promedio de cada grupo."""
        coordenadas_promediadas = []
        
        for i in range(0, len(coordenadas), grupo_size):
            grupo = coordenadas[i:i+grupo_size]
            if len(grupo) == grupo_size:  # Solo procesar grupos completos
                punto_promediado = self.promediar_coordenadas(grupo)
                coordenadas_promediadas.append(punto_promediado)
        
        return coordenadas_promediadas
    
    def calcular_velocidad_promedio(self, coordenadas: List[Coordenada]) -> ResultadoVelocidad:
        """Calcula velocidad promedio de una trayectoria."""
        if len(coordenadas) < 2:
            raise ValueError("Se necesitan al menos 2 coordenadas")
        
        distancia_total = 0
        tiempo_total = 0
        
        for i in range(1, len(coordenadas)):
            distancia = self.calcular_distancia(coordenadas[i-1], coordenadas[i])
            tiempo = coordenadas[i].timestamp - coordenadas[i-1].timestamp
            
            distancia_total += distancia
            tiempo_total += tiempo
        
        if tiempo_total <= 0:
            raise ValueError("El tiempo total debe ser mayor que 0")
        
        velocidad_ms = distancia_total / tiempo_total
        velocidad_kmh = velocidad_ms * 3.6
        velocidad_nudos = velocidad_ms * 1.94384
        
        return ResultadoVelocidad(
            velocidad_kmh=velocidad_kmh,
            velocidad_nudos=velocidad_nudos,
            distancia_metros=distancia_total,
            tiempo_segundos=tiempo_total
        )


def main():
    """
    Calculador de velocidad con coordenadas reales.
    """
    print("=== CALCULADOR DE VELOCIDAD ===\n")
    
    # Coordenadas reales con timestamps (simulando 1 segundo entre cada punto)
    coordenadas = [
        Coordenada(11.653928, -55.592672, 1748051416),  # 01:50:16
        Coordenada(11.653927, -55.59268, 1748051417),   # 01:50:17
        Coordenada(11.653927, -55.59268, 1748051418),   # 01:50:18
        Coordenada(11.653936, -55.592704, 1748051419),  # 01:50:19
        Coordenada(11.653936, -55.592704, 1748051420),  # 01:50:20
        Coordenada(11.653944, -55.592716, 1748051421),  # 01:50:21
        Coordenada(11.653944, -55.592716, 1748051422),  # 01:50:22
        Coordenada(11.653958, -55.592744, 1748051423),  # 01:50:23
        Coordenada(11.653958, -55.592744, 1748051424)   # 01:50:24
    ]
    
    calculador = CalculadorVelocidad()
    
    # Calcular velocidad promedio de toda la trayectoria original
    resultado = calculador.calcular_velocidad_promedio(coordenadas)
    
    print("--- COORDENADAS ORIGINALES ---")
    print(f"Distancia total: {resultado.distancia_metros:.2f} metros")
    print(f"Tiempo total: {resultado.tiempo_segundos:.1f} segundos")
    print(f"Velocidad promedio: {resultado.velocidad_kmh:.2f} km/h")
    print(f"Velocidad promedio: {resultado.velocidad_nudos:.2f} nudos")
    
    print("\n--- COORDENADAS PROMEDIADAS (2 en 2) ---")
    
    # Procesar coordenadas promediadas de 2 en 2
    coordenadas_promediadas = calculador.agrupar_y_promediar(coordenadas, grupo_size=2)
    
    print("Coordenadas promediadas:")
    for i, coord in enumerate(coordenadas_promediadas, 1):
        print(f"  P{i}: ({coord.lat:.6f}, {coord.lon:.6f}) - Tiempo: {coord.timestamp}")
    
    # Calcular velocidad promedio de coordenadas promediadas
    resultado_promediado = calculador.calcular_velocidad_promedio(coordenadas_promediadas)
    
    print(f"\nDistancia total (promediada): {resultado_promediado.distancia_metros:.2f} metros")
    print(f"Tiempo total (promediado): {resultado_promediado.tiempo_segundos:.1f} segundos")
    print(f"Velocidad promedio (promediada): {resultado_promediado.velocidad_kmh:.2f} km/h")
    print(f"Velocidad promedio (promediada): {resultado_promediado.velocidad_nudos:.2f} nudos")
    
    print("\n--- Velocidades entre puntos promediados ---")
    
    # Calcular velocidades entre puntos promediados consecutivos
    for i in range(1, len(coordenadas_promediadas)):
        resultado_punto = calculador.calcular_velocidad(coordenadas_promediadas[i-1], coordenadas_promediadas[i])
        print(f"Punto {i} → {i+1}: {resultado_punto.velocidad_kmh:.2f} km/h ({resultado_punto.velocidad_nudos:.2f} nudos)")


if __name__ == "__main__":
    main()
