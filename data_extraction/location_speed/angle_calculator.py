"""
Calculador de Ángulo de Dirección
Sistema para calcular ángulos de dirección basado en coordenadas GPS.
"""

import math
from typing import List
from dataclasses import dataclass


@dataclass
class Coordenada:
    lat: float
    lon: float


@dataclass
class ResultadoAngulo:
    posicion_actual: Coordenada
    angulo: float
    direccion_basica: str
    direccion_precisa: str


class CalculadorAngulo:
    def __init__(self, ventana_promedio: int = 3, ventana_angulo: int = 9):
        self.ventana_promedio = ventana_promedio
        self.ventana_angulo = ventana_angulo
        
        if ventana_angulo % ventana_promedio != 0:
            raise ValueError(f"VENTANA_ANGULO ({ventana_angulo}) debe ser divisible por VENTANA_PROMEDIO ({ventana_promedio})")
        
        self.num_grupos = ventana_angulo // ventana_promedio
        if self.num_grupos < 2:
            raise ValueError("Se necesitan al menos 2 grupos para calcular dirección")
    
    def promediar_coordenadas(self, coordenadas: List[Coordenada]) -> Coordenada:
        if not coordenadas:
            raise ValueError("Lista de coordenadas vacía")
        
        lat_promedio = sum(coord.lat for coord in coordenadas) / len(coordenadas)
        lon_promedio = sum(coord.lon for coord in coordenadas) / len(coordenadas)
        
        return Coordenada(lat_promedio, lon_promedio)
    
    def agrupar_y_promediar(self, coordenadas: List[Coordenada]) -> List[Coordenada]:
        if len(coordenadas) != self.ventana_angulo:
            raise ValueError(f"Se esperan {self.ventana_angulo} coordenadas, se recibieron {len(coordenadas)}")
        
        puntos_promediados = []
        
        for i in range(self.num_grupos):
            inicio = i * self.ventana_promedio
            fin = inicio + self.ventana_promedio
            grupo = coordenadas[inicio:fin]
            
            punto_promediado = self.promediar_coordenadas(grupo)
            puntos_promediados.append(punto_promediado)
        
        return puntos_promediados
    
    def calcular_bearing(self, punto_inicial: Coordenada, punto_final: Coordenada) -> float:
        lat1_rad = math.radians(punto_inicial.lat)
        lat2_rad = math.radians(punto_final.lat)
        dlon_rad = math.radians(punto_final.lon - punto_inicial.lon)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def angulo_a_direccion_basica(self, angulo: float) -> str:
        direcciones_basicas = {
            (337.5, 22.5): "N",
            (22.5, 67.5): "NE", 
            (67.5, 112.5): "E",
            (112.5, 157.5): "SE",
            (157.5, 202.5): "S",
            (202.5, 247.5): "SW",
            (247.5, 292.5): "W",
            (292.5, 337.5): "NW"
        }
        
        for (min_ang, max_ang), direccion in direcciones_basicas.items():
            if min_ang <= angulo < max_ang or (min_ang > max_ang and (angulo >= min_ang or angulo < max_ang)):
                return direccion
        
        return "N"
    
    def angulo_a_direccion_precisa(self, angulo: float) -> str:
        direcciones_precisas = {
            (348.75, 11.25): "N",
            (11.25, 33.75): "NNE",
            (33.75, 56.25): "NE",
            (56.25, 78.75): "ENE",
            (78.75, 101.25): "E",
            (101.25, 123.75): "ESE",
            (123.75, 146.25): "SE",
            (146.25, 168.75): "SSE",
            (168.75, 191.25): "S",
            (191.25, 213.75): "SSW",
            (213.75, 236.25): "SW",
            (236.25, 258.75): "WSW",
            (258.75, 281.25): "W",
            (281.25, 303.75): "WNW",
            (303.75, 326.25): "NW",
            (326.25, 348.75): "NNW"
        }
        
        for (min_ang, max_ang), direccion in direcciones_precisas.items():
            if min_ang <= angulo < max_ang or (min_ang > max_ang and (angulo >= min_ang or angulo < max_ang)):
                return direccion
        
        return "N"
    
    def calcular_direccion(self, coordenadas: List[Coordenada]) -> ResultadoAngulo:
        puntos_promediados = self.agrupar_y_promediar(coordenadas)
        
        punto_inicial = puntos_promediados[0]
        punto_final = puntos_promediados[-1]
        
        angulo = self.calcular_bearing(punto_inicial, punto_final)
        
        direccion_basica = self.angulo_a_direccion_basica(angulo)
        direccion_precisa = self.angulo_a_direccion_precisa(angulo)
        
        return ResultadoAngulo(
            posicion_actual=punto_final,
            angulo=angulo,
            direccion_basica=direccion_basica,
            direccion_precisa=direccion_precisa
        )


def main():
    """
    Calculador de ángulo de dirección con coordenadas reales.
    """
    print("=== CALCULADOR DE ÁNGULO DE DIRECCIÓN ===\n")
    
    # Coordenadas reales de ejemplo (movimiento en agua)
    coordenadas = [
        Coordenada(11.653928, -55.592672),  # 01:50:16
        Coordenada(11.653927, -55.59268),   # 01:50:17
        Coordenada(11.653927, -55.59268),   # 01:50:18
        Coordenada(11.653936, -55.592704),  # 01:50:19
        Coordenada(11.653936, -55.592704),  # 01:50:20
        Coordenada(11.653944, -55.592716),  # 01:50:21
        Coordenada(11.653944, -55.592716),  # 01:50:22
        Coordenada(11.653958, -55.592744),  # 01:50:23
        Coordenada(11.653958, -55.592744)   # 01:50:24
    ]
    
    # Crear calculador
    calculador = CalculadorAngulo(ventana_promedio=3, ventana_angulo=9)
    
    # Calcular dirección
    resultado = calculador.calcular_direccion(coordenadas)
    
    # Obtener puntos promediados para mostrar dirección completa
    puntos_promediados = calculador.agrupar_y_promediar(coordenadas)
    punto_inicial = puntos_promediados[0]
    
    # Mostrar resultados
    print(f"Posición inicial: ({punto_inicial.lat:.6f}, {punto_inicial.lon:.6f})")
    print(f"Posición final: ({resultado.posicion_actual.lat:.6f}, {resultado.posicion_actual.lon:.6f})")
    print(f"Dirección: {punto_inicial.lat:.6f}, {punto_inicial.lon:.6f} → {resultado.posicion_actual.lat:.6f}, {resultado.posicion_actual.lon:.6f}")
    print(f"Ángulo de dirección: {resultado.angulo:.1f}°")
    print(f"Dirección básica: {resultado.direccion_basica}")
    print(f"Dirección precisa: {resultado.direccion_precisa}")


if __name__ == "__main__":
    main()