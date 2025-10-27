"""
Pipeline de Ángulo y Velocidad
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .angle_calculator import CalculadorAngulo, Coordenada as CoordenadaAngulo
from .speed_calculator import CalculadorVelocidad, Coordenada as CoordenadaVelocidad


class PipelineAngleSpeed:
    def __init__(self):
        self.calculador_angulo = CalculadorAngulo(ventana_promedio=3, ventana_angulo=9)
        self.calculador_velocidad = CalculadorVelocidad()
    
    def convertir_coordenadas(self, coords_velocidad: list) -> list:
        return [CoordenadaAngulo(coord.lat, coord.lon) for coord in coords_velocidad]
    
    def filtrar_coordenadas_por_angulo(self, coordenadas: list, angulo_objetivo: float, tolerancia: float = 30.0) -> list:
        coordenadas_filtradas = [coordenadas[0]]
        
        for i in range(1, len(coordenadas)):
            coord_anterior = coordenadas[i-1]
            coord_actual = coordenadas[i]
            
            coord_ang_anterior = CoordenadaAngulo(coord_anterior.lat, coord_anterior.lon)
            coord_ang_actual = CoordenadaAngulo(coord_actual.lat, coord_actual.lon)
            
            angulo_actual = self.calculador_angulo.calcular_bearing(coord_ang_anterior, coord_ang_actual)
            
            diferencia = abs(angulo_actual - angulo_objetivo)
            if diferencia > 180:
                diferencia = 360 - diferencia
            
            if diferencia <= tolerancia:
                coordenadas_filtradas.append(coord_actual)
        
        return coordenadas_filtradas
    
    def calcular_velocidad_ajustada_por_angulo(self, coordenadas: list, angulo_objetivo: float) -> dict:
        coordenadas_filtradas = self.filtrar_coordenadas_por_angulo(coordenadas, angulo_objetivo)
        
        if len(coordenadas_filtradas) < 2:
            return None
        
        resultado_filtrado = self.calculador_velocidad.calcular_velocidad_promedio(coordenadas_filtradas)
        coordenadas_promediadas_filtradas = self.calculador_velocidad.agrupar_y_promediar(coordenadas_filtradas, grupo_size=2)
        resultado_promediado_filtrado = self.calculador_velocidad.calcular_velocidad_promedio(coordenadas_promediadas_filtradas)
        
        return {
            "coordenadas_filtradas": coordenadas_filtradas,
            "coordenadas_promediadas_filtradas": coordenadas_promediadas_filtradas,
            "velocidad_filtrada": resultado_filtrado,
            "velocidad_promediada_filtrada": resultado_promediado_filtrado
        }
    
    def calcular_angulos_y_velocidades_dinamicos(self, coordenadas_velocidad: list) -> list:
        """Calcula ángulos y velocidades cada 9 coordenadas a lo largo del recorrido"""
        segmentos = []
        
        for i in range(0, len(coordenadas_velocidad) - 8, 9):
            # Tomar 9 coordenadas consecutivas
            coordenadas_segmento = coordenadas_velocidad[i:i+9]
            coordenadas_angulo = self.convertir_coordenadas(coordenadas_segmento)
            resultado_angulo = self.calculador_angulo.calcular_direccion(coordenadas_angulo)
            
            # Calcular velocidad para este segmento
            resultado_velocidad = self.calculador_velocidad.calcular_velocidad_promedio(coordenadas_segmento)
            
            segmentos.append({
                "indice_inicio": i,
                "indice_fin": i + 8,
                "angulo": resultado_angulo.angulo,
                "direccion_basica": resultado_angulo.direccion_basica,
                "direccion_precisa": resultado_angulo.direccion_precisa,
                "velocidad_kmh": resultado_velocidad.velocidad_kmh,
                "velocidad_nudos": resultado_velocidad.velocidad_nudos,
                "distancia_metros": resultado_velocidad.distancia_metros,
                "tiempo_segundos": resultado_velocidad.tiempo_segundos
            })
        
        return segmentos
    
    def calcular_velocidad_ajustada_dinamica(self, coordenadas_velocidad: list, segmentos: list) -> dict:
        """Calcula velocidad ajustada usando segmentos dinámicos"""
        coordenadas_filtradas = []
        
        for i, coord in enumerate(coordenadas_velocidad):
            # Encontrar el segmento correspondiente para esta coordenada
            angulo_aplicar = None
            for segmento in segmentos:
                if segmento["indice_inicio"] <= i <= segmento["indice_fin"]:
                    angulo_aplicar = segmento["angulo"]
                    break
            
            if angulo_aplicar is not None:
                # Verificar si esta coordenada sigue la dirección del ángulo
                if i > 0:
                    coord_anterior = coordenadas_velocidad[i-1]
                    coord_ang_anterior = self.convertir_coordenadas([coord_anterior])[0]
                    coord_ang_actual = self.convertir_coordenadas([coord])[0]
                    
                    angulo_actual = self.calculador_angulo.calcular_bearing(coord_ang_anterior, coord_ang_actual)
                    
                    diferencia = abs(angulo_actual - angulo_aplicar)
                    if diferencia > 180:
                        diferencia = 360 - diferencia
                    
                    if diferencia <= 30.0:  # Tolerancia de 30 grados
                        coordenadas_filtradas.append(coord)
                else:
                    # Primera coordenada siempre se incluye
                    coordenadas_filtradas.append(coord)
        
        if len(coordenadas_filtradas) < 2:
            return None
        
        resultado_filtrado = self.calculador_velocidad.calcular_velocidad_promedio(coordenadas_filtradas)
        coordenadas_promediadas_filtradas = self.calculador_velocidad.agrupar_y_promediar(coordenadas_filtradas, grupo_size=2)
        resultado_promediado_filtrado = self.calculador_velocidad.calcular_velocidad_promedio(coordenadas_promediadas_filtradas)
        
        return {
            "coordenadas_filtradas": coordenadas_filtradas,
            "coordenadas_promediadas_filtradas": coordenadas_promediadas_filtradas,
            "velocidad_filtrada": resultado_filtrado,
            "velocidad_promediada_filtrada": resultado_promediado_filtrado
        }
    
    def procesar_pipeline(self, coordenadas_velocidad: list) -> dict:
        if len(coordenadas_velocidad) < 9:
            raise ValueError("Se necesitan al menos 9 coordenadas")
        
        # Calcular ángulos y velocidades dinámicos cada 9 coordenadas
        segmentos_dinamicos = self.calcular_angulos_y_velocidades_dinamicos(coordenadas_velocidad)
        
        # Calcular ángulo promedio para el resultado principal
        angulos_valores = [s["angulo"] for s in segmentos_dinamicos]
        angulo_promedio = sum(angulos_valores) / len(angulos_valores)
        
        # Calcular velocidad promedio de todos los segmentos
        velocidades_kmh = [s["velocidad_kmh"] for s in segmentos_dinamicos]
        velocidad_promedio_kmh = sum(velocidades_kmh) / len(velocidades_kmh)
        
        velocidades_nudos = [s["velocidad_nudos"] for s in segmentos_dinamicos]
        velocidad_promedio_nudos = sum(velocidades_nudos) / len(velocidades_nudos)
        
        # Determinar dirección basada en el ángulo promedio
        if 0 <= angulo_promedio < 22.5 or 337.5 <= angulo_promedio <= 360:
            direccion_basica = "N"
        elif 22.5 <= angulo_promedio < 67.5:
            direccion_basica = "NE"
        elif 67.5 <= angulo_promedio < 112.5:
            direccion_basica = "E"
        elif 112.5 <= angulo_promedio < 157.5:
            direccion_basica = "SE"
        elif 157.5 <= angulo_promedio < 202.5:
            direccion_basica = "S"
        elif 202.5 <= angulo_promedio < 247.5:
            direccion_basica = "SW"
        elif 247.5 <= angulo_promedio < 292.5:
            direccion_basica = "W"
        else:
            direccion_basica = "NW"
        
        # Crear objeto resultado_angulo con el promedio
        class ResultadoAngulo:
            def __init__(self, angulo, direccion_basica, direccion_precisa):
                self.angulo = angulo
                self.direccion_basica = direccion_basica
                self.direccion_precisa = direccion_precisa
        
        resultado_angulo = ResultadoAngulo(angulo_promedio, direccion_basica, direccion_basica)
        
        # Crear objeto resultado_velocidad con el promedio de segmentos
        class ResultadoVelocidad:
            def __init__(self, velocidad_kmh, velocidad_nudos, distancia_metros, tiempo_segundos):
                self.velocidad_kmh = velocidad_kmh
                self.velocidad_nudos = velocidad_nudos
                self.distancia_metros = distancia_metros
                self.tiempo_segundos = tiempo_segundos
        
        # Calcular distancia y tiempo totales
        distancia_total = sum(s["distancia_metros"] for s in segmentos_dinamicos)
        tiempo_total = sum(s["tiempo_segundos"] for s in segmentos_dinamicos)
        
        resultado_velocidad_original = ResultadoVelocidad(
            velocidad_promedio_kmh, 
            velocidad_promedio_nudos, 
            distancia_total, 
            tiempo_total
        )
        
        resultado_ajustado = self.calcular_velocidad_ajustada_dinamica(coordenadas_velocidad, segmentos_dinamicos)
        
        return {
            "angulo": resultado_angulo,
            "segmentos_dinamicos": segmentos_dinamicos,
            "velocidad_original": resultado_velocidad_original,
            "velocidad_ajustada": resultado_ajustado
        }


def main():
    coordenadas = [
        CoordenadaVelocidad(11.653928, -55.592672, 1748051416),
        CoordenadaVelocidad(11.653927, -55.59268, 1748051417),
        CoordenadaVelocidad(11.653927, -55.59268, 1748051418),
        CoordenadaVelocidad(11.653936, -55.592704, 1748051419),
        CoordenadaVelocidad(11.653936, -55.592704, 1748051420),
        CoordenadaVelocidad(11.653944, -55.592716, 1748051421),
        CoordenadaVelocidad(11.653944, -55.592716, 1748051422),
        CoordenadaVelocidad(11.653958, -55.592744, 1748051423),
        CoordenadaVelocidad(11.653958, -55.592744, 1748051424),
        CoordenadaVelocidad(11.653961, -55.592756, 1748051425),
        CoordenadaVelocidad(11.653961, -55.592756, 1748051426),
        CoordenadaVelocidad(11.653971, -55.592756, 1748051427),
    ]
    
    pipeline = PipelineAngleSpeed()
    resultado = pipeline.procesar_pipeline(coordenadas)
    
    print(f"Ángulo: {resultado['angulo'].angulo:.1f}° ({resultado['angulo'].direccion_precisa})")
    print(f"Velocidad original: {resultado['velocidad_original'].velocidad_kmh:.2f} km/h")
    
    if resultado['velocidad_ajustada']:
        print(f"Velocidad ajustada: {resultado['velocidad_ajustada']['velocidad_promediada_filtrada'].velocidad_kmh:.2f} km/h")
        mejora = resultado['velocidad_ajustada']['velocidad_promediada_filtrada'].velocidad_kmh - resultado['velocidad_original'].velocidad_kmh
        print(f"Mejora: {mejora:+.2f} km/h")


if __name__ == "__main__":
    main()