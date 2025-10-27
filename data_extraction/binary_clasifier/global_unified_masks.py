#!/usr/bin/env python3
"""
Script para unificar múltiples imágenes MODIS binarias (water=1, land=0) 
en un mapa global del planeta completo con escala 5000m.

Basado en datos de Google Earth Engine MODIS/006/MOD44W
- Agua = 1 (azul)
- Tierra = 0 (verde)
- Resolución: ~0.0449 grados/píxel (5000m)
- CRS: EPSG:4326
"""

import os
import glob
import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.transform import from_bounds
import argparse
from pathlib import Path

# ===== CONFIGURACIÓN =====
SCALE = 5000  # Escala en metros (Google Earth Engine)
RESOLUTION_DEGREES = 0.0449  # Grados por píxel con scale=5000
WATER_VALUE = 1
LAND_VALUE = 0
GLOBAL_BOUNDS = (-180, -90, 180, 90)  # (min_lon, min_lat, max_lon, max_lat)
TARGET_CRS = 'EPSG:4326'

# Colores para visualización
WATER_COLOR = (0, 100, 200)    # Azul para agua
LAND_COLOR = (34, 139, 34)     # Verde para tierra

def get_image_files(folder_path):
    """Obtiene todas las imágenes .tif de la carpeta"""
    extensions = ['*.tif', '*.tiff']
    
    image_files = []
    for ext in extensions:
        pattern = os.path.join(folder_path, '**', ext)
        files = glob.glob(pattern, recursive=True)
        image_files.extend(files)
    
    print(f"📁 Encontradas {len(image_files)} imágenes en {folder_path}")
    return sorted(image_files)

def analyze_image_metadata(image_files):
    """Analiza los metadatos de las imágenes para entender la cobertura"""
    print("\n🔍 Analizando metadatos de las imágenes...")
    
    all_bounds = []
    resolutions = []
    crs_list = []
    
    for img_path in image_files[:5]:  # Analizar solo las primeras 5 para eficiencia
        try:
            with rasterio.open(img_path) as src:
                all_bounds.append(src.bounds)
                resolutions.append(abs(src.transform.a))
                crs_list.append(str(src.crs))
                
                print(f"  📄 {os.path.basename(img_path)}")
                print(f"    Bounds: {src.bounds}")
                print(f"    Tamaño: {src.width} x {src.height}")
                print(f"    Resolución: {abs(src.transform.a):.6f} grados/pixel")
                print(f"    CRS: {src.crs}")
                
        except Exception as e:
            print(f"    ❌ Error leyendo {img_path}: {e}")
    
    if all_bounds:
        # Calcular bounds globales reales
        min_x = min(bounds.left for bounds in all_bounds)
        min_y = min(bounds.bottom for bounds in all_bounds)
        max_x = max(bounds.right for bounds in all_bounds)
        max_y = max(bounds.top for bounds in all_bounds)
        
        avg_resolution = np.mean(resolutions)
        unique_crs = set(crs_list)
        
        print(f"\n📊 Resumen:")
        print(f"  Bounds detectados: {min_x:.3f}, {min_y:.3f}, {max_x:.3f}, {max_y:.3f}")
        print(f"  Resolución promedio: {avg_resolution:.6f} grados/pixel")
        print(f"  CRS únicos: {len(unique_crs)} ({', '.join(unique_crs)})")
        
        return {
            'bounds': (min_x, min_y, max_x, max_y),
            'resolution': avg_resolution,
            'crs_list': unique_crs
        }
    
    return None

def create_global_canvas():
    """Crea un canvas global con la resolución correcta"""
    print(f"\n🌍 Creando canvas global...")
    
    # Usar bounds globales completos del planeta
    min_x, min_y, max_x, max_y = GLOBAL_BOUNDS
    
    # Calcular dimensiones basadas en la resolución de scale=5000
    width = int((max_x - min_x) / RESOLUTION_DEGREES)
    height = int((max_y - min_y) / RESOLUTION_DEGREES)
    
    print(f"  Dimensiones globales: {width} x {height}")
    print(f"  Resolución: {RESOLUTION_DEGREES:.6f} grados/pixel")
    print(f"  Bounds: {min_x}, {min_y}, {max_x}, {max_y}")
    
    # Crear transform global
    global_transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    # Inicializar canvas con agua (1) - fondo oceánico
    global_canvas = np.full((height, width), WATER_VALUE, dtype=np.uint8)
    
    profile = {
        'driver': 'GTiff',
        'dtype': np.uint8,
        'nodata': None,
        'width': width,
        'height': height,
        'count': 1,
        'crs': TARGET_CRS,
        'transform': global_transform,
        'compress': 'lzw'
    }
    
    return global_canvas, profile, (min_x, min_y, max_x, max_y)

def process_and_place_image(img_path, global_canvas, global_bounds, global_profile):
    """Procesa una imagen individual y la coloca en el canvas global"""
    try:
        with rasterio.open(img_path) as src:
            # Leer datos
            data = src.read(1)
            
            # Verificar que los datos estén en formato correcto (0=tierra, 1=agua)
            unique_values = np.unique(data)
            print(f"  📄 {os.path.basename(img_path)} - Valores únicos: {unique_values}")
            
            # Asegurar que sean uint8 y manejar posibles NaN
            binary_data = np.where(np.isnan(data), WATER_VALUE, data).astype(np.uint8)
            
            # Obtener bounds de la imagen fuente
            src_bounds = src.bounds
            min_x, min_y, max_x, max_y = global_bounds
            
            # Calcular posición en el canvas global
            col_start = int((src_bounds.left - min_x) / RESOLUTION_DEGREES)
            col_end = int((src_bounds.right - min_x) / RESOLUTION_DEGREES)
            row_start = int((max_y - src_bounds.top) / RESOLUTION_DEGREES)  # Y invertido para rasters
            row_end = int((max_y - src_bounds.bottom) / RESOLUTION_DEGREES)
            
            # Asegurar que estén dentro de los límites del canvas
            col_start = max(0, min(col_start, global_canvas.shape[1]))
            col_end = max(0, min(col_end, global_canvas.shape[1]))
            row_start = max(0, min(row_start, global_canvas.shape[0]))
            row_end = max(0, min(row_end, global_canvas.shape[0]))
            
            if col_end > col_start and row_end > row_start:
                # Redimensionar si es necesario
                target_height = row_end - row_start
                target_width = col_end - col_start
                
                if binary_data.shape != (target_height, target_width):
                    from scipy.ndimage import zoom
                    zoom_y = target_height / binary_data.shape[0]
                    zoom_x = target_width / binary_data.shape[1]
                    binary_data = zoom(binary_data, (zoom_y, zoom_x), order=0)  # nearest neighbor
                
                # Colocar datos en el canvas global
                global_canvas[row_start:row_end, col_start:col_end] = binary_data
                
                # Estadísticas
                land_pixels = (binary_data == LAND_VALUE).sum()
                water_pixels = (binary_data == WATER_VALUE).sum()
                print(f"    ✅ Colocado: {land_pixels:,} tierra, {water_pixels:,} agua")
                
                return True
            else:
                print(f"    ⚠️ Imagen fuera de bounds del canvas global")
                return False
                
    except Exception as e:
        print(f"    ❌ Error procesando {img_path}: {e}")
        return False

def create_colored_output(binary_path, output_path):
    """Crea una versión coloreada del mapa binario"""
    print(f"\n🎨 Creando versión coloreada...")
    
    try:
        with rasterio.open(binary_path) as src:
            data = src.read(1)
            
            # Crear imagen RGB
            rgb_data = np.zeros((3, data.shape[0], data.shape[1]), dtype=np.uint8)
            
            # Aplicar colores
            water_mask = (data == WATER_VALUE)
            land_mask = (data == LAND_VALUE)
            
            rgb_data[0][water_mask] = WATER_COLOR[0]  # R
            rgb_data[1][water_mask] = WATER_COLOR[1]  # G
            rgb_data[2][water_mask] = WATER_COLOR[2]  # B
            
            rgb_data[0][land_mask] = LAND_COLOR[0]    # R
            rgb_data[1][land_mask] = LAND_COLOR[1]    # G
            rgb_data[2][land_mask] = LAND_COLOR[2]    # B
            
            # Crear perfil para imagen RGB
            profile = src.profile.copy()
            profile.update({
                'count': 3,
                'dtype': np.uint8,
                'compress': 'lzw'
            })
            
            # Guardar imagen coloreada
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(rgb_data)
            
            print(f"  ✅ Imagen coloreada guardada: {output_path}")
            
    except Exception as e:
        print(f"  ❌ Error creando imagen coloreada: {e}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description='Unificar imágenes MODIS binarias en mapa global del planeta',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python global_unified_masks.py ./imagenes_modis
  python global_unified_masks.py ./imagenes_modis --output mapa_planeta.tif
  python global_unified_masks.py ./imagenes_modis --colored --output mapa_coloreado.tif
        """
    )
    
    parser.add_argument('input_folder', help='Carpeta con las imágenes MODIS .tif')
    parser.add_argument('--output', default='global_planet_map.tif', 
                       help='Archivo de salida (binario)')
    parser.add_argument('--colored', action='store_true',
                       help='Crear también una versión coloreada')
    parser.add_argument('--colored-output', default='global_planet_map_colored.tif',
                       help='Archivo de salida coloreado')
    
    args = parser.parse_args()
    
    # Verificar carpeta de entrada
    if not os.path.exists(args.input_folder):
        print(f"❌ Error: La carpeta {args.input_folder} no existe")
        return
    
    print("🌍 UNIFICADOR DE MAPAS GLOBALES MODIS")
    print("=" * 50)
    print(f"📁 Carpeta de entrada: {args.input_folder}")
    print(f"📄 Archivo de salida: {args.output}")
    print(f"🎨 Versión coloreada: {'Sí' if args.colored else 'No'}")
    print(f"📏 Escala: {SCALE}m ({RESOLUTION_DEGREES:.4f} grados/pixel)")
    print(f"💧 Agua: {WATER_VALUE} (azul)")
    print(f"🏔️ Tierra: {LAND_VALUE} (verde)")
    print("=" * 50)
    
    # Obtener archivos de imagen
    image_files = get_image_files(args.input_folder)
    
    if not image_files:
        print("❌ No se encontraron imágenes .tif en la carpeta especificada")
        return
    
    # Analizar metadatos
    metadata = analyze_image_metadata(image_files)
    
    # Crear canvas global
    global_canvas, global_profile, global_bounds = create_global_canvas()
    
    # Procesar cada imagen
    print(f"\n🔄 Procesando {len(image_files)} imágenes...")
    successful_placements = 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Procesando: {os.path.basename(img_path)}")
        
        if process_and_place_image(img_path, global_canvas, global_bounds, global_profile):
            successful_placements += 1
    
    # Estadísticas finales
    total_land = (global_canvas == LAND_VALUE).sum()
    total_water = (global_canvas == WATER_VALUE).sum()
    total_pixels = global_canvas.size
    
    print(f"\n📊 ESTADÍSTICAS FINALES")
    print("=" * 30)
    print(f"Imágenes procesadas exitosamente: {successful_placements}/{len(image_files)}")
    print(f"Total píxeles: {total_pixels:,}")
    print(f"Píxeles de tierra (0): {total_land:,} ({total_land/total_pixels*100:.2f}%)")
    print(f"Píxeles de agua (1): {total_water:,} ({total_water/total_pixels*100:.2f}%)")
    print(f"Resolución final: {RESOLUTION_DEGREES:.6f} grados/pixel")
    print(f"Dimensiones: {global_canvas.shape[1]} x {global_canvas.shape[0]}")
    
    # Guardar mapa binario
    print(f"\n💾 Guardando mapa global...")
    with rasterio.open(args.output, 'w', **global_profile) as dst:
        dst.write(global_canvas, 1)
    
    print(f"✅ Mapa global guardado: {args.output}")
    
    # Crear versión coloreada si se solicita
    if args.colored:
        colored_path = args.colored_output
        create_colored_output(args.output, colored_path)
    
    print(f"\n🎉 PROCESAMIENTO COMPLETADO")
    print(f"📄 Archivo binario: {args.output}")
    if args.colored:
        print(f"🎨 Archivo coloreado: {args.colored_output}")
    print(f"🌍 Cobertura: Planeta completo ({GLOBAL_BOUNDS})")
    print(f"📏 Resolución: {SCALE}m por píxel")

if __name__ == "__main__":
    main()
