#!/usr/bin/env python3

import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import glob

def crop_mask(input_path, output_path, min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    if not os.path.exists(input_path):
        return False
    
    try:
        with rasterio.open(input_path) as src:
            data = src.read(1)
            bounds = src.bounds
            transform = src.transform
            
            # Usar bounds de la imagen si no se especifican coordenadas
            if min_lat is None:
                min_lat = bounds.bottom
            if max_lat is None:
                max_lat = bounds.top
            if min_lon is None:
                min_lon = bounds.left
            if max_lon is None:
                max_lon = bounds.right
            
            print(f"  Recortando: lat {min_lat:.3f} a {max_lat:.3f}, lon {min_lon:.3f} a {max_lon:.3f}")
            print(f"  Bounds originales: {bounds}")
            
            # Verificar si el rango solicitado está dentro de los bounds
            if min_lat < bounds.bottom or max_lat > bounds.top or min_lon < bounds.left or max_lon > bounds.right:
                print(f"  ⚠️ Rango solicitado está fuera de los bounds de la imagen")
                return False
            
            # Calcular filas y columnas para el recorte
            min_row = int((bounds.top - max_lat) / abs(transform.e))
            max_row = int((bounds.top - min_lat) / abs(transform.e))
            min_col = int((min_lon - bounds.left) / abs(transform.a))
            max_col = int((max_lon - bounds.left) / abs(transform.a))
            
            # Asegurar que los índices estén en orden correcto
            if min_row > max_row:
                min_row, max_row = max_row, min_row
            if min_col > max_col:
                min_col, max_col = max_col, min_col
            
            # Asegurar que estén dentro de los límites de la imagen
            min_row = max(0, min(min_row, data.shape[0]))
            max_row = max(0, min(max_row, data.shape[0]))
            min_col = max(0, min(min_col, data.shape[1]))
            max_col = max(0, min(max_col, data.shape[1]))
            
            # Verificar si hay datos para recortar
            if min_row >= max_row or min_col >= max_col:
                print(f"  ❌ No hay datos en el rango especificado")
                return False
            
            print(f"  Índices de recorte: filas {min_row}-{max_row}, columnas {min_col}-{max_col}")
            
            # Recortar datos
            cropped_data = data[min_row:max_row, min_col:max_col]
            
            # Ajustar transform para el recorte
            # Calcular las coordenadas correctas del recorte
            new_left = bounds.left + (min_col * abs(transform.a))
            new_right = bounds.left + (max_col * abs(transform.a))
            new_top = bounds.top - (min_row * abs(transform.e))
            new_bottom = bounds.top - (max_row * abs(transform.e))
            
            new_transform = rasterio.transform.from_bounds(
                new_left, new_bottom, new_right, new_top,
                cropped_data.shape[1], cropped_data.shape[0]
            )
            
            print(f"  Nuevos bounds: {new_left:.3f}, {new_bottom:.3f}, {new_right:.3f}, {new_top:.3f}")
            
            # Guardar archivo recortado
            profile = src.profile.copy()
            profile.update({
                'height': cropped_data.shape[0],
                'width': cropped_data.shape[1],
                'transform': new_transform
            })
            
            with rasterio.open(output_path, 'w', **profile) as dst:
                dst.write(cropped_data, 1)
            
            return True
            
    except Exception as e:
        return False

def crop_all_masks(input_dir, output_dir, min_lat=None, max_lat=None, min_lon=None, max_lon=None):
    os.makedirs(output_dir, exist_ok=True)
    
    tif_files = glob.glob(os.path.join(input_dir, "*.tif"))
    
    if not tif_files:
        return
    
    for tif_file in tif_files:
        filename = os.path.basename(tif_file)
        output_path = os.path.join(output_dir, f"cropped_{filename}")
        print(f"\n📄 Procesando: {filename}")
        result = crop_mask(tif_file, output_path, min_lat, max_lat, min_lon, max_lon)
        if result:
            print(f"  ✅ Recorte exitoso: {os.path.basename(output_path)}")
        else:
            print(f"  ❌ Error en recorte: {filename}")

def crop_single_image(input_path, output_path, min_lat, max_lat, min_lon, max_lon):
    """Recorta una imagen específica con coordenadas personalizadas"""
    print(f"🔪 RECORTE PERSONALIZADO")
    print(f"📁 Entrada: {input_path}")
    print(f"📄 Salida: {output_path}")
    print(f"📍 Coordenadas: lat {min_lat} a {max_lat}, lon {min_lon} a {max_lon}")
    print("-" * 50)
    
    result = crop_mask(input_path, output_path, min_lat, max_lat, min_lon, max_lon)
    
    if result:
        print(f"\n✅ Recorte completado exitosamente!")
        print(f"📄 Archivo creado: {output_path}")
    else:
        print(f"\n❌ Error en el recorte")
    
    return result

def main():
    import sys
    
    # ===== CONFIGURACIÓN HARDCODEADA =====
    # Cambia estas variables según necesites
    MAX_LAT = 14.43   # Latitud máxima (norte)
    MIN_LAT = -59.9  # Latitud mínima (sur)
    
    # Longitud se mantiene completa (sin recortar)
    # =====================================
    
    if len(sys.argv) >= 2:
        # Modo: recorte con coordenadas hardcodeadas
        input_path = sys.argv[1]
        
        # Generar nombre de salida automáticamente
        if len(sys.argv) >= 3:
            output_path = sys.argv[2]
        else:
            # Generar nombre automático
            base_name = os.path.splitext(os.path.basename(input_path))[0]
            output_path = f"cropped_{base_name}.tif"
        
        print("🔪 RECORTE CON COORDENADAS HARDCODEADAS")
        print(f"📁 Entrada: {input_path}")
        print(f"📄 Salida: {output_path}")
        print(f"📍 Rango latitud: {MIN_LAT} a {MAX_LAT}")
        print("📍 Longitud: completa (sin recortar)")
        print("-" * 50)
        
        # Usar bounds completos para longitud (no recortar horizontalmente)
        min_lon = None
        max_lon = None
        
        crop_single_image(input_path, output_path, MIN_LAT, MAX_LAT, min_lon, max_lon)
        
    else:
        # Modo: mostrar ayuda
        print("🔪 CROP MASKS - Recorte de Imágenes MODIS")
        print("=" * 50)
        print("📖 USO:")
        print("   python crop_masks.py <imagen_entrada> [imagen_salida]")
        print()
        print("📝 EJEMPLOS:")
        print("   # Con nombre de salida automático")
        print("   python crop_masks.py binary_masks/imagen.tif")
        print()
        print("   # Con nombre de salida personalizado")
        print("   python crop_masks.py binary_masks/imagen.tif mi_imagen_recortada.tif")
        print()
        print("📍 CONFIGURACIÓN ACTUAL:")
        print(f"   Latitud mínima: {MIN_LAT}")
        print(f"   Latitud máxima: {MAX_LAT}")
        print("   Longitud: completa (sin recortar)")
        print()
        print("💡 Para cambiar las coordenadas, edita las variables MIN_LAT y MAX_LAT en el script")
        print("💡 TIP: Usa visualize_mask.py para ver las coordenadas de la imagen")
        print("   python visualize_mask.py binary_masks/imagen.tif")

if __name__ == "__main__":
    main()
