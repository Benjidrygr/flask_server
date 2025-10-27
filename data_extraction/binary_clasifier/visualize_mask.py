#!/usr/bin/env python3

import os
import numpy as np
import rasterio
import matplotlib.pyplot as plt

def debug_and_visualize_tif(tif_path):
    """
    Diagnostica y visualiza un archivo TIF con información detallada
    """
    if not os.path.exists(tif_path):
        print(f"❌ Archivo no encontrado: {tif_path}")
        return
    
    print("="*60)
    print(f"🔍 DIAGNÓSTICO DE ARCHIVO: {os.path.basename(tif_path)}")
    print("="*60)
    
    try:
        with rasterio.open(tif_path) as src:
            # Información básica
            print(f"📏 Dimensiones: {src.width} x {src.height} píxeles")
            print(f"🎯 Bandas: {src.count}")
            print(f"📊 Tipo de datos: {src.dtypes[0]}")
            print(f"🌍 CRS: {src.crs}")
            print(f"📍 Bounds: {src.bounds}")
            print(f"❓ NoData: {src.nodata}")
            
            # Leer datos
            data = src.read(1)
            print(f"🔢 Shape del array: {data.shape}")
            print(f"📈 Tipo numpy: {data.dtype}")
            
            # Estadísticas básicas
            print(f"📊 Min/Max valores: {np.min(data)} / {np.max(data)}")
            print(f"🔍 Valores únicos: {np.unique(data)}")
            
            # Verificar si hay datos
            if np.all(data == src.nodata) if src.nodata is not None else False:
                print("⚠️  PROBLEMA: Todos los píxeles son NoData")
                return
            
            if np.all(np.isnan(data)):
                print("⚠️  PROBLEMA: Todos los píxeles son NaN")
                return
                
            # Contar valores
            unique_vals, counts = np.unique(data, return_counts=True)
            print("\n📋 DISTRIBUCIÓN DE VALORES:")
            for val, count in zip(unique_vals, counts):
                percentage = (count / data.size) * 100
                val_name = "Agua" if val == 1 else "Tierra" if val == 0 else f"Valor_{val}"
                print(f"   {val_name} ({val}): {count:,} píxeles ({percentage:.2f}%)")
            
            bounds = src.bounds
            
            # Mostrar coordenadas de manera clara
            print(f"\n📍 COORDENADAS GEOGRÁFICAS:")
            print(f"   Latitud:  {bounds.bottom:.6f}° a {bounds.top:.6f}°")
            print(f"   Longitud: {bounds.left:.6f}° a {bounds.right:.6f}°")
            print(f"   Rango lat: {bounds.top - bounds.bottom:.6f}°")
            print(f"   Rango lon: {bounds.right - bounds.left:.6f}°")
            
    except Exception as e:
        print(f"❌ Error al leer archivo: {e}")
        return
    
    # Preparar datos para visualización
    if src.nodata is not None:
        # Reemplazar NoData con NaN para mejor visualización
        data_display = data.astype(float)
        data_display[data == src.nodata] = np.nan
    else:
        data_display = data
    
    print("\n" + "="*60)
    print("🖼️  CREANDO VISUALIZACIÓN...")
    print("="*60)
    
    # Crear figura con una sola visualización grande
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Variables para almacenar coordenadas de clic
    click_coords = []
    click_text = None
    
    # Imagen completa
    im = ax.imshow(data_display, cmap='RdYlBu', 
                   extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                   interpolation='nearest')
    ax.set_title(f'Visualizador de Imagen: {os.path.basename(tif_path)}', fontsize=16, pad=20)
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Agregar colorbar
    cbar = plt.colorbar(im, ax=ax, label='Valor (0=Tierra, 1=Agua)', shrink=0.8)
    cbar.ax.tick_params(labelsize=10)
    
    
    # Información en la figura
    valid_data = data_display[~np.isnan(data_display)]
    
    # Función para manejar clics del mouse
    def on_click(event):
        nonlocal click_coords, click_text
        
        if event.inaxes == ax:
            # Obtener coordenadas del clic
            lon = event.xdata
            lat = event.ydata
            
            if lon is not None and lat is not None:
                click_coords.append((lon, lat))
                
                # Limpiar texto anterior
                if click_text:
                    click_text.remove()
                
                # Mostrar coordenadas del clic
                click_text = ax.text(0.02, 0.95, f'Clic {len(click_coords)}: Lon={lon:.6f}, Lat={lat:.6f}', 
                                     transform=ax.transAxes, fontsize=12, fontweight='bold',
                                     bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9))
                
                print(f"🖱️  Clic {len(click_coords)}: Longitud={lon:.6f}°, Latitud={lat:.6f}°")
                
                # Si tenemos 2 clics, mostrar comando de recorte
                if len(click_coords) == 2:
                    lon1, lat1 = click_coords[0]
                    lon2, lat2 = click_coords[1]
                    
                    min_lon = min(lon1, lon2)
                    max_lon = max(lon1, lon2)
                    min_lat = min(lat1, lat2)
                    max_lat = max(lat1, lat2)
                    
                    crop_cmd = f"""Comando de recorte:
python crop_masks.py {tif_path} cropped_{os.path.basename(tif_path)} {min_lat:.3f} {max_lat:.3f} {min_lon:.3f} {max_lon:.3f}"""
                    
                    # Mostrar comando solo en consola
                    
                    print(f"\n🎯 COMANDO DE RECORTE GENERADO:")
                    print(f"   python crop_masks.py {tif_path} cropped_{os.path.basename(tif_path)} {min_lat:.3f} {max_lat:.3f} {min_lon:.3f} {max_lon:.3f}")
                    print(f"   Rango: Lat {min_lat:.3f}° a {max_lat:.3f}°, Lon {min_lon:.3f}° a {max_lon:.3f}°")
                
                # Si tenemos más de 2 clics, reiniciar
                elif len(click_coords) > 2:
                    click_coords = [(lon, lat)]
                    print("🔄 Reiniciando selección...")
                
                fig.canvas.draw()
    
    # Conectar el evento de clic
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Visualización completada")
    
    # Recomendaciones
    print("\n" + "="*60)
    print("💡 RECOMENDACIONES:")
    print("="*60)
    
    if len(valid_data) == 0:
        print("❌ El archivo no contiene datos válidos")
        print("   → Verifica la exportación desde Google Earth Engine")
        print("   → Asegúrate de que la región tenga datos MODIS")
    elif len(np.unique(valid_data)) <= 2:
        print("✅ Archivo binario correcto (tierra/agua)")
        print("   → El archivo debería funcionar bien para análisis")
    else:
        print("⚠️  Archivo contiene múltiples valores")
        print("   → Puede necesitar reclasificación a binario")

def main():
    import sys
    
    if len(sys.argv) > 1:
        tif_path = sys.argv[1]
    else:
        # Buscar archivos TIF en el directorio actual
        current_dir = os.getcwd()
        tif_files = [f for f in os.listdir(current_dir) if f.lower().endswith(('.tif', '.tiff'))]
        
        if not tif_files:
            print("❌ No se encontraron archivos TIF en el directorio actual")
            print("💡 Uso: python debug_tiff.py <ruta_al_archivo.tif>")
            return
        
        print("📁 Archivos TIF encontrados:")
        for i, f in enumerate(tif_files):
            print(f"   {i+1}. {f}")
        
        try:
            choice = int(input("\n🔢 Selecciona un archivo (número): ")) - 1
            tif_path = tif_files[choice]
        except (ValueError, IndexError):
            print("❌ Selección inválida")
            return
    
    debug_and_visualize_tif(tif_path)

if __name__ == "__main__":
    main()