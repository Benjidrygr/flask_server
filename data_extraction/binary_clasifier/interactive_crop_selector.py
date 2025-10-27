#!/usr/bin/env python3
"""
Selector interactivo de coordenadas para recorte de imágenes MODIS
Permite hacer clic en la imagen para seleccionar el área de recorte
"""

import os
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def interactive_crop_selector(tif_path):
    """
    Selector interactivo para definir área de recorte
    """
    if not os.path.exists(tif_path):
        print(f"❌ Archivo no encontrado: {tif_path}")
        return
    
    print("🖱️  SELECTOR INTERACTIVO DE RECORTE")
    print("=" * 50)
    print(f"📄 Archivo: {os.path.basename(tif_path)}")
    print("🖱️  INSTRUCCIONES:")
    print("   • Haz clic en 2 puntos para definir el área de recorte")
    print("   • El primer clic define una esquina")
    print("   • El segundo clic define la esquina opuesta")
    print("   • Presiona 'q' para salir")
    print("=" * 50)
    
    try:
        with rasterio.open(tif_path) as src:
            data = src.read(1)
            bounds = src.bounds
            
            print(f"📍 Bounds de la imagen:")
            print(f"   Latitud:  {bounds.bottom:.6f}° a {bounds.top:.6f}°")
            print(f"   Longitud: {bounds.left:.6f}° a {bounds.right:.6f}°")
            print()
            
    except Exception as e:
        print(f"❌ Error al leer archivo: {e}")
        return
    
    # Preparar datos para visualización
    if src.nodata is not None:
        data_display = data.astype(float)
        data_display[data == src.nodata] = np.nan
    else:
        data_display = data
    
    # Crear figura
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Mostrar imagen
    im = ax.imshow(data_display, cmap='RdYlBu', 
                   extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
                   interpolation='nearest')
    
    ax.set_title(f'Selector de Recorte: {os.path.basename(tif_path)}\n🖱️ Haz clic en 2 puntos para definir el área', 
                 fontsize=14, pad=20)
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.grid(True, alpha=0.3)
    
    # Agregar colorbar
    cbar = plt.colorbar(im, ax=ax, label='Valor (0=Tierra, 1=Agua)')
    
    # Variables para almacenar clics
    click_coords = []
    rectangle = None
    coord_text = None
    
    def on_click(event):
        nonlocal click_coords, rectangle, coord_text
        
        if event.inaxes == ax:
            # Obtener coordenadas del clic
            lon = event.xdata
            lat = event.ydata
            
            if lon is not None and lat is not None:
                click_coords.append((lon, lat))
                
                # Limpiar elementos anteriores
                if rectangle:
                    rectangle.remove()
                if coord_text:
                    coord_text.remove()
                
                # Mostrar coordenadas del clic
                coord_text = ax.text(0.02, 0.98, f'Clic {len(click_coords)}: Lon={lon:.6f}°, Lat={lat:.6f}°', 
                                   transform=ax.transAxes, fontsize=12, 
                                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.9),
                                   verticalalignment='top')
                
                print(f"🖱️  Clic {len(click_coords)}: Longitud={lon:.6f}°, Latitud={lat:.6f}°")
                
                # Si tenemos 2 clics, mostrar rectángulo y comando
                if len(click_coords) == 2:
                    lon1, lat1 = click_coords[0]
                    lon2, lat2 = click_coords[1]
                    
                    min_lon = min(lon1, lon2)
                    max_lon = max(lon1, lon2)
                    min_lat = min(lat1, lat2)
                    max_lat = max(lat1, lat2)
                    
                    # Dibujar rectángulo
                    width = max_lon - min_lon
                    height = max_lat - min_lat
                    rectangle = Rectangle((min_lon, min_lat), width, height, 
                                        linewidth=2, edgecolor='red', facecolor='none', alpha=0.8)
                    ax.add_patch(rectangle)
                    
                    # Mostrar comando de recorte
                    crop_cmd = f"""🎯 COMANDO DE RECORTE GENERADO:
python crop_masks.py {tif_path} cropped_{os.path.basename(tif_path)} {min_lat:.3f} {max_lat:.3f} {min_lon:.3f} {max_lon:.3f}

📍 Rango seleccionado:
   Latitud:  {min_lat:.6f}° a {max_lat:.6f}°
   Longitud: {min_lon:.6f}° a {max_lon:.3f}°"""
                    
                    # Limpiar texto anterior
                    for text in ax.texts:
                        if 'COMANDO DE RECORTE' in text.get_text():
                            text.remove()
                    
                    ax.text(0.98, 0.02, crop_cmd, transform=ax.transAxes, fontsize=10, 
                           verticalalignment='bottom', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))
                    
                    print(f"\n🎯 COMANDO DE RECORTE GENERADO:")
                    print(f"   python crop_masks.py {tif_path} cropped_{os.path.basename(tif_path)} {min_lat:.3f} {max_lat:.3f} {min_lon:.3f} {max_lon:.3f}")
                    print(f"   Rango: Lat {min_lat:.3f}° a {max_lat:.3f}°, Lon {min_lon:.3f}° a {max_lon:.3f}°")
                    print(f"\n💡 Puedes copiar y ejecutar este comando para recortar la imagen")
                
                # Si tenemos más de 2 clics, reiniciar
                elif len(click_coords) > 2:
                    click_coords = [(lon, lat)]
                    print("🔄 Reiniciando selección...")
                
                fig.canvas.draw()
    
    def on_key(event):
        if event.key == 'q':
            plt.close()
        elif event.key == 'r':
            # Reiniciar selección
            nonlocal click_coords, rectangle, coord_text
            click_coords = []
            if rectangle:
                rectangle.remove()
                rectangle = None
            if coord_text:
                coord_text.remove()
                coord_text = None
            # Limpiar todos los textos
            for text in ax.texts:
                if 'COMANDO DE RECORTE' in text.get_text() or 'Clic' in text.get_text():
                    text.remove()
            fig.canvas.draw()
            print("🔄 Selección reiniciada")
    
    # Conectar eventos
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Agregar instrucciones
    instructions = """🖱️ CONTROLES:
• Clic izquierdo: Seleccionar punto
• 'r': Reiniciar selección
• 'q': Salir

📍 Bounds de la imagen:
Lat: {:.3f}° a {:.3f}°
Lon: {:.3f}° a {:.3f}°""".format(bounds.bottom, bounds.top, bounds.left, bounds.right)
    
    ax.text(0.02, 0.02, instructions, transform=ax.transAxes, fontsize=10, 
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    print("\n✅ Selector interactivo completado")

def main():
    if len(sys.argv) > 1:
        tif_path = sys.argv[1]
    else:
        # Buscar archivos TIF en el directorio actual
        current_dir = os.getcwd()
        tif_files = [f for f in os.listdir(current_dir) if f.lower().endswith(('.tif', '.tiff'))]
        
        if not tif_files:
            print("❌ No se encontraron archivos TIF en el directorio actual")
            print("💡 Uso: python interactive_crop_selector.py <ruta_al_archivo.tif>")
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
    
    interactive_crop_selector(tif_path)

if __name__ == "__main__":
    main()
