# 🔧 Workflow de Visualización y Recorte de Imágenes MODIS

Este documento explica cómo usar los scripts mejorados para visualizar y recortar imágenes MODIS binarias antes de unificarlas en un mapa global.

## 📋 Scripts Disponibles

### 1. `visualize_mask.py` - Visualización y Análisis
**Propósito**: Visualizar imágenes .tif y obtener coordenadas geográficas precisas.

**Uso**:
```bash
# Visualizar una imagen específica
python visualize_mask.py ruta/a/imagen.tif

# Visualizar desde el directorio actual (seleccionar interactivamente)
python visualize_mask.py
```

**Características**:
- ✅ Muestra coordenadas geográficas precisas (lat/lon)
- ✅ Visualización dual (vista completa + detalle)
- ✅ Líneas de cuadrícula con coordenadas
- ✅ Estadísticas de distribución de valores
- ✅ Comando de recorte sugerido automáticamente
- ✅ Diagnóstico completo del archivo

### 2. `crop_masks.py` - Recorte Personalizado
**Propósito**: Recortar imágenes con coordenadas específicas.

**Uso**:
```bash
# Recorte de imagen específica con coordenadas personalizadas
python crop_masks.py <imagen_entrada> <imagen_salida> <min_lat> <max_lat> <min_lon> <max_lon>

# Ejemplo:
python crop_masks.py binary_masks/MODIS_WaterMask_2015_AF_5000m_PERFECTO.tif cropped_africa.tif -20.0 20.0 -10.0 30.0

# Recorte por defecto (todas las imágenes en binary_masks/)
python crop_masks.py
```

**Características**:
- ✅ Recorte con coordenadas lat/lon precisas
- ✅ Preserva coordenadas geográficas correctas
- ✅ Validación de rangos dentro de bounds
- ✅ Información detallada del proceso
- ✅ Modo batch para múltiples imágenes

### 3. `global_unified_masks.py` - Unificación Global
**Propósito**: Unificar múltiples imágenes en un mapa global del planeta.

**Uso**:
```bash
# Unificación básica
python global_unified_masks.py ./imagenes_modis

# Con archivo de salida personalizado
python global_unified_masks.py ./imagenes_modis --output mapa_planeta.tif

# Con versión coloreada
python global_unified_masks.py ./imagenes_modis --colored --output mapa_coloreado.tif
```

## 🔄 Workflow Recomendado

### Paso 1: Visualizar la Imagen
```bash
python visualize_mask.py binary_masks/MODIS_WaterMask_2015_AF_5000m_PERFECTO.tif
```

**Resultado esperado**:
- Coordenadas geográficas mostradas en consola
- Visualización con líneas de cuadrícula
- Comando de recorte sugerido

### Paso 2: Determinar Coordenadas de Recorte
Basándote en la visualización, decide las coordenadas:
- **Latitud**: rango vertical (ej: -20.0 a 20.0)
- **Longitud**: rango horizontal (ej: -10.0 a 30.0)

### Paso 3: Recortar la Imagen
```bash
python crop_masks.py binary_masks/MODIS_WaterMask_2015_AF_5000m_PERFECTO.tif cropped_africa.tif -20.0 20.0 -10.0 30.0
```

### Paso 4: Verificar el Recorte
```bash
python visualize_mask.py cropped_africa.tif
```

### Paso 5: Unificar en Mapa Global
```bash
python global_unified_masks.py . --output mapa_final.tif --colored
```

## 📊 Ejemplo de Salida

### Visualización:
```
📍 COORDENADAS GEOGRÁFICAS:
   Latitud:  -48.733604° a 46.667479°
   Longitud: -25.018081° a 60.007461°
   Rango lat: 95.401083°
   Rango lon: 85.025542°
```

### Recorte:
```
🔪 RECORTE PERSONALIZADO
📁 Entrada: binary_masks/MODIS_WaterMask_2015_AF_5000m_PERFECTO.tif
📄 Salida: cropped_africa.tif
📍 Coordenadas: lat -20.0 a 20.0, lon -10.0 a 30.0
  ✅ Recorte completado exitosamente!
```

### Unificación:
```
🌍 UNIFICADOR DE MAPAS GLOBALES MODIS
📊 ESTADÍSTICAS FINALES
Imágenes procesadas exitosamente: 16/16
Píxeles de tierra (0): 8,701,570 (27.08%)
Píxeles de agua (1): 23,430,566 (72.92%)
```

## 🎯 Ventajas del Workflow

1. **Precisión**: Coordenadas geográficas exactas
2. **Flexibilidad**: Recorte con cualquier rango de coordenadas
3. **Validación**: Verificación automática de bounds
4. **Visualización**: Vista clara de la región a recortar
5. **Automatización**: Comandos sugeridos automáticamente
6. **Preservación**: Mantiene resolución y coordenadas originales

## 🔧 Parámetros Técnicos

- **Resolución**: 5000m por píxel (~0.0449 grados/píxel)
- **CRS**: EPSG:4326 (WGS84)
- **Formato**: GeoTIFF binario (0=tierra, 1=agua)
- **Compresión**: LZW
- **Canvas global**: -180° a 180° longitud, -90° a 90° latitud

## 🚨 Notas Importantes

1. **Coordenadas**: Siempre usar formato decimal (ej: -20.5, no -20°30')
2. **Orden**: min_lat < max_lat, min_lon < max_lon
3. **Bounds**: Las coordenadas deben estar dentro de los bounds de la imagen original
4. **Resolución**: El recorte mantiene la resolución original (5000m)
5. **Formato**: Las imágenes deben ser binarias (0=tierra, 1=agua)
