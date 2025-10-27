#!/usr/bin/env python3
"""
Script para medir y comparar overhead de diferentes enfoques de Celery
"""

import time
import psutil
import json
import os
from datetime import datetime
from typing import Dict, Any

# Importar clientes
from .celery_client import YOLOCeleryClient
from .celery_client_optimized import YOLOCeleryClientOptimized

# Importar funciones directas
from yolo.video_processor_celery import process_videos_celery_compatible
# Removed unified_pipeline import
from .celery_config import YOLO_CONFIG

def measure_direct_call(video_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Medir llamada directa (sin Celery)"""
    print("🔄 Midiendo llamada directa...")
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Llamada directa
    result = process_videos_celery_compatible(video_folder, config)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    return {
        "method": "direct_call",
        "duration_seconds": end_time - start_time,
        "memory_mb": (end_memory - start_memory) / (1024 * 1024),
        "success": result.get("success", False),
        "result_size_kb": len(json.dumps(result, default=str)) / 1024
    }

def measure_celery_original(video_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Medir Celery original (con overhead)"""
    print("🔄 Midiendo Celery original...")
    
    client = YOLOCeleryClient()
    
    # Verificar que el servidor esté funcionando
    if not client.check_server_health():
        return {
            "method": "celery_original",
            "error": "Servidor Celery no disponible",
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        }
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Enviar tarea Celery
    task_id = client.send_single_task("process_videos", video_folder, config)
    if not task_id:
        return {
            "method": "celery_original",
            "error": "Error enviando tarea",
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        }
    
    # Monitorear hasta completar
    result = client.monitor_task(task_id, timeout=300)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    if result:
        return {
            "method": "celery_original",
            "duration_seconds": result['duration_seconds'],
            "memory_mb": (end_memory - start_memory) / (1024 * 1024),
            "success": result['success'],
            "result_size_kb": len(json.dumps(result['result'], default=str)) / 1024,
            "task_id": task_id
        }
    else:
        return {
            "method": "celery_original",
            "error": "Tarea falló o timeout",
            "duration_seconds": time.time() - start_time,
            "memory_mb": (end_memory - start_memory) / (1024 * 1024),
            "success": False
        }

def measure_celery_optimized(video_folder: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Medir Celery optimizado (con cache)"""
    print("🔄 Midiendo Celery optimizado...")
    
    client = YOLOCeleryClientOptimized()
    
    # Verificar que el servidor esté funcionando
    if not client.check_server_health():
        return {
            "method": "celery_optimized",
            "error": "Servidor Celery no disponible",
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        }
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Enviar tarea optimizada
    task_id = client.send_cached_yolo_task(video_folder, config)
    if not task_id:
        return {
            "method": "celery_optimized",
            "error": "Error enviando tarea",
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        }
    
    # Monitorear hasta completar
    result = client.monitor_task(task_id, timeout=300)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    if result:
        return {
            "method": "celery_optimized",
            "duration_seconds": result['duration_seconds'],
            "memory_mb": (end_memory - start_memory) / (1024 * 1024),
            "success": result['success'],
            "result_size_kb": len(json.dumps(result['result'], default=str)) / 1024,
            "cached": result.get('cached', False),
            "task_id": task_id
        }
    else:
        return {
            "method": "celery_optimized",
            "error": "Tarea falló o timeout",
            "duration_seconds": time.time() - start_time,
            "memory_mb": (end_memory - start_memory) / (1024 * 1024),
            "success": False
        }

def measure_pipeline_optimized(video_folder: str, unified_config: Dict[str, Any], yolo_config: Dict[str, Any]) -> Dict[str, Any]:
    """Medir pipeline completo optimizado"""
    print("🔄 Midiendo pipeline completo optimizado...")
    
    client = YOLOCeleryClientOptimized()
    
    # Verificar que el servidor esté funcionando
    if not client.check_server_health():
        return {
            "method": "pipeline_optimized",
            "error": "Servidor Celery no disponible",
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        }
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    # Enviar pipeline optimizado
    task_id = client.send_optimized_pipeline(video_folder, unified_config, yolo_config)
    if not task_id:
        return {
            "method": "pipeline_optimized",
            "error": "Error enviando pipeline",
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        }
    
    # Monitorear hasta completar
    result = client.monitor_task(task_id, timeout=600)  # 10 minutos timeout
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    if result:
        return {
            "method": "pipeline_optimized",
            "duration_seconds": result['duration_seconds'],
            "memory_mb": (end_memory - start_memory) / (1024 * 1024),
            "success": result['success'],
            "result_size_kb": len(json.dumps(result['result'], default=str)) / 1024,
            "cached": result.get('cached', False),
            "task_id": task_id
        }
    else:
        return {
            "method": "pipeline_optimized",
            "error": "Pipeline falló o timeout",
            "duration_seconds": time.time() - start_time,
            "memory_mb": (end_memory - start_memory) / (1024 * 1024),
            "success": False
        }

def print_comparison(results: list):
    """Imprimir comparación de resultados"""
    print("\n" + "="*80)
    print("📊 COMPARACIÓN DE OVERHEAD")
    print("="*80)
    
    # Filtrar resultados exitosos
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("❌ No hay resultados exitosos para comparar")
        return
    
    # Encontrar el más rápido
    fastest = min(successful_results, key=lambda x: x['duration_seconds'])
    
    print(f"{'Método':<20} {'Tiempo (s)':<12} {'Memoria (MB)':<15} {'Tamaño (KB)':<12} {'Overhead':<10}")
    print("-" * 80)
    
    for result in successful_results:
        method = result['method']
        duration = result['duration_seconds']
        memory = result['memory_mb']
        size = result.get('result_size_kb', 0)
        
        # Calcular overhead relativo al más rápido
        if fastest['duration_seconds'] > 0:
            overhead = ((duration / fastest['duration_seconds']) - 1) * 100
        else:
            overhead = 0
        
        # Indicador de cache
        cache_indicator = "🎯" if result.get('cached', False) else "🔄"
        
        print(f"{method:<20} {duration:<12.3f} {memory:<15.2f} {size:<12.1f} {overhead:>+8.1f}% {cache_indicator}")
    
    print("-" * 80)
    
    # Análisis detallado
    print("\n📈 ANÁLISIS DETALLADO:")
    
    for result in successful_results:
        method = result['method']
        duration = result['duration_seconds']
        memory = result['memory_mb']
        
        if fastest['duration_seconds'] > 0:
            time_overhead = ((duration / fastest['duration_seconds']) - 1) * 100
        else:
            time_overhead = 0
        
        if fastest['memory_mb'] > 0:
            memory_overhead = ((memory / fastest['memory_mb']) - 1) * 100
        else:
            memory_overhead = 0
        
        print(f"\n🔍 {method.upper()}:")
        print(f"   ⏱️  Tiempo: {duration:.3f}s ({time_overhead:+.1f}% vs más rápido)")
        print(f"   💾 Memoria: {memory:.2f}MB ({memory_overhead:+.1f}% vs más rápido)")
        
        if result.get('cached', False):
            print(f"   🎯 Cache: Hit (overhead reducido)")
        else:
            print(f"   🔄 Cache: Miss (procesado)")
    
    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    
    if any(r.get('cached', False) for r in successful_results):
        print("   ✅ Usar cache para reducir overhead en ejecuciones repetitivas")
    
    if any('optimized' in r['method'] for r in successful_results):
        print("   ✅ Usar configuración optimizada de Celery")
    
    if any('direct' in r['method'] for r in successful_results):
        print("   ✅ Usar llamadas directas para pipelines secuenciales")
    
    print("   ✅ Monitorear métricas regularmente para optimización continua")

def main():
    """Función principal de medición"""
    print("🚀 Iniciando medición de overhead de Celery")
    print("="*50)
    
    # Configuración de prueba
    video_folder = "./videos_test"
    yolo_config = YOLO_CONFIG.copy()
    unified_config = {
        "enable_geographic_classification": True,
        "enable_speed_analysis": True,
        "enable_video_processing": True,
        "verbose": True
    }
    
    # Verificar que la carpeta existe
    if not os.path.exists(video_folder):
        print(f"❌ La carpeta {video_folder} no existe")
        print("💡 Creando carpeta de prueba...")
        os.makedirs(video_folder, exist_ok=True)
        print(f"✅ Carpeta creada: {video_folder}")
    
    results = []
    
    # Medir llamada directa
    try:
        result = measure_direct_call(video_folder, yolo_config)
        results.append(result)
    except Exception as e:
        print(f"❌ Error en llamada directa: {e}")
        results.append({
            "method": "direct_call",
            "error": str(e),
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        })
    
    # Medir Celery original
    try:
        result = measure_celery_original(video_folder, yolo_config)
        results.append(result)
    except Exception as e:
        print(f"❌ Error en Celery original: {e}")
        results.append({
            "method": "celery_original",
            "error": str(e),
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        })
    
    # Medir Celery optimizado
    try:
        result = measure_celery_optimized(video_folder, yolo_config)
        results.append(result)
    except Exception as e:
        print(f"❌ Error en Celery optimizado: {e}")
        results.append({
            "method": "celery_optimized",
            "error": str(e),
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        })
    
    # Medir pipeline optimizado
    try:
        result = measure_pipeline_optimized(video_folder, unified_config, yolo_config)
        results.append(result)
    except Exception as e:
        print(f"❌ Error en pipeline optimizado: {e}")
        results.append({
            "method": "pipeline_optimized",
            "error": str(e),
            "duration_seconds": 0,
            "memory_mb": 0,
            "success": False
        })
    
    # Mostrar comparación
    print_comparison(results)
    
    # Guardar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"overhead_measurement_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "video_folder": video_folder,
            "yolo_config": yolo_config,
            "unified_config": unified_config,
            "results": results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Resultados guardados en: {output_file}")

if __name__ == "__main__":
    main()
