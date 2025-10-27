#!/usr/bin/env python3
"""
Sistema de cache para reducir overhead en Celery
"""

import redis
import json
import hashlib
import time
import logging
from typing import Any, Optional, Dict
from functools import wraps

# Configurar logging
logger = logging.getLogger(__name__)

class CacheManager:
    """Gestor de cache optimizado para Celery"""
    
    def __init__(self, host='localhost', port=6379, db=1):
        """
        Inicializar gestor de cache
        
        Args:
            host: Host de Redis
            port: Puerto de Redis
            db: Base de datos de Redis (1 para cache, 0 para Celery)
        """
        try:
            self.redis_client = redis.Redis(
                host=host, 
                port=port, 
                db=db,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            # Verificar conexión
            self.redis_client.ping()
            logger.info("✅ Cache Redis conectado exitosamente")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo conectar a Redis cache: {e}")
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Generar clave única para el cache
        
        Args:
            prefix: Prefijo para la clave
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Clave única para el cache
        """
        # Crear contenido único
        content = {
            'prefix': prefix,
            'args': args,
            'kwargs': kwargs
        }
        
        # Generar hash MD5
        content_str = json.dumps(content, sort_keys=True, default=str)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        
        return f"cache:{prefix}:{content_hash}"
    
    def get(self, prefix: str, *args, **kwargs) -> Optional[Any]:
        """
        Obtener valor del cache
        
        Args:
            prefix: Prefijo para la clave
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            Valor del cache o None si no existe
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(prefix, *args, **kwargs)
            cached_value = self.redis_client.get(cache_key)
            
            if cached_value:
                logger.debug(f"🎯 Cache hit: {prefix}")
                return json.loads(cached_value)
            else:
                logger.debug(f"❌ Cache miss: {prefix}")
                return None
                
        except Exception as e:
            logger.warning(f"⚠️ Error obteniendo cache: {e}")
            return None
    
    def set(self, prefix: str, value: Any, ttl: int = 3600, *args, **kwargs) -> bool:
        """
        Guardar valor en el cache
        
        Args:
            prefix: Prefijo para la clave
            value: Valor a guardar
            ttl: Tiempo de vida en segundos (default: 1 hora)
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            True si se guardó exitosamente, False en caso contrario
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(prefix, *args, **kwargs)
            value_json = json.dumps(value, default=str, ensure_ascii=False)
            
            # Guardar con TTL
            self.redis_client.setex(cache_key, ttl, value_json)
            logger.debug(f"💾 Cache guardado: {prefix} (TTL: {ttl}s)")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error guardando cache: {e}")
            return False
    
    def delete(self, prefix: str, *args, **kwargs) -> bool:
        """
        Eliminar valor del cache
        
        Args:
            prefix: Prefijo para la clave
            *args: Argumentos posicionales
            **kwargs: Argumentos con nombre
            
        Returns:
            True si se eliminó exitosamente, False en caso contrario
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(prefix, *args, **kwargs)
            deleted = self.redis_client.delete(cache_key)
            logger.debug(f"🗑️ Cache eliminado: {prefix}")
            return deleted > 0
            
        except Exception as e:
            logger.warning(f"⚠️ Error eliminando cache: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Limpiar todo el cache
        
        Returns:
            True si se limpió exitosamente, False en caso contrario
        """
        if not self.redis_client:
            return False
        
        try:
            # Eliminar todas las claves que empiecen con "cache:"
            keys = self.redis_client.keys("cache:*")
            if keys:
                deleted = self.redis_client.delete(*keys)
                logger.info(f"🧹 Cache limpiado: {deleted} claves eliminadas")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ Error limpiando cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtener estadísticas del cache
        
        Returns:
            Diccionario con estadísticas del cache
        """
        if not self.redis_client:
            return {"error": "Redis no disponible"}
        
        try:
            info = self.redis_client.info()
            keys = self.redis_client.keys("cache:*")
            
            return {
                "redis_version": info.get("redis_version"),
                "used_memory": info.get("used_memory_human"),
                "connected_clients": info.get("connected_clients"),
                "total_keys": len(keys),
                "cache_keys": len([k for k in keys if k.startswith("cache:")]),
                "uptime_seconds": info.get("uptime_in_seconds")
            }
            
        except Exception as e:
            return {"error": f"Error obteniendo estadísticas: {e}"}

# Instancia global del gestor de cache
cache_manager = CacheManager()

def cached(prefix: str, ttl: int = 3600):
    """
    Decorador para cachear resultados de funciones
    
    Args:
        prefix: Prefijo para las claves de cache
        ttl: Tiempo de vida en segundos (default: 1 hora)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Intentar obtener del cache
            cached_result = cache_manager.get(prefix, *args, **kwargs)
            if cached_result is not None:
                logger.info(f"🎯 Cache hit para {func.__name__}")
                return cached_result
            
            # Ejecutar función y guardar resultado
            logger.info(f"🔄 Ejecutando {func.__name__} (no en cache)")
            result = func(*args, **kwargs)
            
            # Guardar en cache
            cache_manager.set(prefix, result, ttl, *args, **kwargs)
            
            return result
        
        return wrapper
    return decorator

def cache_video_processing(video_folder: str, config: Dict[str, Any], ttl: int = 1800) -> Optional[Any]:
    """
    Cache específico para procesamiento de videos
    
    Args:
        video_folder: Carpeta de videos
        config: Configuración de procesamiento
        ttl: Tiempo de vida en segundos (default: 30 minutos)
        
    Returns:
        Resultado del cache o None
    """
    return cache_manager.get("video_processing", video_folder, config)
    
def set_video_processing_cache(video_folder: str, config: Dict[str, Any], result: Any, ttl: int = 1800) -> bool:
    """
    Guardar resultado de procesamiento de videos en cache
    
    Args:
        video_folder: Carpeta de videos
        config: Configuración de procesamiento
        result: Resultado a guardar
        ttl: Tiempo de vida en segundos (default: 30 minutos)
        
    Returns:
        True si se guardó exitosamente
    """
    return cache_manager.set("video_processing", result, ttl, video_folder, config)

# Funciones de cache para unified_pipeline eliminadas

if __name__ == "__main__":
    # Ejemplo de uso
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Crear instancia
    cache = CacheManager()
    
    # Ejemplo de cache
    test_data = {"videos": 5, "processed": 3, "time": 120.5}
    cache.set("test", test_data, ttl=60)
    
    # Recuperar del cache
    result = cache.get("test")
    print(f"Resultado del cache: {result}")
    
    # Estadísticas
    stats = cache.get_stats()
    print(f"Estadísticas del cache: {stats}")
