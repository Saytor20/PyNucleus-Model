"""
Redis Cache Integration for PyNucleus

Provides distributed caching functionality for RAG responses, reducing
computation overhead and improving response times.

Features:
- Query-based response caching
- TTL management
- Cache warming strategies
- Performance metrics
- Fallback handling
"""

import os
import json
import hashlib
import logging
import redis
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from functools import wraps

logger = logging.getLogger(__name__)

class RedisCache:
    """Redis-based distributed cache for PyNucleus"""
    
    def __init__(self, redis_url: str = None, default_ttl: int = 3600):
        self.redis_url = redis_url or os.getenv('REDIS_URL', 'redis://localhost:6379/0')
        self.default_ttl = default_ttl
        self.client = None
        self.enabled = True
        
        try:
            self.client = redis.from_url(self.redis_url, decode_responses=True)
            # Test connection
            self.client.ping()
            logger.info(f"Redis cache initialized: {self.redis_url}")
        except Exception as e:
            logger.warning(f"Redis unavailable, caching disabled: {e}")
            self.enabled = False
    
    def _generate_key(self, query: str, context: Dict = None) -> str:
        """Generate cache key from query and context"""
        # Create hash from query and relevant context
        key_data = {
            'query': query.lower().strip(),
            'context': context or {}
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return f"rag:{hashlib.sha256(key_string.encode()).hexdigest()[:16]}"
    
    def get(self, query: str, context: Dict = None) -> Optional[Dict]:
        """Get cached response for query"""
        if not self.enabled:
            return None
        
        try:
            key = self._generate_key(query, context)
            cached_data = self.client.get(key)
            
            if cached_data:
                response = json.loads(cached_data)
                
                # Add cache metadata
                response['metadata'] = response.get('metadata', {})
                response['metadata']['cache_hit'] = True
                response['metadata']['cached_at'] = response.get('cached_at')
                
                logger.debug(f"Cache hit for query: {query[:50]}...")
                return response
                
        except Exception as e:
            logger.error(f"Cache get error: {e}")
        
        return None
    
    def set(self, query: str, response: Dict, context: Dict = None, ttl: int = None) -> bool:
        """Cache response for query"""
        if not self.enabled:
            return False
        
        try:
            key = self._generate_key(query, context)
            cache_ttl = ttl or self.default_ttl
            
            # Add cache metadata to response
            cached_response = response.copy()
            cached_response['cached_at'] = datetime.now().isoformat()
            cached_response['cache_ttl'] = cache_ttl
            
            self.client.setex(key, cache_ttl, json.dumps(cached_response))
            logger.debug(f"Cached response for query: {query[:50]}... (TTL: {cache_ttl}s)")
            return True
            
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def delete(self, query: str, context: Dict = None) -> bool:
        """Delete cached response"""
        if not self.enabled:
            return False
        
        try:
            key = self._generate_key(query, context)
            result = self.client.delete(key)
            return result > 0
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        if not self.enabled:
            return 0
        
        try:
            keys = list(self.client.scan_iter(match=f"rag:{pattern}*"))
            if keys:
                return self.client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self.enabled:
            return {'enabled': False}
        
        try:
            info = self.client.info()
            
            # Count RAG cache keys
            rag_keys = list(self.client.scan_iter(match="rag:*"))
            
            return {
                'enabled': True,
                'total_keys': len(rag_keys),
                'memory_usage_mb': info.get('used_memory', 0) / (1024 * 1024),
                'hit_rate': info.get('keyspace_hits', 0) / max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0)),
                'connected_clients': info.get('connected_clients', 0),
                'uptime_seconds': info.get('uptime_in_seconds', 0)
            }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {'enabled': False, 'error': str(e)}
    
    def warm_cache(self, queries: List[str], context: Dict = None) -> int:
        """Warm cache with common queries"""
        if not self.enabled:
            return 0
        
        warmed_count = 0
        logger.info(f"Warming cache with {len(queries)} queries")
        
        for query in queries:
            try:
                # Only warm if not already cached
                if not self.get(query, context):
                    # This would need to integrate with your RAG pipeline
                    # For now, just log the query
                    logger.debug(f"Would warm cache for: {query[:50]}...")
                    warmed_count += 1
            except Exception as e:
                logger.error(f"Cache warming error for query '{query}': {e}")
        
        logger.info(f"Cache warming completed: {warmed_count} queries")
        return warmed_count

# Global cache instance
_cache_instance = None

def get_cache() -> RedisCache:
    """Get global cache instance"""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = RedisCache()
    return _cache_instance

def cached_rag_response(ttl: int = None, context_keys: List[str] = None):
    """Decorator for caching RAG responses"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Extract query from arguments
            query = None
            if args:
                query = args[0] if isinstance(args[0], str) else None
            if not query and 'query' in kwargs:
                query = kwargs['query']
            if not query and 'question' in kwargs:
                query = kwargs['question']
            
            if not query:
                # No query to cache, call function directly
                return func(*args, **kwargs)
            
            # Build context from specified keys
            context = {}
            if context_keys:
                for key in context_keys:
                    if key in kwargs:
                        context[key] = kwargs[key]
            
            # Try to get from cache
            cached_response = cache.get(query, context)
            if cached_response:
                return cached_response
            
            # Call function and cache result
            response = func(*args, **kwargs)
            
            # Only cache successful responses
            if response and not response.get('error'):
                cache.set(query, response, context, ttl)
            
            return response
        
        return wrapper
    return decorator

def invalidate_cache_on_error(func):
    """Decorator to invalidate cache on errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # On error, invalidate relevant cache entries
            cache = get_cache()
            query = args[0] if args and isinstance(args[0], str) else None
            if query:
                cache.delete(query)
            raise
    
    return wrapper

class CacheMetrics:
    """Cache performance metrics collector"""
    
    def __init__(self):
        self.cache = get_cache()
        self.hit_count = 0
        self.miss_count = 0
        self.error_count = 0
    
    def record_hit(self):
        """Record cache hit"""
        self.hit_count += 1
    
    def record_miss(self):
        """Record cache miss"""
        self.miss_count += 1
    
    def record_error(self):
        """Record cache error"""
        self.error_count += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        total_requests = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total_requests) if total_requests > 0 else 0
        
        cache_stats = self.cache.get_stats()
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'error_count': self.error_count,
            'hit_rate': hit_rate,
            'cache_stats': cache_stats
        }
    
    def reset(self):
        """Reset metrics counters"""
        self.hit_count = 0
        self.miss_count = 0
        self.error_count = 0

# Global metrics instance
_metrics_instance = CacheMetrics()

def get_cache_metrics() -> CacheMetrics:
    """Get global cache metrics instance"""
    return _metrics_instance 