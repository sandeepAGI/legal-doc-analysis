from langchain_community.llms import Ollama
from .prompts import QNA_PROMPT_TEMPLATE
import hashlib
import json
import os
import time
from pathlib import Path

def get_llm():
    return Ollama(model="llama3", temperature=0.0)

def synthesize_answer(query, retrieved_chunks, llm=None):
    if not llm:
        llm = get_llm()
    context = "\n\n".join(doc.page_content for doc, _ in retrieved_chunks)
    prompt = QNA_PROMPT_TEMPLATE.format(context=context, question=query)
    return llm.invoke(prompt)

def synthesize_answer_stream(query, retrieved_chunks, llm=None):
    """Streaming version of synthesize_answer that yields response chunks."""
    if not llm:
        llm = get_llm()
    context = "\n\n".join(doc.page_content for doc, _ in retrieved_chunks)
    prompt = QNA_PROMPT_TEMPLATE.format(context=context, question=query)
    for chunk in llm.stream(prompt):
        yield chunk

# Query Response Caching Infrastructure
def _get_cache_dir():
    """Get or create the query cache directory."""
    cache_dir = Path("chroma_stores/query_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    responses_dir = cache_dir / "responses"
    responses_dir.mkdir(exist_ok=True)
    
    return cache_dir

def _generate_query_cache_key(query: str, retrieved_chunks: list, llm_model: str = "llama3") -> str:
    """Generate unique cache key for query + context + model combination."""
    context_content = "\n\n".join(doc.page_content for doc, _ in retrieved_chunks)
    combined = f"{query}|{context_content}|{llm_model}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def _load_from_cache(cache_key: str):
    """Load cached response if available and not expired."""
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / "responses" / f"{cache_key}.json"
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        # Check if cache is still valid (24 hours TTL)
        cache_age = time.time() - cache_data.get('timestamp', 0)
        if cache_age > 86400:  # 24 hours in seconds
            cache_file.unlink()  # Remove expired cache
            return None
        
        return cache_data.get('response')
    except (json.JSONDecodeError, KeyError, OSError):
        # Remove corrupted cache file
        if cache_file.exists():
            cache_file.unlink()
        return None

def _save_to_cache(cache_key: str, response: str):
    """Save response to cache with metadata."""
    cache_dir = _get_cache_dir()
    cache_file = cache_dir / "responses" / f"{cache_key}.json"
    
    cache_data = {
        'response': response,
        'timestamp': time.time(),
        'model': 'llama3'
    }
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
    except OSError:
        # Silently fail if cache write fails - don't break functionality
        pass

def _simulate_streaming(text: str, chunk_delay: float = 0.05):
    """Simulate streaming for cached responses to maintain UI consistency."""
    # Split response into words for smoother streaming simulation
    words = text.split()
    current_chunk = ""
    
    for i, word in enumerate(words):
        current_chunk += word + " "
        # Yield chunks of ~3-5 words to simulate natural streaming
        if (i + 1) % 4 == 0 or i == len(words) - 1:
            yield current_chunk
            current_chunk = ""
            if chunk_delay > 0:
                time.sleep(chunk_delay)

def _update_cache_stats(cache_hit: bool):
    """Update cache statistics for monitoring."""
    cache_dir = _get_cache_dir()
    stats_file = cache_dir / "cache_stats.json"
    
    try:
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                stats = json.load(f)
        else:
            stats = {'hits': 0, 'misses': 0, 'total_queries': 0}
        
        stats['total_queries'] += 1
        if cache_hit:
            stats['hits'] += 1
        else:
            stats['misses'] += 1
        
        stats['hit_rate'] = stats['hits'] / stats['total_queries'] if stats['total_queries'] > 0 else 0
        stats['last_updated'] = time.time()
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
    except (json.JSONDecodeError, OSError):
        # Silently fail if stats update fails
        pass

# Cached wrapper functions
def synthesize_answer_cached(query, retrieved_chunks, llm=None, use_cache=True):
    """Cached version of synthesize_answer with same API."""
    if not use_cache:
        return synthesize_answer(query, retrieved_chunks, llm)
    
    # Check cache first
    cache_key = _generate_query_cache_key(query, retrieved_chunks, "llama3")
    cached_response = _load_from_cache(cache_key)
    
    if cached_response:
        _update_cache_stats(cache_hit=True)
        return cached_response
    
    # Cache miss - call original function
    _update_cache_stats(cache_hit=False)
    response = synthesize_answer(query, retrieved_chunks, llm)
    _save_to_cache(cache_key, response)
    return response

def synthesize_answer_stream_cached(query, retrieved_chunks, llm=None, use_cache=True):
    """Cached version of synthesize_answer_stream."""
    if not use_cache:
        yield from synthesize_answer_stream(query, retrieved_chunks, llm)
        return
    
    cache_key = _generate_query_cache_key(query, retrieved_chunks, "llama3")
    cached_response = _load_from_cache(cache_key)
    
    if cached_response:
        _update_cache_stats(cache_hit=True)
        # Simulate streaming for cached response
        yield from _simulate_streaming(cached_response, chunk_delay=0.05)
        return
    
    # Cache miss - stream and cache simultaneously
    _update_cache_stats(cache_hit=False)
    full_response = ""
    for chunk in synthesize_answer_stream(query, retrieved_chunks, llm):
        full_response += chunk
        yield chunk
    
    _save_to_cache(cache_key, full_response)

def get_query_cache_stats():
    """Get current query cache statistics."""
    cache_dir = _get_cache_dir()
    stats_file = cache_dir / "cache_stats.json"
    
    if not stats_file.exists():
        return {'hits': 0, 'misses': 0, 'total_queries': 0, 'hit_rate': 0.0}
    
    try:
        with open(stats_file, 'r') as f:
            stats = json.load(f)
        return stats
    except (json.JSONDecodeError, OSError):
        return {'hits': 0, 'misses': 0, 'total_queries': 0, 'hit_rate': 0.0}

def _cleanup_expired_cache_entries():
    """Remove expired cache entries (older than 24 hours)."""
    cache_dir = _get_cache_dir()
    responses_dir = cache_dir / "responses"
    
    if not responses_dir.exists():
        return
    
    current_time = time.time()
    removed_count = 0
    
    for cache_file in responses_dir.glob("*.json"):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            cache_age = current_time - cache_data.get('timestamp', 0)
            if cache_age > 86400:  # 24 hours
                cache_file.unlink()
                removed_count += 1
        except (json.JSONDecodeError, OSError):
            # Remove corrupted files
            cache_file.unlink()
            removed_count += 1
    
    return removed_count

def _enforce_cache_size_limit(max_entries=100):
    """Enforce LRU cache size limit by removing oldest entries."""
    cache_dir = _get_cache_dir()
    responses_dir = cache_dir / "responses"
    
    if not responses_dir.exists():
        return 0
    
    # Get all cache files with their access times
    cache_files = []
    for cache_file in responses_dir.glob("*.json"):
        try:
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            timestamp = cache_data.get('timestamp', 0)
            cache_files.append((cache_file, timestamp))
        except (json.JSONDecodeError, OSError):
            # Remove corrupted files immediately
            cache_file.unlink()
    
    # If we're under the limit, no cleanup needed
    if len(cache_files) <= max_entries:
        return 0
    
    # Sort by timestamp (oldest first) and remove excess entries
    cache_files.sort(key=lambda x: x[1])
    excess_count = len(cache_files) - max_entries
    removed_count = 0
    
    for cache_file, _ in cache_files[:excess_count]:
        try:
            cache_file.unlink()
            removed_count += 1
        except OSError:
            pass
    
    return removed_count

def cleanup_query_cache():
    """Perform comprehensive query cache cleanup."""
    expired_removed = _cleanup_expired_cache_entries()
    lru_removed = _enforce_cache_size_limit()
    
    return {
        'expired_removed': expired_removed,
        'lru_removed': lru_removed,
        'total_removed': expired_removed + lru_removed
    }