# backend/smart_vectorstore.py

import os
import json
import hashlib
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

class SmartVectorStore:
    """
    Smart Vector Store Management with document fingerprinting, intelligent caching,
    and automatic storage management for optimal latency.
    """
    
    def __init__(self, base_dir: str = "chroma_stores"):
        self.base_dir = Path(base_dir)
        self.metadata_dir = self.base_dir / "metadata"
        self.collections_dir = self.base_dir / "collections" 
        self.temp_dir = self.base_dir / "temp"
        
        # Storage management configuration
        self.max_collections_per_model = 10
        self.max_total_storage_gb = 1.0
        self.collection_ttl_days = 30
        
        # Ensure directories exist
        self._ensure_directories()
        
        # Load cache statistics
        self.cache_stats = self._load_cache_stats()
    
    def _ensure_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.metadata_dir, self.collections_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _load_cache_stats(self) -> Dict[str, Any]:
        """Load cache statistics from disk."""
        stats_file = self.metadata_dir / "cache_stats.json"
        if stats_file.exists():
            with open(stats_file, 'r') as f:
                return json.load(f)
        return {
            "cache_hits": 0,
            "cache_misses": 0,
            "total_requests": 0,
            "storage_cleanups": 0,
            "last_cleanup": None
        }
    
    def _save_cache_stats(self):
        """Save cache statistics to disk."""
        stats_file = self.metadata_dir / "cache_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.cache_stats, f, indent=2)
    
    def _generate_document_fingerprint(self, document_content: str, embedding_model: str, 
                                     chunk_params: Dict[str, Any]) -> str:
        """
        Generate unique fingerprint for document based on content, model, and chunking parameters.
        
        Args:
            document_content: Full text content of the document
            embedding_model: Name of the embedding model
            chunk_params: Dictionary of chunking parameters (max_size, overlap, etc.)
            
        Returns:
            MD5 hash string as unique identifier
        """
        # Create combined string for hashing
        chunk_params_str = json.dumps(chunk_params, sort_keys=True)
        combined = f"{document_content}|{embedding_model}|{chunk_params_str}"
        
        # Generate MD5 hash
        return hashlib.md5(combined.encode('utf-8')).hexdigest()
    
    def _get_collection_name(self, fingerprint: str, embedding_model: str) -> str:
        """Generate collection name from fingerprint and model."""
        model_safe = embedding_model.replace("-", "_").replace(" ", "_").replace("(", "").replace(")", "")
        return f"doc_{fingerprint[:12]}_{model_safe}"
    
    def _get_collection_path(self, collection_name: str) -> Path:
        """Get full path to collection directory."""
        return self.collections_dir / collection_name
    
    def _get_metadata_path(self, collection_name: str) -> Path:
        """Get path to collection metadata file."""
        return self.metadata_dir / f"{collection_name}.json"
    
    def _save_collection_metadata(self, collection_name: str, fingerprint: str,
                                embedding_model: str, document_info: Dict[str, Any],
                                chunk_count: int):
        """Save metadata for a collection."""
        metadata = {
            "fingerprint": fingerprint,
            "embedding_model": embedding_model,
            "document_info": document_info,
            "chunk_count": chunk_count,
            "created_at": datetime.now().isoformat(),
            "last_accessed": datetime.now().isoformat(),
            "access_count": 1,
            "collection_name": collection_name
        }
        
        metadata_path = self._get_metadata_path(collection_name)
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _load_collection_metadata(self, collection_name: str) -> Optional[Dict[str, Any]]:
        """Load metadata for a collection."""
        metadata_path = self._get_metadata_path(collection_name)
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def _update_access_time(self, collection_name: str):
        """Update last access time for a collection."""
        metadata = self._load_collection_metadata(collection_name)
        if metadata:
            metadata["last_accessed"] = datetime.now().isoformat()
            metadata["access_count"] = metadata.get("access_count", 0) + 1
            
            metadata_path = self._get_metadata_path(collection_name)
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def _check_collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists and is valid."""
        collection_path = self._get_collection_path(collection_name)
        metadata_path = self._get_metadata_path(collection_name)
        
        return (collection_path.exists() and 
                metadata_path.exists() and 
                len(list(collection_path.glob("*"))) > 0)
    
    def _get_storage_usage(self) -> float:
        """Get total storage usage in GB."""
        total_size = 0
        if self.collections_dir.exists():
            for collection_dir in self.collections_dir.iterdir():
                if collection_dir.is_dir():
                    for file_path in collection_dir.rglob("*"):
                        if file_path.is_file():
                            total_size += file_path.stat().st_size
        
        return total_size / (1024 ** 3)  # Convert to GB
    
    def _cleanup_storage(self, force: bool = False):
        """Clean up storage based on LRU and size limits."""
        current_time = datetime.now()
        
        # Check if cleanup is needed
        storage_usage = self._get_storage_usage()
        needs_cleanup = (storage_usage > self.max_total_storage_gb or 
                        force or 
                        not self.cache_stats.get("last_cleanup") or
                        datetime.fromisoformat(self.cache_stats["last_cleanup"]) < 
                        current_time - timedelta(days=1))
        
        if not needs_cleanup:
            return
        
        # Get all collections with metadata
        collections_info = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            if metadata_file.name != "cache_stats.json":
                collection_name = metadata_file.stem
                metadata = self._load_collection_metadata(collection_name)
                if metadata:
                    collections_info.append((collection_name, metadata))
        
        # Sort by last accessed time (oldest first)
        collections_info.sort(key=lambda x: x[1]["last_accessed"])
        
        # Remove collections based on TTL and count limits
        removed_count = 0
        ttl_cutoff = current_time - timedelta(days=self.collection_ttl_days)
        
        # Group by embedding model
        model_collections = {}
        for collection_name, metadata in collections_info:
            model = metadata["embedding_model"]
            if model not in model_collections:
                model_collections[model] = []
            model_collections[model].append((collection_name, metadata))
        
        # Remove old collections and enforce per-model limits
        for model, model_colls in model_collections.items():
            # Remove TTL-expired collections
            for collection_name, metadata in model_colls:
                last_accessed = datetime.fromisoformat(metadata["last_accessed"])
                if last_accessed < ttl_cutoff:
                    self._remove_collection(collection_name)
                    removed_count += 1
            
            # Remove excess collections beyond per-model limit
            active_collections = [(name, meta) for name, meta in model_colls 
                                if datetime.fromisoformat(meta["last_accessed"]) >= ttl_cutoff]
            
            if len(active_collections) > self.max_collections_per_model:
                # Sort by last accessed (oldest first) and remove excess
                active_collections.sort(key=lambda x: x[1]["last_accessed"])
                excess_count = len(active_collections) - self.max_collections_per_model
                
                for i in range(excess_count):
                    collection_name = active_collections[i][0]
                    self._remove_collection(collection_name)
                    removed_count += 1
        
        # Continue removing collections if still over storage limit
        while self._get_storage_usage() > self.max_total_storage_gb and collections_info:
            collection_name = collections_info.pop(0)[0]
            if self._check_collection_exists(collection_name):
                self._remove_collection(collection_name)
                removed_count += 1
        
        # Update cleanup statistics
        self.cache_stats["storage_cleanups"] += 1
        self.cache_stats["last_cleanup"] = current_time.isoformat()
        self._save_cache_stats()
        
        if removed_count > 0:
            print(f"Storage cleanup: removed {removed_count} collections")
    
    def _remove_collection(self, collection_name: str):
        """Remove a collection and its metadata."""
        # Remove collection directory
        collection_path = self._get_collection_path(collection_name)
        if collection_path.exists():
            shutil.rmtree(collection_path)
        
        # Remove metadata file
        metadata_path = self._get_metadata_path(collection_name)
        if metadata_path.exists():
            metadata_path.unlink()
    
    def get_or_create_vectorstore(self, embedder, chunks: List, document_content: str,
                                embedding_model: str, chunk_params: Dict[str, Any],
                                document_info: Optional[Dict[str, Any]] = None) -> Chroma:
        """
        Get existing vector store or create new one with smart caching.
        
        Args:
            embedder: Embedding model instance
            chunks: List of document chunks
            document_content: Full document text for fingerprinting
            embedding_model: Name of embedding model
            chunk_params: Chunking parameters used
            document_info: Optional document metadata (filename, size, etc.)
            
        Returns:
            Chroma vector store instance
        """
        # Update total requests counter
        self.cache_stats["total_requests"] += 1
        
        # Generate document fingerprint
        fingerprint = self._generate_document_fingerprint(
            document_content, embedding_model, chunk_params
        )
        
        # Generate collection name
        collection_name = self._get_collection_name(fingerprint, embedding_model)
        collection_path = self._get_collection_path(collection_name)
        
        # Check if collection exists (cache hit)
        if self._check_collection_exists(collection_name):
            print(f"Cache HIT: Reusing existing vector store for {collection_name[:20]}...")
            self.cache_stats["cache_hits"] += 1
            self._update_access_time(collection_name)
            self._save_cache_stats()
            
            # Load existing vector store
            vectordb = Chroma(
                persist_directory=str(collection_path),
                embedding_function=embedder
            )
            
            # Debug: Check if the collection has data
            try:
                collection_size = vectordb._collection.count()
                print(f"Cache HIT: Loaded collection with {collection_size} documents")
            except Exception as e:
                print(f"Cache HIT: Loaded collection (size check failed: {e})")
            
            return vectordb
        
        # Cache miss - create new vector store
        print(f"Cache MISS: Creating new vector store for {collection_name[:20]}...")
        self.cache_stats["cache_misses"] += 1
        
        # Clean up storage before creating new collection
        self._cleanup_storage()
        
        # Handle different input types for chunks
        docs = []
        for item in chunks:
            if isinstance(item, Document):
                docs.append(item)
            elif isinstance(item, tuple):
                text, metadata = item
                docs.append(Document(page_content=str(text), metadata=metadata))
            else:
                docs.append(Document(page_content=str(item), metadata={}))
        
        if not docs:
            raise ValueError("No valid chunks provided for vector store creation.")
        
        # Create vector store
        vectordb = Chroma.from_documents(
            docs, 
            embedding=embedder, 
            persist_directory=str(collection_path)
        )
        
        # Explicitly persist the data to disk
        vectordb.persist()
        print(f"Cache MISS: Created and persisted vector store with {len(docs)} documents")
        
        # Save collection metadata
        if document_info is None:
            document_info = {"chunks_processed": len(docs)}
        
        self._save_collection_metadata(
            collection_name, fingerprint, embedding_model, 
            document_info, len(docs)
        )
        
        self._save_cache_stats()
        
        return vectordb
    
    def query_vectorstore(self, vectordb: Chroma, query: str, k: int = 10):
        """Query the vector store with similarity search."""
        return vectordb.similarity_search_with_score(query, k=k)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats["total_requests"]
        if total_requests > 0:
            hit_rate = (self.cache_stats["cache_hits"] / total_requests) * 100
        else:
            hit_rate = 0.0
        
        return {
            "cache_hits": self.cache_stats["cache_hits"],
            "cache_misses": self.cache_stats["cache_misses"],
            "total_requests": total_requests,
            "hit_rate_percent": round(hit_rate, 2),
            "storage_usage_gb": round(self._get_storage_usage(), 3),
            "storage_cleanups": self.cache_stats["storage_cleanups"],
            "last_cleanup": self.cache_stats.get("last_cleanup")
        }
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all cached collections with their metadata."""
        collections = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            if metadata_file.name != "cache_stats.json":
                collection_name = metadata_file.stem
                metadata = self._load_collection_metadata(collection_name)
                if metadata and self._check_collection_exists(collection_name):
                    collections.append(metadata)
        
        # Sort by last accessed (most recent first)
        collections.sort(key=lambda x: x["last_accessed"], reverse=True)
        return collections
    
    def force_cleanup(self):
        """Force storage cleanup regardless of thresholds."""
        print("Forcing storage cleanup...")
        self._cleanup_storage(force=True)
        print("Storage cleanup completed.")