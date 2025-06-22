#!/usr/bin/env python3
"""
Unit tests for Smart Vector Store Management system.
Tests caching, fingerprinting, storage management, and performance optimization.
"""

import os
import sys
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.smart_vectorstore import SmartVectorStore
from backend.embedder import get_embedder
from langchain_core.documents import Document

class TestSmartVectorStore:
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Create temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.smart_vs = SmartVectorStore(base_dir=self.temp_dir)
    
    def teardown_method(self):
        """Clean up after each test."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_fingerprint_generation(self):
        """Test document fingerprinting produces consistent hashes."""
        content1 = "This is test document content for fingerprinting."
        content2 = "This is different test document content."
        model = "bge-small-en"
        chunk_params = {"max_chunk_size": 1000, "overlap_size": 200}
        
        # Same content should produce same fingerprint
        fp1a = self.smart_vs._generate_document_fingerprint(content1, model, chunk_params)
        fp1b = self.smart_vs._generate_document_fingerprint(content1, model, chunk_params)
        assert fp1a == fp1b, "Same content should produce identical fingerprints"
        
        # Different content should produce different fingerprints
        fp2 = self.smart_vs._generate_document_fingerprint(content2, model, chunk_params)
        assert fp1a != fp2, "Different content should produce different fingerprints"
        
        # Different model should produce different fingerprints
        fp3 = self.smart_vs._generate_document_fingerprint(content1, "bge-base-en", chunk_params)
        assert fp1a != fp3, "Different model should produce different fingerprints"
        
        # Different chunk params should produce different fingerprints
        fp4 = self.smart_vs._generate_document_fingerprint(content1, model, {"max_chunk_size": 500})
        assert fp1a != fp4, "Different chunk params should produce different fingerprints"
    
    def test_collection_name_generation(self):
        """Test collection name generation from fingerprint and model."""
        fingerprint = "abcd1234567890ef"
        model = "bge-small-en"
        
        collection_name = self.smart_vs._get_collection_name(fingerprint, model)
        
        assert collection_name.startswith("doc_"), "Collection name should start with 'doc_'"
        assert "abcd12345678" in collection_name, "Collection name should contain fingerprint prefix"
        assert "bge_small_en" in collection_name, "Collection name should contain sanitized model name"
    
    def test_directory_creation(self):
        """Test that necessary directories are created."""
        assert self.smart_vs.metadata_dir.exists(), "Metadata directory should exist"
        assert self.smart_vs.collections_dir.exists(), "Collections directory should exist"
        assert self.smart_vs.temp_dir.exists(), "Temp directory should exist"
    
    def test_cache_statistics_initialization(self):
        """Test cache statistics are properly initialized."""
        stats = self.smart_vs.get_cache_statistics()
        
        assert stats["cache_hits"] == 0, "Initial cache hits should be 0"
        assert stats["cache_misses"] == 0, "Initial cache misses should be 0"
        assert stats["total_requests"] == 0, "Initial total requests should be 0"
        assert stats["hit_rate_percent"] == 0.0, "Initial hit rate should be 0%"
        assert stats["storage_usage_gb"] >= 0, "Storage usage should be non-negative"
    
    def test_metadata_save_and_load(self):
        """Test saving and loading collection metadata."""
        collection_name = "test_collection"
        fingerprint = "test_fingerprint"
        embedding_model = "bge-small-en"
        document_info = {"filename": "test.pdf", "size": 1024}
        chunk_count = 50
        
        # Save metadata
        self.smart_vs._save_collection_metadata(
            collection_name, fingerprint, embedding_model, document_info, chunk_count
        )
        
        # Load metadata
        loaded_metadata = self.smart_vs._load_collection_metadata(collection_name)
        
        assert loaded_metadata is not None, "Metadata should be loaded successfully"
        assert loaded_metadata["fingerprint"] == fingerprint, "Fingerprint should match"
        assert loaded_metadata["embedding_model"] == embedding_model, "Model should match"
        assert loaded_metadata["document_info"] == document_info, "Document info should match"
        assert loaded_metadata["chunk_count"] == chunk_count, "Chunk count should match"
        assert "created_at" in loaded_metadata, "Created timestamp should exist"
        assert "last_accessed" in loaded_metadata, "Last accessed timestamp should exist"
    
    def test_access_time_update(self):
        """Test updating access time for collections."""
        collection_name = "test_collection"
        
        # Create initial metadata
        self.smart_vs._save_collection_metadata(
            collection_name, "fp", "model", {}, 10
        )
        
        original_metadata = self.smart_vs._load_collection_metadata(collection_name)
        original_access_time = original_metadata["last_accessed"]
        original_access_count = original_metadata["access_count"]
        
        # Update access time
        import time
        time.sleep(0.01)  # Small delay to ensure different timestamp
        self.smart_vs._update_access_time(collection_name)
        
        updated_metadata = self.smart_vs._load_collection_metadata(collection_name)
        
        assert updated_metadata["last_accessed"] != original_access_time, "Access time should be updated"
        assert updated_metadata["access_count"] == original_access_count + 1, "Access count should increment"
    
    def test_storage_usage_calculation(self):
        """Test storage usage calculation."""
        # Initially should be 0 or very small
        initial_usage = self.smart_vs._get_storage_usage()
        assert initial_usage >= 0, "Storage usage should be non-negative"
        
        # Create a test file to increase usage
        test_file = self.smart_vs.collections_dir / "test_collection" / "test_file.txt"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        
        test_content = "x" * 1024  # 1KB content
        with open(test_file, 'w') as f:
            f.write(test_content)
        
        new_usage = self.smart_vs._get_storage_usage()
        assert new_usage > initial_usage, "Storage usage should increase after adding file"
    
    @patch('backend.smart_vectorstore.Chroma')
    def test_cache_hit_scenario(self, mock_chroma_class):
        """Test cache hit scenario when collection already exists."""
        # Mock Chroma class
        mock_vectordb = Mock()
        mock_chroma_class.return_value = mock_vectordb
        
        # Mock embedder
        mock_embedder = Mock()
        
        # Test data
        document_content = "Test document for cache hit"
        embedding_model = "bge-small-en"
        chunk_params = {"max_chunk_size": 1000}
        chunks = [Document(page_content="Test chunk", metadata={})]
        
        # Generate fingerprint and collection name
        fingerprint = self.smart_vs._generate_document_fingerprint(
            document_content, embedding_model, chunk_params
        )
        collection_name = self.smart_vs._get_collection_name(fingerprint, embedding_model)
        
        # Create collection directory and metadata to simulate existing collection
        collection_path = self.smart_vs._get_collection_path(collection_name)
        collection_path.mkdir(parents=True, exist_ok=True)
        
        # Create a dummy file to make collection appear valid
        (collection_path / "dummy.txt").touch()
        
        self.smart_vs._save_collection_metadata(
            collection_name, fingerprint, embedding_model, {}, len(chunks)
        )
        
        # Call get_or_create_vectorstore - should hit cache
        result = self.smart_vs.get_or_create_vectorstore(
            mock_embedder, chunks, document_content, embedding_model, chunk_params
        )
        
        # Verify cache hit
        stats = self.smart_vs.get_cache_statistics()
        assert stats["cache_hits"] == 1, "Should have 1 cache hit"
        assert stats["cache_misses"] == 0, "Should have 0 cache misses"
        assert stats["total_requests"] == 1, "Should have 1 total request"
        assert stats["hit_rate_percent"] == 100.0, "Hit rate should be 100%"
        
        # Verify Chroma was called to load existing store (not create new)
        mock_chroma_class.assert_called_once()
        call_args = mock_chroma_class.call_args
        assert 'persist_directory' in call_args.kwargs
        assert call_args.kwargs['persist_directory'] == str(collection_path)
    
    @patch('backend.smart_vectorstore.Chroma')
    def test_cache_miss_scenario(self, mock_chroma_class):
        """Test cache miss scenario when collection doesn't exist."""
        # Mock Chroma.from_documents
        mock_vectordb = Mock()
        mock_chroma_class.from_documents.return_value = mock_vectordb
        
        # Mock embedder
        mock_embedder = Mock()
        
        # Test data
        document_content = "Test document for cache miss"
        embedding_model = "bge-small-en"
        chunk_params = {"max_chunk_size": 1000}
        chunks = [Document(page_content="Test chunk", metadata={})]
        
        # Call get_or_create_vectorstore - should miss cache
        result = self.smart_vs.get_or_create_vectorstore(
            mock_embedder, chunks, document_content, embedding_model, chunk_params
        )
        
        # Verify cache miss
        stats = self.smart_vs.get_cache_statistics()
        assert stats["cache_hits"] == 0, "Should have 0 cache hits"
        assert stats["cache_misses"] == 1, "Should have 1 cache miss"
        assert stats["total_requests"] == 1, "Should have 1 total request"
        assert stats["hit_rate_percent"] == 0.0, "Hit rate should be 0%"
        
        # Verify Chroma.from_documents was called to create new store
        mock_chroma_class.from_documents.assert_called_once()
    
    def test_collection_listing(self):
        """Test listing cached collections."""
        # Initially should be empty
        collections = self.smart_vs.list_collections()
        assert len(collections) == 0, "Should have no collections initially"
        
        # Create some test metadata
        for i in range(3):
            collection_name = f"test_collection_{i}"
            self.smart_vs._save_collection_metadata(
                collection_name, f"fingerprint_{i}", "bge-small-en", 
                {"test": True}, 10 + i
            )
            
            # Create dummy collection directory
            collection_path = self.smart_vs._get_collection_path(collection_name)
            collection_path.mkdir(parents=True, exist_ok=True)
            (collection_path / "dummy.txt").touch()
        
        # Now should have 3 collections
        collections = self.smart_vs.list_collections()
        assert len(collections) == 3, "Should have 3 collections"
        
        # Verify collections are sorted by last accessed (most recent first)
        for i, collection in enumerate(collections):
            assert collection["collection_name"] == f"test_collection_{2-i}", \
                "Collections should be sorted by last accessed time"

def run_tests():
    """Run all tests."""
    test_instance = TestSmartVectorStore()
    
    test_methods = [
        method for method in dir(test_instance) 
        if method.startswith('test_') and callable(getattr(test_instance, method))
    ]
    
    passed = 0
    failed = 0
    
    for test_method in test_methods:
        try:
            print(f"Running {test_method}...")
            test_instance.setup_method()
            getattr(test_instance, test_method)()
            test_instance.teardown_method()
            print(f"✅ {test_method} PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ {test_method} FAILED: {str(e)}")
            failed += 1
            test_instance.teardown_method()
    
    print(f"\n📊 Test Results: {passed} passed, {failed} failed")
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)