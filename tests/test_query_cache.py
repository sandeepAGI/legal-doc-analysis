#!/usr/bin/env python3
"""
Unit tests for query response caching functionality.
Tests the caching infrastructure in llm_wrapper.py
"""

import unittest
import tempfile
import shutil
import time
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import the functions to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backend.llm_wrapper import (
    _generate_query_cache_key, 
    _load_from_cache, 
    _save_to_cache,
    _get_cache_dir,
    _update_cache_stats,
    get_query_cache_stats,
    synthesize_answer_cached,
    synthesize_answer_stream_cached,
    cleanup_query_cache
)

class TestQueryCacheInfrastructure(unittest.TestCase):
    """Test core caching infrastructure functions."""
    
    def setUp(self):
        """Set up test environment with temporary cache directory."""
        self.test_dir = tempfile.mkdtemp()
        self.original_cache_dir = None
        
        # Mock the cache directory to use our temp directory
        def mock_get_cache_dir():
            cache_dir = Path(self.test_dir) / "query_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "responses").mkdir(exist_ok=True)
            return cache_dir
        
        self.mock_get_cache_dir_patcher = patch('backend.llm_wrapper._get_cache_dir', side_effect=mock_get_cache_dir)
        self.mock_get_cache_dir_patcher.start()
    
    def tearDown(self):
        """Clean up test environment."""
        self.mock_get_cache_dir_patcher.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_cache_key_generation(self):
        """Test cache key generation with different inputs."""
        # Mock document chunks
        mock_doc1 = Mock()
        mock_doc1.page_content = "This is test content for document 1"
        mock_doc2 = Mock()
        mock_doc2.page_content = "This is test content for document 2"
        
        chunks1 = [(mock_doc1, 0.8), (mock_doc2, 0.7)]
        chunks2 = [(mock_doc2, 0.7), (mock_doc1, 0.8)]  # Different order
        chunks3 = [(mock_doc1, 0.8)]  # Different content
        
        query1 = "What is the main argument?"
        query2 = "What are the key points?"
        
        # Same query and chunks should produce same key
        key1 = _generate_query_cache_key(query1, chunks1)
        key1_repeat = _generate_query_cache_key(query1, chunks1)
        self.assertEqual(key1, key1_repeat)
        
        # Different query should produce different key
        key2 = _generate_query_cache_key(query2, chunks1)
        self.assertNotEqual(key1, key2)
        
        # Different chunk order should produce different key (context order matters)
        key3 = _generate_query_cache_key(query1, chunks2)
        self.assertNotEqual(key1, key3)
        
        # Different chunks should produce different key
        key4 = _generate_query_cache_key(query1, chunks3)
        self.assertNotEqual(key1, key4)
        
        # All keys should be valid MD5 hashes (32 hex characters)
        for key in [key1, key2, key3, key4]:
            self.assertEqual(len(key), 32)
            self.assertTrue(all(c in '0123456789abcdef' for c in key))
    
    def test_cache_save_and_load(self):
        """Test saving and loading cache entries."""
        cache_key = "test_key_123"
        test_response = "This is a test response from the LLM."
        
        # Initially no cache should exist
        loaded_response = _load_from_cache(cache_key)
        self.assertIsNone(loaded_response)
        
        # Save to cache
        _save_to_cache(cache_key, test_response)
        
        # Load from cache
        loaded_response = _load_from_cache(cache_key)
        self.assertEqual(loaded_response, test_response)
        
        # Verify cache file structure
        cache_dir = Path(self.test_dir) / "query_cache"
        cache_file = cache_dir / "responses" / f"{cache_key}.json"
        self.assertTrue(cache_file.exists())
        
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)
        
        self.assertEqual(cache_data['response'], test_response)
        self.assertEqual(cache_data['model'], 'llama3')
        self.assertIsInstance(cache_data['timestamp'], (int, float))
        self.assertGreater(cache_data['timestamp'], 0)
    
    def test_cache_expiration(self):
        """Test cache expiration (24 hour TTL)."""
        cache_key = "expired_test_key"
        test_response = "This response should expire."
        
        # Save cache with old timestamp
        cache_dir = Path(self.test_dir) / "query_cache"
        responses_dir = cache_dir / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)
        cache_file = responses_dir / f"{cache_key}.json"
        
        old_timestamp = time.time() - 86401  # 24 hours + 1 second ago
        cache_data = {
            'response': test_response,
            'timestamp': old_timestamp,
            'model': 'llama3'
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
        
        # Load should return None and remove expired file
        loaded_response = _load_from_cache(cache_key)
        self.assertIsNone(loaded_response)
        self.assertFalse(cache_file.exists())
    
    def test_corrupted_cache_handling(self):
        """Test handling of corrupted cache files."""
        cache_key = "corrupted_test_key"
        
        # Create corrupted cache file
        cache_dir = Path(self.test_dir) / "query_cache"
        responses_dir = cache_dir / "responses"
        responses_dir.mkdir(parents=True, exist_ok=True)
        cache_file = responses_dir / f"{cache_key}.json"
        
        with open(cache_file, 'w') as f:
            f.write("invalid json content")
        
        # Load should return None and remove corrupted file
        loaded_response = _load_from_cache(cache_key)
        self.assertIsNone(loaded_response)
        self.assertFalse(cache_file.exists())
    
    def test_cache_statistics_tracking(self):
        """Test cache statistics tracking."""
        # Initial stats should be zero
        stats = get_query_cache_stats()
        expected_initial = {'hits': 0, 'misses': 0, 'total_queries': 0, 'hit_rate': 0.0}
        self.assertEqual(stats, expected_initial)
        
        # Record a cache miss
        _update_cache_stats(cache_hit=False)
        stats = get_query_cache_stats()
        self.assertEqual(stats['hits'], 0)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['total_queries'], 1)
        self.assertEqual(stats['hit_rate'], 0.0)
        
        # Record a cache hit
        _update_cache_stats(cache_hit=True)
        stats = get_query_cache_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['total_queries'], 2)
        self.assertEqual(stats['hit_rate'], 0.5)
        
        # Record another hit
        _update_cache_stats(cache_hit=True)
        stats = get_query_cache_stats()
        self.assertEqual(stats['hits'], 2)
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['total_queries'], 3)
        self.assertAlmostEqual(stats['hit_rate'], 2/3, places=4)


class TestCachedFunctions(unittest.TestCase):
    """Test cached wrapper functions."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Mock the cache directory
        def mock_get_cache_dir():
            cache_dir = Path(self.test_dir) / "query_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "responses").mkdir(exist_ok=True)
            return cache_dir
        
        self.mock_get_cache_dir_patcher = patch('backend.llm_wrapper._get_cache_dir', side_effect=mock_get_cache_dir)
        self.mock_get_cache_dir_patcher.start()
        
        # Mock document chunks
        self.mock_doc = Mock()
        self.mock_doc.page_content = "Test document content for caching tests."
        self.test_chunks = [(self.mock_doc, 0.9)]
        self.test_query = "What is the main point?"
        self.test_response = "The main point is that caching works correctly."
    
    def tearDown(self):
        """Clean up test environment."""
        self.mock_get_cache_dir_patcher.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    @patch('backend.llm_wrapper.synthesize_answer')
    def test_synthesize_answer_cached_miss_then_hit(self, mock_synthesize):
        """Test cached function behavior on cache miss then hit."""
        mock_synthesize.return_value = self.test_response
        
        # First call should be cache miss
        result1 = synthesize_answer_cached(self.test_query, self.test_chunks)
        self.assertEqual(result1, self.test_response)
        mock_synthesize.assert_called_once()
        
        # Second call should be cache hit
        mock_synthesize.reset_mock()
        result2 = synthesize_answer_cached(self.test_query, self.test_chunks)
        self.assertEqual(result2, self.test_response)
        mock_synthesize.assert_not_called()  # Should not call original function
        
        # Verify cache statistics
        stats = get_query_cache_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
    
    @patch('backend.llm_wrapper.synthesize_answer')
    def test_synthesize_answer_cached_disabled(self, mock_synthesize):
        """Test cached function with caching disabled."""
        mock_synthesize.return_value = self.test_response
        
        # Call with use_cache=False
        result = synthesize_answer_cached(self.test_query, self.test_chunks, use_cache=False)
        self.assertEqual(result, self.test_response)
        mock_synthesize.assert_called_once()
        
        # Second call should still call original function
        mock_synthesize.reset_mock()
        result2 = synthesize_answer_cached(self.test_query, self.test_chunks, use_cache=False)
        self.assertEqual(result2, self.test_response)
        mock_synthesize.assert_called_once()
    
    @patch('backend.llm_wrapper.synthesize_answer_stream')
    def test_synthesize_answer_stream_cached_miss_then_hit(self, mock_stream):
        """Test cached streaming function behavior."""
        # Mock streaming response
        response_chunks = ["The main ", "point is ", "that caching ", "works correctly."]
        mock_stream.return_value = iter(response_chunks)
        
        # First call should be cache miss - collect streamed response
        result_chunks1 = list(synthesize_answer_stream_cached(self.test_query, self.test_chunks))
        full_response1 = "".join(result_chunks1)
        self.assertEqual(full_response1, self.test_response)
        mock_stream.assert_called_once()
        
        # Second call should be cache hit - simulate streaming from cache
        mock_stream.reset_mock()
        result_chunks2 = list(synthesize_answer_stream_cached(self.test_query, self.test_chunks))
        full_response2 = "".join(result_chunks2).strip()  # Strip trailing spaces from simulation
        self.assertEqual(full_response2, self.test_response)
        mock_stream.assert_not_called()  # Should not call original function
        
        # Verify cache statistics
        stats = get_query_cache_stats()
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)
    
    @patch('backend.llm_wrapper.synthesize_answer_stream')
    def test_synthesize_answer_stream_cached_disabled(self, mock_stream):
        """Test cached streaming function with caching disabled."""
        response_chunks = ["Test ", "streaming ", "response."]
        mock_stream.return_value = iter(response_chunks)
        
        # Call with use_cache=False
        result_chunks = list(synthesize_answer_stream_cached(self.test_query, self.test_chunks, use_cache=False))
        self.assertEqual("".join(result_chunks), "Test streaming response.")
        mock_stream.assert_called_once()


class TestCacheCleanup(unittest.TestCase):
    """Test cache cleanup and LRU management."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        def mock_get_cache_dir():
            cache_dir = Path(self.test_dir) / "query_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "responses").mkdir(exist_ok=True)
            return cache_dir
        
        self.mock_get_cache_dir_patcher = patch('backend.llm_wrapper._get_cache_dir', side_effect=mock_get_cache_dir)
        self.mock_get_cache_dir_patcher.start()
        
        self.cache_dir = Path(self.test_dir) / "query_cache"
        self.responses_dir = self.cache_dir / "responses"
    
    def tearDown(self):
        """Clean up test environment."""
        self.mock_get_cache_dir_patcher.stop()
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _create_cache_entry(self, key, response, timestamp=None):
        """Helper to create cache entries with specific timestamps."""
        if timestamp is None:
            timestamp = time.time()
        
        # Ensure directory exists
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            'response': response,
            'timestamp': timestamp,
            'model': 'llama3'
        }
        
        cache_file = self.responses_dir / f"{key}.json"
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f)
    
    def test_cleanup_expired_entries(self):
        """Test cleanup of expired cache entries."""
        current_time = time.time()
        
        # Create mix of fresh and expired entries
        self._create_cache_entry("fresh1", "Fresh response 1", current_time - 3600)  # 1 hour ago
        self._create_cache_entry("fresh2", "Fresh response 2", current_time - 7200)  # 2 hours ago
        self._create_cache_entry("expired1", "Expired response 1", current_time - 86401)  # 24+ hours ago
        self._create_cache_entry("expired2", "Expired response 2", current_time - 172800)  # 48 hours ago
        
        # Initial count
        initial_files = list(self.responses_dir.glob("*.json"))
        self.assertEqual(len(initial_files), 4)
        
        # Run cleanup
        cleanup_result = cleanup_query_cache()
        
        # Verify results
        self.assertEqual(cleanup_result['expired_removed'], 2)
        self.assertEqual(cleanup_result['lru_removed'], 0)
        self.assertEqual(cleanup_result['total_removed'], 2)
        
        # Verify only fresh entries remain
        remaining_files = list(self.responses_dir.glob("*.json"))
        self.assertEqual(len(remaining_files), 2)
        
        remaining_keys = [f.stem for f in remaining_files]
        self.assertIn("fresh1", remaining_keys)
        self.assertIn("fresh2", remaining_keys)
    
    def test_cleanup_lru_limit(self):
        """Test LRU cleanup when cache size exceeds limit."""
        current_time = time.time()
        
        # Create 5 entries with different timestamps
        for i in range(5):
            timestamp = current_time - (i * 3600)  # Each entry 1 hour older
            self._create_cache_entry(f"entry_{i}", f"Response {i}", timestamp)
        
        # Verify all 5 entries exist
        initial_files = list(self.responses_dir.glob("*.json"))
        self.assertEqual(len(initial_files), 5)
        
        # Test the LRU function directly with a small limit
        from backend.llm_wrapper import _enforce_cache_size_limit
        removed_count = _enforce_cache_size_limit(max_entries=3)
        
        # Should have removed 2 entries (5 - 3 = 2)
        self.assertEqual(removed_count, 2)
        
        # Verify only 3 entries remain
        remaining_files = list(self.responses_dir.glob("*.json"))
        self.assertEqual(len(remaining_files), 3)
        
        # Verify the most recent entries are kept (entry_0, entry_1, entry_2)
        remaining_keys = [f.stem for f in remaining_files]
        self.assertIn("entry_0", remaining_keys)  # Most recent
        self.assertIn("entry_1", remaining_keys)
        self.assertIn("entry_2", remaining_keys)
        self.assertNotIn("entry_3", remaining_keys)  # Should be removed
        self.assertNotIn("entry_4", remaining_keys)  # Should be removed
    
    def test_cleanup_corrupted_files(self):
        """Test cleanup of corrupted cache files."""
        # Create valid entry
        self._create_cache_entry("valid", "Valid response")
        
        # Create corrupted entry
        corrupted_file = self.responses_dir / "corrupted.json"
        with open(corrupted_file, 'w') as f:
            f.write("invalid json content")
        
        # Initial count
        initial_files = list(self.responses_dir.glob("*.json"))
        self.assertEqual(len(initial_files), 2)
        
        # Run cleanup
        cleanup_result = cleanup_query_cache()
        
        # Corrupted file should be removed during cleanup
        remaining_files = list(self.responses_dir.glob("*.json"))
        self.assertEqual(len(remaining_files), 1)
        
        remaining_keys = [f.stem for f in remaining_files]
        self.assertIn("valid", remaining_keys)
        self.assertNotIn("corrupted", remaining_keys)


if __name__ == "__main__":
    unittest.main()