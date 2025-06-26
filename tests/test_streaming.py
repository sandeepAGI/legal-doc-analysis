"""
Unit tests for LLM streaming functionality.
Tests both streaming and non-streaming responses to ensure compatibility.
"""

import unittest
from unittest.mock import Mock, patch
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from backend.llm_wrapper import synthesize_answer, synthesize_answer_stream, get_llm


class TestLLMStreaming(unittest.TestCase):
    """Test suite for LLM streaming functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock document chunks for testing
        self.mock_chunks = [
            (Mock(page_content="This is test content from chunk 1."), 0.95),
            (Mock(page_content="This is additional content from chunk 2."), 0.87)
        ]
        self.test_query = "What is the main topic discussed?"
        
        # Mock LLM responses
        self.mock_full_response = "Based on the provided context, the main topic is testing."
        self.mock_streaming_chunks = ["Based ", "on the ", "provided ", "context, ", "the main ", "topic is ", "testing."]
    
    @patch('backend.llm_wrapper.get_llm')
    def test_synthesize_answer_original_functionality(self, mock_get_llm):
        """Test that original synthesize_answer still works correctly."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = self.mock_full_response
        mock_get_llm.return_value = mock_llm
        
        result = synthesize_answer(self.test_query, self.mock_chunks)
        
        self.assertEqual(result, self.mock_full_response)
        mock_llm.invoke.assert_called_once()
    
    @patch('backend.llm_wrapper.get_llm')
    def test_synthesize_answer_stream_yields_chunks(self, mock_get_llm):
        """Test that streaming function yields response chunks."""
        mock_llm = Mock()
        mock_llm.stream.return_value = iter(self.mock_streaming_chunks)
        mock_get_llm.return_value = mock_llm
        
        result_chunks = list(synthesize_answer_stream(self.test_query, self.mock_chunks))
        
        self.assertEqual(result_chunks, self.mock_streaming_chunks)
        mock_llm.stream.assert_called_once()
    
    @patch('backend.llm_wrapper.get_llm')
    def test_streaming_reassembled_matches_full_response(self, mock_get_llm):
        """Test that streaming chunks reassemble to full response."""
        mock_llm = Mock()
        mock_llm.stream.return_value = iter(self.mock_streaming_chunks)
        mock_get_llm.return_value = mock_llm
        
        streaming_result = "".join(synthesize_answer_stream(self.test_query, self.mock_chunks))
        expected_full = "".join(self.mock_streaming_chunks)
        
        self.assertEqual(streaming_result, expected_full)
    
    @patch('backend.llm_wrapper.get_llm')
    def test_custom_llm_parameter_streaming(self, mock_get_llm):
        """Test streaming with custom LLM parameter."""
        custom_llm = Mock()
        custom_llm.stream.return_value = iter(["Custom ", "LLM ", "response"])
        
        result_chunks = list(synthesize_answer_stream(
            self.test_query, self.mock_chunks, llm=custom_llm
        ))
        
        self.assertEqual(result_chunks, ["Custom ", "LLM ", "response"])
        # get_llm should not be called when custom LLM is provided
        mock_get_llm.assert_not_called()
    
    @patch('backend.llm_wrapper.get_llm')
    def test_custom_llm_parameter_non_streaming(self, mock_get_llm):
        """Test non-streaming with custom LLM parameter."""
        custom_llm = Mock()
        custom_llm.invoke.return_value = "Custom LLM full response"
        
        result = synthesize_answer(self.test_query, self.mock_chunks, llm=custom_llm)
        
        self.assertEqual(result, "Custom LLM full response")
        # get_llm should not be called when custom LLM is provided
        mock_get_llm.assert_not_called()
    
    def test_get_llm_configuration(self):
        """Test that get_llm returns properly configured Ollama instance."""
        with patch('backend.llm_wrapper.Ollama') as mock_ollama:
            mock_instance = Mock()
            mock_ollama.return_value = mock_instance
            
            result = get_llm()
            
            mock_ollama.assert_called_once_with(model="llama3", temperature=0.0)
            self.assertEqual(result, mock_instance)
    
    @patch('backend.llm_wrapper.get_llm')
    def test_context_formation_consistent(self, mock_get_llm):
        """Test that context formation is consistent between streaming and non-streaming."""
        mock_llm = Mock()
        mock_llm.invoke.return_value = self.mock_full_response
        mock_llm.stream.return_value = iter(self.mock_streaming_chunks)
        mock_get_llm.return_value = mock_llm
        
        # Call both functions
        synthesize_answer(self.test_query, self.mock_chunks)
        list(synthesize_answer_stream(self.test_query, self.mock_chunks))
        
        # Both should be called with the same formatted prompt
        invoke_call_args = mock_llm.invoke.call_args[0][0]
        stream_call_args = mock_llm.stream.call_args[0][0]
        
        self.assertEqual(invoke_call_args, stream_call_args)
        # Verify context contains both chunks
        self.assertIn("chunk 1", invoke_call_args)
        self.assertIn("chunk 2", invoke_call_args)
    
    @patch('backend.llm_wrapper.get_llm')
    def test_empty_chunks_handling(self, mock_get_llm):
        """Test handling of empty retrieved chunks."""
        mock_llm = Mock()
        mock_llm.stream.return_value = iter(["No context available"])
        mock_get_llm.return_value = mock_llm
        
        empty_chunks = []
        result_chunks = list(synthesize_answer_stream(self.test_query, empty_chunks))
        
        self.assertEqual(result_chunks, ["No context available"])
        # Verify the prompt was called with empty context
        call_args = mock_llm.stream.call_args[0][0]
        self.assertIn(self.test_query, call_args)
    
    @patch('backend.llm_wrapper.get_llm')
    def test_streaming_generator_behavior(self, mock_get_llm):
        """Test that streaming function returns a generator."""
        mock_llm = Mock()
        mock_llm.stream.return_value = iter(self.mock_streaming_chunks)
        mock_get_llm.return_value = mock_llm
        
        result = synthesize_answer_stream(self.test_query, self.mock_chunks)
        
        # Should be a generator
        self.assertTrue(hasattr(result, '__iter__'))
        self.assertTrue(hasattr(result, '__next__'))
        
        # Should yield expected chunks
        first_chunk = next(result)
        self.assertEqual(first_chunk, self.mock_streaming_chunks[0])


class TestStreamingIntegration(unittest.TestCase):
    """Integration tests for streaming with real LLM calls (mocked at transport level)."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.mock_chunks = [
            (Mock(page_content="Integration test content for streaming."), 0.90)
        ]
        self.test_query = "What is this about?"
    
    @patch('backend.llm_wrapper.Ollama')
    def test_end_to_end_streaming_flow(self, mock_ollama_class):
        """Test complete streaming flow from initialization to response."""
        # Mock Ollama instance
        mock_ollama_instance = Mock()
        mock_ollama_instance.stream.return_value = iter(["End-to-end ", "streaming ", "test"])
        mock_ollama_class.return_value = mock_ollama_instance
        
        # Test the complete flow
        chunks = list(synthesize_answer_stream(self.test_query, self.mock_chunks))
        
        # Verify results
        self.assertEqual(chunks, ["End-to-end ", "streaming ", "test"])
        mock_ollama_class.assert_called_with(model="llama3", temperature=0.0)
        mock_ollama_instance.stream.assert_called_once()
    
    @patch('backend.llm_wrapper.Ollama')
    def test_backward_compatibility_preserved(self, mock_ollama_class):
        """Test that original non-streaming functionality is preserved."""
        # Mock Ollama instance
        mock_ollama_instance = Mock()
        mock_ollama_instance.invoke.return_value = "Non-streaming response"
        mock_ollama_class.return_value = mock_ollama_instance
        
        # Test original function still works
        result = synthesize_answer(self.test_query, self.mock_chunks)
        
        # Verify results
        self.assertEqual(result, "Non-streaming response")
        mock_ollama_class.assert_called_with(model="llama3", temperature=0.0)
        mock_ollama_instance.invoke.assert_called_once()


if __name__ == '__main__':
    # Configure test runner for detailed output
    unittest.main(verbosity=2, buffer=True)