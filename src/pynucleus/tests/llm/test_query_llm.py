"""
Comprehensive tests for the LLM Query utility with template rendering and token management.
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.pynucleus.llm.query_llm import LLMQueryManager, quick_ask_llm
from src.pynucleus.llm.llm_runner import LLMRunner
from src.pynucleus.utils.token_utils import TokenCounter


class TestLLMQueryManager:
    """Test cases for LLMQueryManager class."""
    
    @pytest.fixture
    def temp_template_dir(self):
        """Create a temporary directory with test templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)
            
            # Create test template
            template_content = """You are a helpful assistant.

{% if system_message %}
{{ system_message }}
{% endif %}

{% if report_content %}
**Report:**
{{ report_content }}
{% endif %}

**Query:** {{ user_query }}

**Response:**"""
            
            template_file = template_dir / "test_template.j2"
            template_file.write_text(template_content)
            
            yield str(template_dir)
    
    @pytest.fixture
    def temp_report_file(self):
        """Create a temporary report file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            report_content = """This is a test report with important information.
            
It contains multiple paragraphs and sentences. This helps test the token management
functionality of the LLM query system. The content should be long enough to
potentially trigger truncation when token limits are set low.

Section 1: Introduction
This section introduces the topic and provides background information.

Section 2: Methodology
This section describes the methods used in the analysis.

Section 3: Results
This section presents the findings of the analysis.

Section 4: Conclusion
This section summarizes the key points and provides recommendations."""
            
            f.write(report_content)
            f.flush()
            
            yield f.name
        
        # Cleanup
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass
    
    @pytest.fixture
    def mock_llm_runner(self):
        """Mock LLMRunner for testing without actual model loading."""
        with patch('src.pynucleus.llm.query_llm.LLMRunner') as mock_runner_class:
            mock_runner = MagicMock()
            mock_runner.ask.return_value = "This is a mock LLM response."
            mock_runner.get_model_info.return_value = {
                'model_id': 'gpt2',
                'device': 'cpu',
                'vocab_size': 50257
            }
            mock_runner_class.return_value = mock_runner
            yield mock_runner
    
    @pytest.fixture
    def mock_token_counter(self):
        """Mock TokenCounter for predictable token counting."""
        with patch('src.pynucleus.llm.query_llm.TokenCounter') as mock_counter_class:
            mock_counter = MagicMock()
            # Mock token counting: roughly 4 characters per token
            mock_counter.count_tokens.side_effect = lambda text: len(text) // 4 if text else 0
            mock_counter_class.return_value = mock_counter
            yield mock_counter
    
    def test_init_with_defaults(self, temp_template_dir, mock_llm_runner, mock_token_counter):
        """Test LLMQueryManager initialization with default parameters."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        assert manager.model_id == "gpt2"
        assert manager.device == "cpu"
        assert manager.max_tokens == 8192
        assert manager.template_dir == Path(temp_template_dir)
        assert manager.jinja_env is not None
    
    def test_init_with_custom_params(self, temp_template_dir, mock_llm_runner, mock_token_counter):
        """Test LLMQueryManager initialization with custom parameters."""
        manager = LLMQueryManager(
            model_id="custom-model",
            device="cuda",
            template_dir=temp_template_dir,
            max_tokens=4096
        )
        
        assert manager.model_id == "custom-model"
        assert manager.device == "cuda"
        assert manager.max_tokens == 4096
    
    def test_load_report_from_file(self, temp_template_dir, temp_report_file, 
                                 mock_llm_runner, mock_token_counter):
        """Test loading report content from file."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        content = manager.load_report_from_file(temp_report_file)
        
        assert isinstance(content, str)
        assert len(content) > 0
        assert "test report" in content.lower()
    
    def test_load_nonexistent_file(self, temp_template_dir, mock_llm_runner, mock_token_counter):
        """Test loading from non-existent file raises error."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        with pytest.raises(FileNotFoundError):
            manager.load_report_from_file("nonexistent_file.txt")
    
    def test_render_prompt(self, temp_template_dir, mock_llm_runner, mock_token_counter):
        """Test prompt rendering with Jinja2 template."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        rendered = manager.render_prompt(
            template_name="test_template.j2",
            user_query="What is the main topic?",
            system_message="Be helpful",
            report_content="Sample report content"
        )
        
        assert "What is the main topic?" in rendered
        assert "Be helpful" in rendered
        assert "Sample report content" in rendered
    
    def test_render_prompt_missing_template(self, temp_template_dir, 
                                          mock_llm_runner, mock_token_counter):
        """Test that missing template raises appropriate error."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        with pytest.raises(Exception):  # TemplateNotFound
            manager.render_prompt(
                template_name="nonexistent_template.j2",
                user_query="Test query"
            )
    
    def test_truncate_content_end_strategy(self, temp_template_dir, 
                                         mock_llm_runner, mock_token_counter):
        """Test content truncation with 'end' strategy."""
        manager = LLMQueryManager(template_dir=temp_template_dir, max_tokens=100)  # Very low limit
        
        long_content = "This is a very long content. " * 100  # ~2800 chars
        reserved_tokens = 80  # Reserve 80 tokens, leaving only 20 for content
        
        truncated = manager.truncate_content_for_tokens(
            long_content, 
            reserved_tokens, 
            "end"
        )
        
        assert len(truncated) < len(long_content)
        assert truncated.endswith("[Content truncated due to length...]")
    
    def test_truncate_content_start_strategy(self, temp_template_dir, 
                                           mock_llm_runner, mock_token_counter):
        """Test content truncation with 'start' strategy."""
        manager = LLMQueryManager(template_dir=temp_template_dir, max_tokens=100)  # Very low limit
        
        long_content = "This is a very long content. " * 100
        reserved_tokens = 80  # Reserve 80 tokens, leaving only 20 for content
        
        truncated = manager.truncate_content_for_tokens(
            long_content, 
            reserved_tokens, 
            "start"
        )
        
        assert len(truncated) < len(long_content)
        assert truncated.startswith("[Content truncated from beginning...]")
    
    def test_truncate_content_middle_strategy(self, temp_template_dir, 
                                            mock_llm_runner, mock_token_counter):
        """Test content truncation with 'middle' strategy."""
        manager = LLMQueryManager(template_dir=temp_template_dir, max_tokens=100)  # Very low limit
        
        long_content = "Start content. " + "Middle content. " * 100 + "End content."
        reserved_tokens = 80  # Reserve 80 tokens, leaving only 20 for content
        
        truncated = manager.truncate_content_for_tokens(
            long_content, 
            reserved_tokens, 
            "middle"
        )
        
        assert len(truncated) < len(long_content)
        assert "[... middle content truncated ...]" in truncated
        assert "Start content." in truncated
        assert "End content." in truncated
    
    def test_truncate_content_no_truncation_needed(self, temp_template_dir, 
                                                  mock_llm_runner, mock_token_counter):
        """Test that short content is not truncated."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        short_content = "This is short content."
        reserved_tokens = 100  # Much larger than content needs
        
        result = manager.truncate_content_for_tokens(
            short_content, 
            reserved_tokens, 
            "end"
        )
        
        assert result == short_content  # No truncation
    
    def test_ask_llm_with_file(self, temp_template_dir, temp_report_file, 
                              mock_llm_runner, mock_token_counter):
        """Test asking LLM with report file."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        response = manager.ask_llm(
            user_query="What is the main topic?",
            report_file_path=temp_report_file,
            template_name="test_template.j2"
        )
        
        assert response == "This is a mock LLM response."
        mock_llm_runner.ask.assert_called_once()
    
    def test_ask_llm_with_content(self, temp_template_dir, mock_llm_runner, mock_token_counter):
        """Test asking LLM with direct report content."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        response = manager.ask_llm(
            user_query="What is the main topic?",
            report_content="This is direct report content.",
            template_name="test_template.j2"
        )
        
        assert response == "This is a mock LLM response."
        mock_llm_runner.ask.assert_called_once()
    
    def test_ask_llm_with_system_message(self, temp_template_dir, 
                                       mock_llm_runner, mock_token_counter):
        """Test asking LLM with custom system message."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        response = manager.ask_llm(
            user_query="What is the main topic?",
            system_message="Be very detailed in your response.",
            template_name="test_template.j2"
        )
        
        assert response == "This is a mock LLM response."
        mock_llm_runner.ask.assert_called_once()
    
    def test_ask_llm_with_generation_params(self, temp_template_dir, 
                                          mock_llm_runner, mock_token_counter):
        """Test asking LLM with custom generation parameters."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        custom_params = {
            'temperature': 0.5,
            'max_length': 200,
            'top_p': 0.8
        }
        
        response = manager.ask_llm(
            user_query="Test query",
            generation_params=custom_params,
            template_name="test_template.j2"
        )
        
        assert response == "This is a mock LLM response."
        
        # Verify custom parameters were passed
        call_args = mock_llm_runner.ask.call_args
        assert call_args[1]['temperature'] == 0.5
        assert call_args[1]['top_p'] == 0.8
    
    def test_get_model_info(self, temp_template_dir, mock_llm_runner, mock_token_counter):
        """Test getting model information."""
        manager = LLMQueryManager(template_dir=temp_template_dir)
        
        info = manager.get_model_info()
        
        assert 'model_id' in info
        assert 'device' in info
        assert 'max_tokens' in info
        assert 'template_dir' in info
        assert 'llm_info' in info
        assert info['model_id'] == 'gpt2'
        assert info['max_tokens'] == 8192


class TestQuickAskLLM:
    """Test cases for the convenience function."""
    
    @pytest.fixture
    def temp_report_file(self):
        """Create a temporary report file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a quick test report.")
            f.flush()
            yield f.name
        
        try:
            os.unlink(f.name)
        except FileNotFoundError:
            pass
    
    @patch('src.pynucleus.llm.query_llm.LLMQueryManager')
    def test_quick_ask_llm(self, mock_manager_class, temp_report_file):
        """Test the convenience function for quick LLM querying."""
        # Mock the manager instance
        mock_manager = MagicMock()
        mock_manager.ask_llm.return_value = "Quick response"
        mock_manager_class.return_value = mock_manager
        
        response = quick_ask_llm(
            user_query="Quick question?",
            report_file_path=temp_report_file,
            model_id="test-model",
            max_tokens=4096
        )
        
        assert response == "Quick response"
        mock_manager_class.assert_called_once_with(
            model_id="test-model", 
            max_tokens=4096
        )
        mock_manager.ask_llm.assert_called_once()


class TestTokenManagement:
    """Test cases specifically for token management functionality."""
    
    @pytest.fixture
    def temp_template_dir(self):
        """Create a temporary directory with test templates."""
        with tempfile.TemporaryDirectory() as temp_dir:
            template_dir = Path(temp_dir)
            template_content = "{{ user_query }}{% if report_content %}\n{{ report_content }}{% endif %}"
            template_file = template_dir / "simple.j2"
            template_file.write_text(template_content)
            yield str(template_dir)
    
    @patch('src.pynucleus.llm.query_llm.LLMRunner')
    @patch('src.pynucleus.llm.query_llm.TokenCounter')
    def test_token_limit_enforcement(self, mock_counter_class, mock_runner_class, temp_template_dir):
        """Test that token limits are properly enforced."""
        # Setup mocks
        mock_counter = MagicMock()
        mock_counter.count_tokens.side_effect = lambda text: len(text) // 4
        mock_counter_class.return_value = mock_counter
        
        mock_runner = MagicMock()
        mock_runner.ask.return_value = "Response"
        mock_runner_class.return_value = mock_runner
        
        # Test with very low token limit
        manager = LLMQueryManager(
            template_dir=temp_template_dir,
            max_tokens=100  # Very low limit
        )
        
        long_content = "This is very long content. " * 50  # ~1400 chars
        
        response = manager.ask_llm(
            user_query="Question?",
            report_content=long_content,
            template_name="simple.j2"
        )
        
        # Verify that ask was called (meaning processing completed)
        mock_runner.ask.assert_called_once()
        
        # The prompt passed to LLM should be within token limits
        actual_prompt = mock_runner.ask.call_args[0][0]
        prompt_tokens = len(actual_prompt) // 4  # Our mock calculation
        
        # Should be well under the max_tokens limit
        assert prompt_tokens < 100


@pytest.fixture(scope="session")
def setup_template_directory():
    """Set up a test template directory for integration tests."""
    # This would create the actual prompts directory if needed
    project_root = Path(__file__).parent.parent.parent.parent.parent
    template_dir = project_root / "prompts"
    
    if not template_dir.exists():
        template_dir.mkdir(exist_ok=True)
        
        # Create the default template if it doesn't exist
        default_template = template_dir / "qwen_prompt.j2"
        if not default_template.exists():
            template_content = """You are an AI assistant.

{% if system_message %}
{{ system_message }}
{% endif %}

{% if report_content %}
**Report:**
{{ report_content }}
{% endif %}

**Query:** {{ user_query }}

**Response:**"""
            default_template.write_text(template_content)
    
    return str(template_dir)


class TestIntegration:
    """Integration tests that use actual components (without loading heavy models)."""
    
    @pytest.mark.integration
    @patch('src.pynucleus.llm.query_llm.LLMRunner')
    def test_end_to_end_workflow(self, mock_runner_class, setup_template_directory):
        """Test complete workflow from template to LLM response."""
        # Mock LLM runner to avoid loading actual model
        mock_runner = MagicMock()
        mock_runner.ask.return_value = "Integration test response"
        mock_runner.get_model_info.return_value = {'model_id': 'gpt2'}
        mock_runner_class.return_value = mock_runner
        
        # Create manager with actual template directory
        manager = LLMQueryManager(template_dir=setup_template_directory)
        
        # Test the complete workflow
        response = manager.ask_llm(
            user_query="What are the key findings?",
            report_content="This is a test report with key findings about renewable energy.",
            system_message="Provide a concise summary."
        )
        
        assert response == "Integration test response"
        
        # Verify the prompt was properly constructed
        call_args = mock_runner.ask.call_args
        prompt = call_args[0][0]
        
        assert "What are the key findings?" in prompt
        assert "renewable energy" in prompt
        assert "Provide a concise summary." in prompt


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 