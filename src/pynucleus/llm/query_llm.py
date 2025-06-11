"""
LLM Query Utility with Template Rendering and Token Management.

This module provides utilities for querying LLMs with prompt template rendering
using Jinja2 and intelligent token management for handling text reports.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, Union
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

from .llm_runner import LLMRunner
from ..utils.token_utils import TokenCounter

# Set up logging
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_MAX_TOKENS = 8192
DEFAULT_TEMPLATE_DIR = "prompt_templates"
DEFAULT_TEMPLATE_NAME = "qwen_prompt.j2"


class LLMQueryManager:
    """
    Manages LLM queries with template rendering and token management.
    
    This class provides functionality to:
    - Render prompts using Jinja2 templates
    - Load and process text reports from files
    - Manage token limits by intelligently truncating content
    - Query LLMs using the LLMRunner class
    """
    
    def __init__(self, 
                 model_id: str = "gpt2",
                 device: str = "cpu",
                 template_dir: Optional[str] = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS):
        """
        Initialize the LLM Query Manager.
        
        Args:
            model_id (str): HuggingFace model identifier for LLM
            device (str): Device to run the model on ('cpu' or 'cuda')
            template_dir (str): Directory containing Jinja2 templates
            max_tokens (int): Maximum tokens for prompt (default: 8192)
        """
        self.model_id = model_id
        self.device = device
        self.max_tokens = max_tokens
        
        # Initialize LLM runner
        self.llm_runner = LLMRunner(model_id=model_id, device=device)
        
        # Initialize token counter
        self.token_counter = TokenCounter(model_id=model_id)
        
        # Set up template environment
        if template_dir is None:
            # Default to prompt_templates in project root
            project_root = Path(__file__).parent.parent.parent.parent
            template_dir = project_root / DEFAULT_TEMPLATE_DIR
        
        self.template_dir = Path(template_dir)
        self.jinja_env = None
        self._setup_template_environment()
        
        logger.info(f"LLMQueryManager initialized with model: {model_id}, max_tokens: {max_tokens}")
    
    def _setup_template_environment(self):
        """Set up the Jinja2 template environment."""
        try:
            if not self.template_dir.exists():
                raise FileNotFoundError(f"Template directory not found: {self.template_dir}")
            
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(self.template_dir)),
                trim_blocks=True,
                lstrip_blocks=True
            )
            logger.info(f"Template environment set up with directory: {self.template_dir}")
            
        except Exception as e:
            logger.error(f"Failed to setup template environment: {e}")
            raise
    
    def load_report_from_file(self, file_path: Union[str, Path]) -> str:
        """
        Load textual report content from a file.
        
        Args:
            file_path (Union[str, Path]): Path to the report file
            
        Returns:
            str: Content of the report file
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            IOError: If there's an error reading the file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Report file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            logger.info(f"Loaded report from {file_path} ({len(content)} characters)")
            return content
            
        except Exception as e:
            logger.error(f"Error reading report file {file_path}: {e}")
            raise IOError(f"Failed to read report file: {e}")
    
    def truncate_content_for_tokens(self, 
                                  content: str, 
                                  reserved_tokens: int,
                                  truncate_strategy: str = "end") -> str:
        """
        Intelligently truncate content to fit within token limits.
        
        Args:
            content (str): Content to truncate
            reserved_tokens (int): Tokens reserved for prompt template and response
            truncate_strategy (str): How to truncate ('end', 'middle', 'start')
            
        Returns:
            str: Truncated content that fits within token limits
        """
        available_tokens = self.max_tokens - reserved_tokens
        content_tokens = self.token_counter.count_tokens(content)
        
        if content_tokens <= available_tokens:
            logger.debug(f"Content fits within limits: {content_tokens}/{available_tokens} tokens")
            return content
        
        logger.info(f"Truncating content: {content_tokens} -> ~{available_tokens} tokens")
        
        # Estimate characters per token for truncation
        chars_per_token = len(content) / content_tokens
        target_chars = int(available_tokens * chars_per_token * 0.9)  # 90% safety margin
        
        if truncate_strategy == "end":
            truncated = content[:target_chars]
            # Try to end at a sentence boundary
            last_period = truncated.rfind('.')
            if last_period > target_chars * 0.8:  # If period is in last 20%
                truncated = truncated[:last_period + 1]
            truncated += "\n\n[Content truncated due to length...]"
            
        elif truncate_strategy == "start":
            truncated = content[-target_chars:]
            # Try to start at a sentence boundary
            first_period = truncated.find('.')
            if first_period < target_chars * 0.2:  # If period is in first 20%
                truncated = truncated[first_period + 1:].lstrip()
            truncated = "[Content truncated from beginning...]\n\n" + truncated
            
        elif truncate_strategy == "middle":
            start_chars = target_chars // 2
            end_chars = target_chars - start_chars
            start_part = content[:start_chars]
            end_part = content[-end_chars:]
            truncated = start_part + "\n\n[... middle content truncated ...]\n\n" + end_part
            
        else:
            raise ValueError(f"Invalid truncate_strategy: {truncate_strategy}")
        
        # Verify truncated content fits
        final_tokens = self.token_counter.count_tokens(truncated)
        logger.debug(f"Truncated content tokens: {final_tokens}")
        
        return truncated
    
    def render_prompt(self, 
                     template_name: str = DEFAULT_TEMPLATE_NAME,
                     **template_vars) -> str:
        """
        Render a prompt using a Jinja2 template.
        
        Args:
            template_name (str): Name of the template file
            **template_vars: Variables to pass to the template
            
        Returns:
            str: Rendered prompt
            
        Raises:
            TemplateNotFound: If the template file doesn't exist
            Exception: If there's an error rendering the template
        """
        try:
            template = self.jinja_env.get_template(template_name)
            rendered = template.render(**template_vars)
            
            logger.debug(f"Rendered prompt using template: {template_name}")
            return rendered
            
        except TemplateNotFound:
            logger.error(f"Template not found: {template_name}")
            raise
        except Exception as e:
            logger.error(f"Error rendering template {template_name}: {e}")
            raise
    
    def ask_llm(self, 
                user_query: str,
                report_file_path: Optional[Union[str, Path]] = None,
                report_content: Optional[str] = None,
                system_message: Optional[str] = None,
                template_name: str = DEFAULT_TEMPLATE_NAME,
                truncate_strategy: str = "end",
                generation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Query the LLM with prompt rendering and token management.
        
        Args:
            user_query (str): The user's question or query
            report_file_path (Optional[Union[str, Path]]): Path to report file to load
            report_content (Optional[str]): Direct report content (alternative to file)
            system_message (Optional[str]): System message for the prompt
            template_name (str): Jinja2 template to use
            truncate_strategy (str): How to truncate content if needed
            generation_params (Optional[Dict[str, Any]]): Parameters for text generation
            
        Returns:
            str: LLM response
            
        Raises:
            ValueError: If neither report_file_path nor report_content is provided when needed
            Exception: If there's an error during processing
        """
        # Load report content if file path is provided
        if report_file_path is not None:
            report_content = self.load_report_from_file(report_file_path)
        
        # Prepare template variables
        template_vars = {
            'user_query': user_query,
            'system_message': system_message,
            'report_content': report_content,
            'max_tokens': self.max_tokens
        }
        
        # Calculate tokens for base prompt (without report content)
        base_template_vars = template_vars.copy()
        base_template_vars['report_content'] = ""
        base_prompt = self.render_prompt(template_name, **base_template_vars)
        base_tokens = self.token_counter.count_tokens(base_prompt)
        
        # Reserve tokens for generation (estimate 30% of max tokens)
        generation_reserve = int(self.max_tokens * 0.3)
        reserved_tokens = base_tokens + generation_reserve
        
        logger.info(f"Base prompt tokens: {base_tokens}, Reserved for generation: {generation_reserve}")
        
        # Truncate report content if necessary
        if report_content:
            report_content = self.truncate_content_for_tokens(
                report_content, 
                reserved_tokens, 
                truncate_strategy
            )
            template_vars['report_content'] = report_content
        
        # Render final prompt
        final_prompt = self.render_prompt(template_name, **template_vars)
        final_tokens = self.token_counter.count_tokens(final_prompt)
        
        logger.info(f"Final prompt tokens: {final_tokens}/{self.max_tokens}")
        
        # Set default generation parameters
        default_params = {
            'max_length': min(final_tokens + 150, self.max_tokens),  # Leave room for response
            'temperature': 0.7,
            'do_sample': True,
            'top_p': 0.9,
            'top_k': 50
        }
        
        if generation_params:
            default_params.update(generation_params)
        
        # Query the LLM
        try:
            logger.info("Querying LLM...")
            response = self.llm_runner.ask(final_prompt, **default_params)
            logger.info(f"LLM response generated ({len(response)} characters)")
            return response
            
        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model and configuration.
        
        Returns:
            Dict[str, Any]: Model and configuration information
        """
        return {
            'model_id': self.model_id,
            'device': self.device,
            'max_tokens': self.max_tokens,
            'template_dir': str(self.template_dir),
            'llm_info': self.llm_runner.get_model_info()
        }


# Convenience function for quick querying
def quick_ask_llm(user_query: str,
                  report_file_path: Optional[Union[str, Path]] = None,
                  model_id: str = "gpt2",
                  max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """
    Convenience function for quick LLM querying.
    
    Args:
        user_query (str): The user's question
        report_file_path (Optional[Union[str, Path]]): Path to report file
        model_id (str): Model to use
        max_tokens (int): Maximum tokens for prompt
        
    Returns:
        str: LLM response
    """
    manager = LLMQueryManager(model_id=model_id, max_tokens=max_tokens)
    return manager.ask_llm(user_query, report_file_path=report_file_path) 