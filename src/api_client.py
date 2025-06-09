"""
API Client for External LLM Services

This module provides wrappers for external API services like OpenAI,
with caching functionality to avoid redundant API calls.
"""

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️  OpenAI module not available - using FREE local models only")

import json
import os
import hashlib
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

class OpenAIClient:
    """
    Wrapper for OpenAI API calls with caching and error handling.
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "data/cache"):
        """
        Initialize OpenAI client.
        
        Args:
            api_key: OpenAI API key (if None, tries to get from environment)
            cache_dir: Directory to store cache files
        """
        if not OPENAI_AVAILABLE:
            print("❌ OpenAI not available. Install with: pip install openai")
            print("💡 Recommendation: Use FREE local models instead!")
            self.client = None
            return
            
        # Set API key
        if api_key:
            self.client = openai.OpenAI(api_key=api_key)
        elif os.getenv("OPENAI_API_KEY"):
            self.client = openai.OpenAI()
        else:
            print("⚠️  No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            print("💡 Recommendation: Use FREE local models instead!")
            self.client = None
        
        # Setup cache
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "openai_cache.json"
        self.cache = self._load_cache()
        
        print(f"🔧 OpenAI client initialized with cache at {self.cache_file}")
    
    def _load_cache(self) -> Dict:
        """Load cache from JSON file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load cache: {e}")
        return {}
    
    def _save_cache(self):
        """Save cache to JSON file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"⚠️ Warning: Could not save cache: {e}")
    
    def _generate_cache_key(self, prompt: str, model: str, **kwargs) -> str:
        """Generate unique cache key for request."""
        # Create a string representation of the request
        request_str = json.dumps({
            'prompt': prompt,
            'model': model,
            **kwargs
        }, sort_keys=True)
        
        # Generate MD5 hash
        return hashlib.md5(request_str.encode()).hexdigest()
    
    def call_openai(self, prompt: str, model: str = 'gpt-3.5-turbo', 
                   max_tokens: int = 100, temperature: float = 0.0,
                   use_cache: bool = True, **kwargs) -> Dict[str, Any]:
        """
        Call OpenAI API with caching.
        
        Args:
            prompt: Input prompt
            model: Model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            use_cache: Whether to use cached responses
            **kwargs: Additional API parameters
            
        Returns:
            API response dictionary
        """
        if not OPENAI_AVAILABLE or self.client is None:
            print("❌ OpenAI not available - use FREE local models instead!")
            return {
                'error': "OpenAI not available",
                'choices': [{'text': '', 'message': {'content': ''}}]
            }
        
        # Generate cache key
        cache_key = self._generate_cache_key(prompt, model, max_tokens=max_tokens, 
                                           temperature=temperature, **kwargs)
        
        # Check cache first
        if use_cache and cache_key in self.cache:
            print(f"📋 Using cached response for prompt: {prompt[:50]}...")
            return self.cache[cache_key]
        
        try:
            # Make API call
            print(f"🌐 Calling OpenAI API for prompt: {prompt[:50]}...")
            
            if model.startswith('gpt-3.5') or model.startswith('gpt-4'):
                # Chat completion API
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            else:
                # Text completion API (for older models)
                response = self.client.completion.create(
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
            
            # Convert to dict for caching
            response_dict = dict(response)
            
            # Cache the response
            if use_cache:
                self.cache[cache_key] = response_dict
                self._save_cache()
            
            return response_dict
            
        except Exception as e:
            print(f"❌ OpenAI API error: {e}")
            return {
                'error': str(e),
                'choices': [{'text': '', 'message': {'content': ''}}]
            }
    
    def extract_text(self, response: Dict[str, Any]) -> str:
        """
        Extract text from OpenAI API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            Generated text
        """
        try:
            if 'error' in response:
                return ''
            
            if 'choices' in response and response['choices']:
                choice = response['choices'][0]
                
                # Chat completion format
                if 'message' in choice and 'content' in choice['message']:
                    return choice['message']['content'].strip()
                
                # Text completion format
                elif 'text' in choice:
                    return choice['text'].strip()
            
            return ''
            
        except Exception as e:
            print(f"❌ Error extracting text: {e}")
            return ''
    
    def batch_generate(self, prompts: List[str], model: str = 'gpt-3.5-turbo',
                      **kwargs) -> List[str]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts
            model: Model name
            **kwargs: Additional API parameters
            
        Returns:
            List of generated texts
        """
        if not OPENAI_AVAILABLE or self.client is None:
            print("❌ OpenAI not available - use FREE local models instead!")
            return [None] * len(prompts)
        
        responses = []
        
        for i, prompt in enumerate(prompts):
            print(f"🔄 Processing prompt {i+1}/{len(prompts)}")
            
            response = self.call_openai(prompt, model=model, **kwargs)
            text = self.extract_text(response)
            responses.append(text)
            
            # Rate limiting
            time.sleep(0.1)
        
        return responses
    
    def get_token_probabilities(self, prompt: str, choices: List[str], 
                               model: str = 'gpt-3.5-turbo') -> List[float]:
        """
        Get probabilities for specific choices (approximation for chat models).
        
        Args:
            prompt: Input prompt
            choices: List of choice strings
            model: Model name
            
        Returns:
            List of estimated probabilities
        """
        print("⚠️ Note: Token probabilities not directly available for chat models.")
        print("Using completion likelihood as approximation.")
        
        probabilities = []
        
        for choice in choices:
            full_prompt = f"{prompt} {choice}"
            
            # Get log probability estimate by asking model to rate likelihood
            eval_prompt = f"""Rate the likelihood of this completion on a scale of 0.0 to 1.0:

Prompt: {prompt}
Completion: {choice}

Likelihood (0.0-1.0):"""
            
            response = self.call_openai(eval_prompt, model=model, max_tokens=10, temperature=0.0)
            text = self.extract_text(response)
            
            try:
                # Extract numerical probability
                prob = float(text.strip().split()[0])
                prob = max(0.0, min(1.0, prob))  # Clamp to [0, 1]
            except:
                prob = 0.5  # Default fallback
            
            probabilities.append(prob)
        
        return probabilities
    
    def clear_cache(self):
        """Clear the API cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()
        print("🗑️ Cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'total_entries': len(self.cache),
            'cache_size_mb': self.cache_file.stat().st_size / (1024 * 1024) if self.cache_file.exists() else 0
        }

# Convenience functions
def call_openai(prompt: str, model: str = 'gpt-3.5-turbo', **kwargs) -> Dict[str, Any]:
    """Convenience function to call OpenAI API."""
    if not OPENAI_AVAILABLE:
        print("💡 Install OpenAI with: pip install openai")
        print("🆓 Or use FREE local models instead!")
        return {
            'error': "OpenAI not available",
            'choices': [{'text': '', 'message': {'content': ''}}]
        }
        
    client = OpenAIClient()
    return client.call_openai(prompt, model=model, **kwargs)

def extract_text(response: Dict[str, Any]) -> str:
    """Convenience function to extract text from response."""
    if not OPENAI_AVAILABLE:
        print("💡 Install OpenAI with: pip install openai")
        print("🆓 Or use FREE local models instead!")
        return ''
    
    client = OpenAIClient()
    return client.extract_text(response)

def batch_openai_generate(prompts: List[str], model: str = 'gpt-3.5-turbo', **kwargs) -> List[str]:
    """Convenience function for batch generation."""
    if not OPENAI_AVAILABLE:
        print("💡 Install OpenAI with: pip install openai")
        print("🆓 Or use FREE local models instead!")
        return [None] * len(prompts)
    
    client = OpenAIClient()
    return client.batch_generate(prompts, model=model, **kwargs)

# Alternative API clients (for future expansion)
class AnthropicClient:
    """Placeholder for Anthropic Claude API client."""
    
    def __init__(self):
        print("⚠️ Anthropic client not implemented yet")
    
    def call_anthropic(self, prompt: str, **kwargs):
        raise NotImplementedError("Anthropic API not implemented")

class CohereClient:
    """Placeholder for Cohere API client."""
    
    def __init__(self):
        print("⚠️ Cohere client not implemented yet")
    
    def call_cohere(self, prompt: str, **kwargs):
        raise NotImplementedError("Cohere API not implemented")

# Export main classes and functions
__all__ = [
    'OpenAIClient',
    'AnthropicClient',
    'CohereClient',
    'call_openai',
    'extract_text',
    'batch_openai_generate'
] 