"""
LLM Helper Functions for Bias Evaluation

This module provides utilities for loading models and computing surprisal values
for bias evaluation experiments.
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TFAutoModelForCausalLM,
    pipeline
)
import numpy as np
import logging
from typing import List, Tuple, Dict, Union, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

class LLMProber:
    """
    A unified interface for probing language models for bias evaluation.
    Supports both PyTorch and TensorFlow models.
    """
    
    def __init__(self, model_name: str = "gpt2", device: str = "auto", use_tensorflow: bool = False):
        """
        Initialize the LLM prober.
        
        Args:
            model_name: Name of the model to load
            device: Device to use ('cuda', 'mps', 'cpu', or 'auto')
            use_tensorflow: Whether to use TensorFlow instead of PyTorch
        """
        self.model_name = model_name
        self.use_tensorflow = use_tensorflow
        
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        print(f"🔧 Loading {model_name} on {self.device} (TF: {use_tensorflow})")
        
        # Load tokenizer and model
        self.tokenizer, self.model = self.load_model()
        
    def load_model(self) -> Tuple[AutoTokenizer, Union[AutoModelForCausalLM, TFAutoModelForCausalLM]]:
        """
        Load model and tokenizer.
        
        Returns:
            Tuple of (tokenizer, model)
        """
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            if self.use_tensorflow:
                try:
                    import tensorflow as tf
                    model = TFAutoModelForCausalLM.from_pretrained(self.model_name)
                    print(f"✅ Loaded TensorFlow model: {self.model_name}")
                except ImportError:
                    print("⚠️ TensorFlow not available, falling back to PyTorch")
                    model = AutoModelForCausalLM.from_pretrained(self.model_name)
                    if self.device != "cpu":
                        model = model.to(self.device)
            else:
                model = AutoModelForCausalLM.from_pretrained(self.model_name)
                if self.device != "cpu":
                    model = model.to(self.device)
            
            model.eval()
            print(f"✅ Model loaded successfully")
            return tokenizer, model
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def next_seq_prob(self, context: str, choices: List[str]) -> List[float]:
        """
        Compute the probability of each choice given the context.
        
        Args:
            context: The context string
            choices: List of possible continuations
            
        Returns:
            List of probabilities for each choice
        """
        probabilities = []
        
        for choice in choices:
            full_text = context + choice
            
            # Tokenize
            inputs = self.tokenizer(full_text, return_tensors="pt")
            context_inputs = self.tokenizer(context, return_tensors="pt")
            
            if not self.use_tensorflow and self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                context_inputs = {k: v.to(self.device) for k, v in context_inputs.items()}
            
            with torch.no_grad():
                # Get model outputs
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Calculate probability for the choice tokens
                context_len = context_inputs['input_ids'].shape[1]
                choice_len = inputs['input_ids'].shape[1] - context_len
                
                if choice_len > 0:
                    # Get logits for choice tokens
                    choice_logits = logits[0, context_len-1:context_len-1+choice_len, :]
                    choice_tokens = inputs['input_ids'][0, context_len:context_len+choice_len]
                    
                    # Calculate log probabilities
                    log_probs = F.log_softmax(choice_logits, dim=-1)
                    token_log_probs = log_probs.gather(1, choice_tokens.unsqueeze(1)).squeeze(1)
                    
                    # Sum log probabilities and convert to probability
                    total_log_prob = token_log_probs.sum()
                    prob = torch.exp(total_log_prob).item()
                else:
                    prob = 1.0
                    
            probabilities.append(prob)
        
        return probabilities
    
    def compute_surprisal(self, context: str, choices: List[str]) -> List[float]:
        """
        Compute surprisal (-log probability) for each choice.
        
        Args:
            context: The context string
            choices: List of possible continuations
            
        Returns:
            List of surprisal values
        """
        probabilities = self.next_seq_prob(context, choices)
        surprisal_values = []
        
        for prob in probabilities:
            if prob > 0:
                surprisal = -np.log(prob)
            else:
                surprisal = float('inf')  # Handle zero probability
            surprisal_values.append(surprisal)
            
        return surprisal_values
    
    def compute_bias_score(self, surprisal_values: List[float]) -> float:
        """
        Compute bias score as difference in surprisal values.
        
        Args:
            surprisal_values: List of surprisal values [choice_1, choice_2]
            
        Returns:
            Bias score (surprisal_choice1 - surprisal_choice2)
        """
        if len(surprisal_values) != 2:
            raise ValueError("Expected exactly 2 surprisal values for bias computation")
        
        return surprisal_values[0] - surprisal_values[1]
    
    def batch_evaluate(self, contexts: List[str], choices_list: List[List[str]]) -> List[Dict]:
        """
        Evaluate multiple context-choice pairs in batch.
        
        Args:
            contexts: List of context strings
            choices_list: List of choice lists for each context
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for i, (context, choices) in enumerate(zip(contexts, choices_list)):
            try:
                surprisal = self.compute_surprisal(context, choices)
                probabilities = self.next_seq_prob(context, choices)
                bias_score = self.compute_bias_score(surprisal)
                
                result = {
                    'id': i,
                    'context': context,
                    'choices': choices,
                    'probabilities': probabilities,
                    'surprisal': surprisal,
                    'bias_score': bias_score
                }
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    print(f"✅ Processed {i + 1}/{len(contexts)} examples")
                    
            except Exception as e:
                print(f"❌ Error processing example {i}: {e}")
                result = {
                    'id': i,
                    'context': context,
                    'choices': choices,
                    'error': str(e)
                }
                results.append(result)
        
        return results

# Legacy function wrappers for backward compatibility
def load_model(model_name: str = "gpt2", device: str = "auto") -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """Legacy function for loading models."""
    prober = LLMProber(model_name, device)
    return prober.tokenizer, prober.model

def next_seq_prob(model, tokenizer, context: str, choices: List[str], device: str = "cpu") -> List[float]:
    """Legacy function for computing sequence probabilities."""
    # Create temporary prober instance
    prober = LLMProber("gpt2")  # model_name doesn't matter here
    prober.model = model
    prober.tokenizer = tokenizer
    prober.device = device
    return prober.next_seq_prob(context, choices)

def compute_surprisal(model, tokenizer, context: str, choices: List[str], device: str = "cpu") -> List[float]:
    """Legacy function for computing surprisal."""
    # Create temporary prober instance
    prober = LLMProber("gpt2")  # model_name doesn't matter here
    prober.model = model
    prober.tokenizer = tokenizer
    prober.device = device
    return prober.compute_surprisal(context, choices) 