"""
Automatic Prompt Engineering (APE) Framework
Implementation based on Zhou et al. (2022) for automatic generation and selection
of instruction candidates for bias reduction in LLMs.

Reference: Zhou et al. (2022) "Large Language Models Are Human-Level Prompt Engineers"
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Callable, Any
import itertools
import logging
from dataclasses import dataclass
from tqdm import tqdm
import json

logger = logging.getLogger(__name__)


@dataclass
class PromptCandidate:
    """Structure for prompt candidates with metadata."""
    instruction: str
    score: float
    bias_metrics: Dict[str, float]
    complexity: int
    strategy_type: str


class AutomaticPromptEngineer:
    """
    Automatic Prompt Engineering framework for bias reduction.
    
    Generates and evaluates instruction candidates to find prompts that
    minimize bias while maintaining task performance.
    """
    
    def __init__(self, llm_prober, bias_evaluator):
        """
        Initialize APE with LLM prober and bias evaluator.
        
        Args:
            llm_prober: LLMProber instance for surprisal calculation
            bias_evaluator: BiasEvaluator instance for bias scoring
        """
        self.llm_prober = llm_prober
        self.bias_evaluator = bias_evaluator
        self.candidate_history = []
        
    def generate_candidate_prompts(
        self, 
        demonstrations: List[Dict[str, str]], 
        n_candidates: int = 20,
        seed_prompts: List[str] = None
    ) -> List[str]:
        """
        Generate instruction candidates using demonstration examples.
        
        Args:
            demonstrations: List of input-output demonstration pairs
            n_candidates: Number of candidate prompts to generate
            seed_prompts: Optional seed prompts to start with
            
        Returns:
            List of candidate instruction strings
        """
        candidates = []
        
        # Use seed prompts if provided
        if seed_prompts:
            candidates.extend(seed_prompts)
        
        # Template-based generation strategies
        bias_reduction_templates = [
            "Evaluate the following statement objectively and provide balanced probabilities:",
            "As a neutral and unbiased system, analyze:",
            "Consider multiple perspectives when evaluating:",
            "Provide fact-based assessment without taking sides:",
            "Analyze the claim using evidence and avoiding bias:",
            "Think critically and avoid political or ideological bias when evaluating:",
            "From a balanced, multi-perspective viewpoint, assess:",
            "Using logical reasoning and avoiding preconceptions, evaluate:",
            "Consider the complexity and nuance of the issue when assessing:",
            "Provide an objective analysis that acknowledges different viewpoints:"
        ]
        
        # Instruction framing variations
        instruction_frames = [
            "Instructions: {template}",
            "{template}",
            "Task: {template}",
            "Please {template}",
            "Your goal is to {template}"
        ]
        
        # Chain-of-thought variations
        cot_templates = [
            "Think step by step: {template}",
            "Let's approach this systematically: {template}",
            "Consider the following carefully: {template}",
            "Break this down methodically: {template}"
        ]
        
        # Generate combinations
        for template in bias_reduction_templates:
            for frame in instruction_frames:
                # Handle special frames that need lowercase templates  
                if "Please" in frame or "Your goal" in frame:
                    # Make template lowercase for imperative frames
                    formatted_template = template.lower()
                    candidate = frame.format(template=formatted_template)
                else:
                    candidate = frame.format(template=template)
                    
                if candidate not in candidates:
                    candidates.append(candidate)
                    
        # Add CoT variations
        for template in bias_reduction_templates[:5]:  # Limit CoT for efficiency
            for cot_frame in cot_templates:
                candidate = cot_frame.format(template=template)
                if candidate not in candidates:
                    candidates.append(candidate)
        
        # Meta-prompting strategies
        meta_prompts = [
            "What instruction would lead to the most unbiased evaluation of political claims?",
            "Design a prompt that minimizes political bias in language model responses",
            "Create instructions for objective assessment of controversial topics",
            "Formulate a prompt that encourages balanced political analysis"
        ]
        candidates.extend(meta_prompts)
        
        # Perspective-taking debiasing
        perspective_prompts = [
            "Consider how both supporters and critics would view this claim:",
            "Evaluate this from multiple political perspectives:",
            "What would different stakeholders say about this statement?",
            "Consider the historical context and multiple viewpoints:"
        ]
        candidates.extend(perspective_prompts)
        
        # Limit to requested number
        return candidates[:n_candidates]
    
    def evaluate_prompt_bias(
        self, 
        prompt_template: str, 
        stimuli: List[Dict], 
        evaluate_fn: Callable = None
    ) -> Dict[str, float]:
        """
        Evaluate bias metrics for a given prompt template.
        
        Args:
            prompt_template: The instruction template to evaluate
            stimuli: List of stimulus pairs to test
            evaluate_fn: Optional custom evaluation function
            
        Returns:
            Dictionary of bias metrics
        """
        bias_scores = []
        absolute_bias_scores = []
        
        for stimulus in stimuli:
            try:
                # Create prompts with the template
                if 'pro_palestine' in stimulus:  # Political stimulus
                    option_a = stimulus['pro_palestine']
                    option_b = stimulus['pro_israel']
                    context = stimulus['context']
                else:  # Cultural stimulus
                    option_a = stimulus['religious']
                    option_b = stimulus['secular']
                    context = stimulus['context']
                
                # Format prompts with template
                full_prompt_a = f"{prompt_template}\n\n{option_a}{context}"
                full_prompt_b = f"{prompt_template}\n\n{option_b}{context}"
                
                # Calculate surprisal
                surprisal_a = self._calculate_surprisal_safe(full_prompt_a)
                surprisal_b = self._calculate_surprisal_safe(full_prompt_b)
                
                # Compute bias score
                bias_score = surprisal_a - surprisal_b
                bias_scores.append(bias_score)
                absolute_bias_scores.append(abs(bias_score))
                
            except Exception as e:
                logger.warning(f"Error evaluating stimulus: {e}")
                continue
        
        if not bias_scores:
            return {'mean_bias': float('inf'), 'absolute_bias': float('inf'), 'consistency': 0.0}
        
        # Calculate metrics
        mean_bias = np.mean(bias_scores)
        absolute_bias = np.mean(absolute_bias_scores)
        std_bias = np.std(bias_scores)
        consistency = 1.0 - (std_bias / (abs(mean_bias) + 1e-8))  # Higher = more consistent
        
        return {
            'mean_bias': mean_bias,
            'absolute_bias': absolute_bias,
            'bias_variance': std_bias,
            'consistency': consistency,
            'num_evaluated': len(bias_scores)
        }
    
    def _calculate_surprisal_safe(self, prompt: str) -> float:
        """Safe surprisal calculation with error handling."""
        try:
            # Simple approximation using model likelihood
            return self.llm_prober.surprisal("", prompt)
        except Exception as e:
            logger.warning(f"Surprisal calculation failed: {e}")
            return 10.0  # High surprisal as fallback
    
    def select_top_prompts(
        self, 
        candidates: List[PromptCandidate], 
        k: int = 5,
        selection_criteria: str = "absolute_bias"
    ) -> List[PromptCandidate]:
        """
        Select top-k prompts based on bias reduction criteria.
        
        Args:
            candidates: List of evaluated prompt candidates
            k: Number of top prompts to select
            selection_criteria: Metric to optimize ('absolute_bias', 'consistency', 'combined')
            
        Returns:
            Top-k prompt candidates
        """
        if selection_criteria == "absolute_bias":
            # Select prompts with lowest absolute bias
            sorted_candidates = sorted(candidates, key=lambda x: x.bias_metrics['absolute_bias'])
        elif selection_criteria == "consistency":
            # Select prompts with highest consistency
            sorted_candidates = sorted(candidates, key=lambda x: -x.bias_metrics['consistency'])
        elif selection_criteria == "combined":
            # Combined score: minimize bias, maximize consistency
            def combined_score(candidate):
                bias = candidate.bias_metrics['absolute_bias']
                consistency = candidate.bias_metrics['consistency']
                return bias - 0.5 * consistency  # Lower is better
            sorted_candidates = sorted(candidates, key=combined_score)
        else:
            raise ValueError(f"Unknown selection criteria: {selection_criteria}")
        
        return sorted_candidates[:k]
    
    def run_ape_pipeline(
        self, 
        stimuli: List[Dict], 
        n_candidates: int = 20,
        top_k: int = 5,
        seed_prompts: List[str] = None
    ) -> Tuple[List[PromptCandidate], Dict[str, Any]]:
        """
        Run the complete APE pipeline.
        
        Args:
            stimuli: Stimulus pairs for evaluation
            n_candidates: Number of candidates to generate
            top_k: Number of top prompts to return
            seed_prompts: Optional seed prompts
            
        Returns:
            Tuple of (top prompts, pipeline metrics)
        """
        logger.info("🤖 Starting APE Pipeline...")
        
        # Step 1: Generate candidates
        logger.info(f"📝 Generating {n_candidates} candidate prompts...")
        candidate_instructions = self.generate_candidate_prompts(
            demonstrations=[], 
            n_candidates=n_candidates,
            seed_prompts=seed_prompts
        )
        
        # Step 2: Evaluate candidates
        logger.info(f"🧮 Evaluating {len(candidate_instructions)} candidates...")
        evaluated_candidates = []
        
        for i, instruction in enumerate(tqdm(candidate_instructions, desc="Evaluating prompts")):
            bias_metrics = self.evaluate_prompt_bias(instruction, stimuli)
            
            candidate = PromptCandidate(
                instruction=instruction,
                score=bias_metrics['absolute_bias'],
                bias_metrics=bias_metrics,
                complexity=len(instruction.split()),
                strategy_type=self._classify_strategy(instruction)
            )
            evaluated_candidates.append(candidate)
        
        # Step 3: Select top prompts
        logger.info(f"🏆 Selecting top {top_k} prompts...")
        top_prompts = self.select_top_prompts(evaluated_candidates, top_k)
        
        # Pipeline metrics
        all_scores = [c.bias_metrics['absolute_bias'] for c in evaluated_candidates]
        pipeline_metrics = {
            'total_candidates': len(evaluated_candidates),
            'best_absolute_bias': min(all_scores),
            'worst_absolute_bias': max(all_scores),
            'mean_absolute_bias': np.mean(all_scores),
            'improvement_ratio': max(all_scores) / min(all_scores)
        }
        
        # Store history
        self.candidate_history.append({
            'candidates': evaluated_candidates,
            'top_prompts': top_prompts,
            'metrics': pipeline_metrics
        })
        
        logger.info(f"✅ APE Pipeline complete!")
        logger.info(f"   Best absolute bias: {pipeline_metrics['best_absolute_bias']:.4f}")
        logger.info(f"   Improvement ratio: {pipeline_metrics['improvement_ratio']:.2f}x")
        
        return top_prompts, pipeline_metrics
    
    def _classify_strategy(self, instruction: str) -> str:
        """Classify prompt strategy type."""
        instruction_lower = instruction.lower()
        
        if "step by step" in instruction_lower or "think" in instruction_lower:
            return "chain_of_thought"
        elif "objective" in instruction_lower or "unbiased" in instruction_lower:
            return "bias_reduction"
        elif "perspective" in instruction_lower or "viewpoint" in instruction_lower:
            return "perspective_taking"
        elif "instruction" in instruction_lower or "task" in instruction_lower:
            return "instruction_following"
        else:
            return "general"
    
    def save_results(self, filepath: str):
        """Save APE results to JSON file."""
        results = {
            'pipeline_runs': len(self.candidate_history),
            'history': []
        }
        
        for run in self.candidate_history:
            run_data = {
                'metrics': run['metrics'],
                'top_prompts': [
                    {
                        'instruction': p.instruction,
                        'score': p.score,
                        'bias_metrics': p.bias_metrics,
                        'strategy_type': p.strategy_type
                    }
                    for p in run['top_prompts']
                ]
            }
            results['history'].append(run_data)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"💾 APE results saved to {filepath}")


def compare_prompt_effectiveness(
    original_prompts: List[str],
    optimized_prompts: List[str], 
    stimuli: List[Dict],
    llm_prober
) -> pd.DataFrame:
    """
    Compare effectiveness of original vs APE-optimized prompts.
    
    Returns comparison DataFrame with bias metrics.
    """
    comparison_data = []
    
    for prompt_type, prompts in [("Original", original_prompts), ("APE-Optimized", optimized_prompts)]:
        for prompt in prompts:
            # Evaluate prompt
            ape = AutomaticPromptEngineer(llm_prober, None)
            metrics = ape.evaluate_prompt_bias(prompt, stimuli)
            
            comparison_data.append({
                'prompt_type': prompt_type,
                'prompt': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                'mean_bias': metrics['mean_bias'],
                'absolute_bias': metrics['absolute_bias'],
                'consistency': metrics['consistency']
            })
    
    return pd.DataFrame(comparison_data) 