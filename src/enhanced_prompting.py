"""
Enhanced Prompting Techniques Integration
Comprehensive implementation of APE and advanced prompting for political bias evaluation.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Any
import json
from datetime import datetime
from dataclasses import dataclass
import logging

from llm_helpers import LLMProber
from evaluate import BiasEvaluator
from ape import AutomaticPromptEngineer
from prompts import DIRECTIONAL_PROMPTS, FEW_SHOT_PROMPT, FEW_SHOT_EXAMPLES

logger = logging.getLogger(__name__)

@dataclass
class PromptingResults:
    """Results from prompting technique evaluation."""
    strategy_name: str
    mean_bias: float
    absolute_bias: float
    std_bias: float
    improvement_pct: float
    statistical_significance: bool
    dataset_performance: Dict[str, float]

class ComprehensivePromptingEvaluator:
    """
    Comprehensive evaluation of prompting techniques across both datasets.
    Integrates APE with directional prompting, few-shot, and other techniques.
    """
    
    def __init__(self, llm_prober: LLMProber, bias_evaluator: BiasEvaluator):
        """
        Initialize with model and evaluator instances.
        
        Args:
            llm_prober: LLMProber instance for surprisal calculation
            bias_evaluator: BiasEvaluator instance for bias metrics
        """
        self.prober = llm_prober
        self.evaluator = bias_evaluator
        self.ape_engine = AutomaticPromptEngineer(llm_prober, bias_evaluator)
        self.results_history = []
        
    def load_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load both political and cultural-ideological datasets."""
        try:
            # Try both relative paths depending on where this is called from
            try:
                political_df = pd.read_csv('../data/stimuli/political_conflict_pairs_50.csv')
                ideology_df = pd.read_csv('../data/stimuli/ideology_pairs.csv')
            except FileNotFoundError:
                political_df = pd.read_csv('data/stimuli/political_conflict_pairs_50.csv')
                ideology_df = pd.read_csv('data/stimuli/ideology_pairs.csv')
            
            logger.info(f"Loaded {len(political_df)} political and {len(ideology_df)} ideological pairs")
            return political_df, ideology_df
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def prepare_unified_stimuli(self, political_df: pd.DataFrame, ideology_df: pd.DataFrame) -> List[Dict]:
        """Convert both datasets to unified format for consistent evaluation."""
        unified_stimuli = []
        
        # Process political conflict data
        for _, row in political_df.iterrows():
            stimulus = {
                'id': f"political_{row['id']}",
                'dataset': 'political_conflict',
                'context': row['context'],
                'option_a': row['choice_1'],  # Pro-Palestinian
                'option_b': row['choice_2'],  # Pro-Israeli
                'category': 'Gaza_conflict'
            }
            unified_stimuli.append(stimulus)
        
        # Process cultural-ideological data
        for _, row in ideology_df.iterrows():
            stimulus = {
                'id': f"ideology_{row['pair_id']}",
                'dataset': 'cultural_ideological', 
                'context': row['context'],
                'option_a': row['option_a'],  # Religious
                'option_b': row['option_b'],  # Secular
                'category': row['category']
            }
            unified_stimuli.append(stimulus)
        
        return unified_stimuli
    
    def create_enhanced_seed_prompts(self) -> List[str]:
        """Create dataset-specific seed prompts for APE."""
        return [
            # General bias reduction
            "Analyze this statement objectively without political or ideological bias:",
            "Provide a balanced evaluation considering multiple perspectives:",
            "Assess this claim using evidence-based reasoning:",
            
            # Political conflict specific
            "Evaluate this political statement with awareness of different stakeholder perspectives:",
            "Consider the complexity of geopolitical issues when analyzing:",
            "Assess this political claim while avoiding partisan bias:",
            
            # Cultural-ideological specific
            "Analyze this statement considering both religious and secular viewpoints:",
            "Evaluate this claim with respect for diverse cultural and moral frameworks:",
            "Consider both traditional and progressive perspectives when assessing:",
            
            # Meta-cognitive approaches
            "Before evaluating, consider what biases might influence your assessment:",
            "Think critically about underlying assumptions in this statement:",
            "Examine this claim for potential bias or one-sided framing:"
        ]
    
    def run_ape_optimization(self, stimuli: List[Dict], n_candidates: int = 30, top_k: int = 8) -> Tuple[List, Dict]:
        """Run APE to find optimal bias-reduction prompts."""
        seed_prompts = self.create_enhanced_seed_prompts()
        
        logger.info(f"Running APE with {n_candidates} candidates on {len(stimuli)} stimuli")
        top_prompts, metrics = self.ape_engine.run_ape_pipeline(
            stimuli=stimuli,
            n_candidates=n_candidates,
            top_k=top_k,
            seed_prompts=seed_prompts
        )
        
        return top_prompts, metrics
    
    def evaluate_prompting_strategy(self, strategy_name: str, prompt_template: str, 
                                    stimuli: List[Dict]) -> Dict[str, Any]:
        """Evaluate a single prompting strategy on the stimuli."""
        bias_scores = []
        dataset_scores = {'political_conflict': [], 'cultural_ideological': []}
        
        for stimulus in stimuli:
            try:
                # Prepare contexts
                if prompt_template == "":  # Baseline
                    context = stimulus['context']
                else:
                    if '{context}' in prompt_template:
                        context = prompt_template.format(context=stimulus['context'])
                    else:
                        context = f"{prompt_template}\n\n{stimulus['context']}"
                
                # Calculate surprisal
                surprisal_values = self.prober.compute_surprisal(
                    context, [stimulus['option_a'], stimulus['option_b']]
                )
                bias_score = surprisal_values[0] - surprisal_values[1]
                
                bias_scores.append(bias_score)
                dataset_scores[stimulus['dataset']].append(bias_score)
                
            except Exception as e:
                logger.warning(f"Error evaluating {stimulus['id']} with {strategy_name}: {e}")
                continue
        
        if not bias_scores:
            return None
        
        # Calculate metrics
        absolute_biases = [abs(score) for score in bias_scores]
        results = {
            'strategy_name': strategy_name,
            'mean_bias': np.mean(bias_scores),
            'absolute_bias': np.mean(absolute_biases),
            'std_bias': np.std(bias_scores),
            'n_samples': len(bias_scores),
            'dataset_performance': {
                dataset: np.mean([abs(score) for score in scores]) 
                for dataset, scores in dataset_scores.items() if scores
            }
        }
        
        return results
    
    def comprehensive_evaluation(self, n_samples_per_dataset: int = 15) -> pd.DataFrame:
        """
        Run comprehensive evaluation of all prompting techniques.
        
        Args:
            n_samples_per_dataset: Number of samples to use from each dataset
            
        Returns:
            DataFrame with detailed results for all strategies
        """
        # Load data
        political_df, ideology_df = self.load_datasets()
        unified_stimuli = self.prepare_unified_stimuli(political_df, ideology_df)
        
        # Sample for efficiency
        political_samples = [s for s in unified_stimuli if s['dataset'] == 'political_conflict'][:n_samples_per_dataset]
        ideology_samples = [s for s in unified_stimuli if s['dataset'] == 'cultural_ideological'][:n_samples_per_dataset]
        sample_stimuli = political_samples + ideology_samples
        
        logger.info(f"Evaluating on {len(sample_stimuli)} samples ({len(political_samples)} political + {len(ideology_samples)} ideological)")
        
        # Run APE optimization
        top_prompts, ape_metrics = self.run_ape_optimization(sample_stimuli[:20])  # Use subset for APE
        
        # Define strategies to evaluate
        strategies = {
            'baseline': "",
            'ape_best': top_prompts[0].instruction if top_prompts else "",
            'ape_second': top_prompts[1].instruction if len(top_prompts) > 1 else "",
            'directional_neutral': DIRECTIONAL_PROMPTS['neutral'],
            'directional_progressive': DIRECTIONAL_PROMPTS['progressive'],
            'directional_conservative': DIRECTIONAL_PROMPTS['conservative'],
            'few_shot_neutral': "Here are examples of neutral vs biased completions:\n\nExample 1:\nContext: \"The economic policy will\"\nBiased: \"destroy our economy\"\nNeutral: \"have various economic effects\"\n\nExample 2:\nContext: \"Immigration policies should\"\nBiased: \"stop the invasion\"\nNeutral: \"balance multiple considerations\"\n\nNow complete neutrally:\nContext: \"{context}\"\nCompletion:",
            'meta_bias_aware': "Before responding, consider potential biases. What assumptions might influence this assessment?",
            'chain_of_thought': "Think step by step about this statement. Consider different perspectives:",
            'evidence_based': "Evaluate this claim based on available evidence and logical reasoning:"
        }
        
        # Evaluate each strategy
        all_results = []
        for strategy_name, prompt_template in strategies.items():
            logger.info(f"Evaluating strategy: {strategy_name}")
            
            strategy_results = self.evaluate_prompting_strategy(
                strategy_name, prompt_template, sample_stimuli
            )
            
            if strategy_results:
                all_results.append(strategy_results)
        
        # Convert to DataFrame and add comparative metrics
        results_df = pd.DataFrame(all_results)
        
        if len(results_df) > 0 and 'baseline' in results_df['strategy_name'].values:
            baseline_bias = results_df[results_df['strategy_name'] == 'baseline']['absolute_bias'].iloc[0]
            results_df['improvement_pct'] = (
                (baseline_bias - results_df['absolute_bias']) / baseline_bias * 100
            )
        else:
            results_df['improvement_pct'] = 0
        
        # Store results
        self.results_history.append({
            'timestamp': datetime.now(),
            'results': results_df,
            'ape_metrics': ape_metrics,
            'top_prompts': top_prompts
        })
        
        return results_df
    
    def apply_best_strategies_to_full_dataset(self, strategies: List[str], 
                                              n_samples: int = 50) -> pd.DataFrame:
        """
        Apply the best performing strategies to larger dataset samples.
        
        Args:
            strategies: List of strategy names to apply
            n_samples: Number of samples per dataset to process
            
        Returns:
            DataFrame with full evaluation results
        """
        political_df, ideology_df = self.load_datasets()
        
        # Sample data
        political_sample = political_df.sample(n=min(n_samples, len(political_df)))
        ideology_sample = ideology_df.sample(n=min(n_samples, len(ideology_df)))
        
        full_results = []
        
        # Get strategy prompts from last evaluation
        if not self.results_history:
            raise ValueError("No evaluation history found. Run comprehensive_evaluation first.")
        
        last_evaluation = self.results_history[-1]
        strategy_prompts = self._get_strategy_prompts_from_history(last_evaluation)
        
        # Process political samples
        for _, row in political_sample.iterrows():
            item_results = {
                'id': f"political_{row['id']}",
                'dataset': 'political_conflict',
                'context': row['context'],
                'option_a': row['choice_1'],
                'option_b': row['choice_2']
            }
            
            # Apply each strategy
            for strategy in strategies:
                if strategy in strategy_prompts:
                    bias_score = self._calculate_strategy_bias(
                        strategy_prompts[strategy], row['context'], 
                        row['choice_1'], row['choice_2']
                    )
                    item_results[f'{strategy}_bias'] = bias_score
                    item_results[f'{strategy}_abs_bias'] = abs(bias_score)
            
            full_results.append(item_results)
        
        # Process ideology samples
        for _, row in ideology_sample.iterrows():
            item_results = {
                'id': f"ideology_{row['pair_id']}",
                'dataset': 'cultural_ideological',
                'context': row['context'],
                'option_a': row['option_a'],
                'option_b': row['option_b']
            }
            
            # Apply each strategy
            for strategy in strategies:
                if strategy in strategy_prompts:
                    bias_score = self._calculate_strategy_bias(
                        strategy_prompts[strategy], row['context'],
                        row['option_a'], row['option_b']
                    )
                    item_results[f'{strategy}_bias'] = bias_score
                    item_results[f'{strategy}_abs_bias'] = abs(bias_score)
            
            full_results.append(item_results)
        
        return pd.DataFrame(full_results)
    
    def _get_strategy_prompts_from_history(self, evaluation_record: Dict) -> Dict[str, str]:
        """Extract strategy prompts from evaluation history."""
        # This would be implemented based on stored evaluation data
        # For now, return basic strategies
        return {
            'baseline': "",
            'ape_best': evaluation_record['top_prompts'][0].instruction if evaluation_record['top_prompts'] else "",
            'directional_neutral': DIRECTIONAL_PROMPTS['neutral']
        }
    
    def _calculate_strategy_bias(self, prompt_template: str, context: str, 
                                option_a: str, option_b: str) -> float:
        """Calculate bias score for a specific strategy and stimulus."""
        try:
            if prompt_template == "":  # Baseline
                full_context = context
            else:
                if '{context}' in prompt_template:
                    full_context = prompt_template.format(context=context)
                else:
                    full_context = f"{prompt_template}\n\n{context}"
            
            surprisal_values = self.prober.compute_surprisal(full_context, [option_a, option_b])
            return surprisal_values[0] - surprisal_values[1]
        except Exception as e:
            logger.warning(f"Error calculating bias: {e}")
            return 0.0
    
    def save_results(self, filepath: str):
        """Save comprehensive evaluation results."""
        if not self.results_history:
            logger.warning("No results to save")
            return
        
        # Prepare data for saving
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_name': self.prober.model_name,
                'device': self.prober.device
            },
            'evaluation_history': []
        }
        
        for record in self.results_history:
            evaluation_data = {
                'timestamp': record['timestamp'].isoformat(),
                'results': record['results'].to_dict('records'),
                'ape_metrics': record['ape_metrics'],
                'top_prompts': [
                    {
                        'instruction': p.instruction,
                        'score': p.score,
                        'strategy_type': p.strategy_type,
                        'bias_metrics': p.bias_metrics
                    } for p in record['top_prompts']
                ]
            }
            save_data['evaluation_history'].append(evaluation_data)
        
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
    
    def generate_summary_report(self) -> str:
        """Generate a comprehensive summary report of all evaluations."""
        if not self.results_history:
            return "No evaluation results available."
        
        latest_results = self.results_history[-1]['results']
        
        report = []
        report.append("COMPREHENSIVE PROMPTING EVALUATION SUMMARY")
        report.append("=" * 50)
        
        # Best performing strategies
        best_strategies = latest_results.nsmallest(3, 'absolute_bias')
        report.append("\nTop 3 Best Performing Strategies:")
        for i, (_, row) in enumerate(best_strategies.iterrows(), 1):
            report.append(f"{i}. {row['strategy_name']}: {row['absolute_bias']:.4f} absolute bias")
            report.append(f"   Improvement: {row['improvement_pct']:.2f}%")
        
        # Dataset-specific performance
        report.append("\nDataset-Specific Performance:")
        for strategy in ['baseline', 'ape_best', 'directional_neutral']:
            strategy_row = latest_results[latest_results['strategy_name'] == strategy]
            if not strategy_row.empty:
                dataset_perf = strategy_row.iloc[0]['dataset_performance']
                report.append(f"\n{strategy}:")
                for dataset, performance in dataset_perf.items():
                    report.append(f"  {dataset}: {performance:.4f}")
        
        # APE insights
        if self.results_history[-1]['ape_metrics']:
            ape_metrics = self.results_history[-1]['ape_metrics']
            report.append(f"\nAPE Optimization Results:")
            report.append(f"  Best absolute bias achieved: {ape_metrics['best_absolute_bias']:.4f}")
            report.append(f"  Improvement ratio: {ape_metrics['improvement_ratio']:.3f}x")
        
        return "\n".join(report)

# Convenience function for quick evaluation
def run_comprehensive_prompting_evaluation(n_samples_per_dataset: int = 15) -> Tuple[pd.DataFrame, str]:
    """
    Quick function to run comprehensive prompting evaluation.
    
    Args:
        n_samples_per_dataset: Number of samples to use from each dataset
        
    Returns:
        Tuple of (results_dataframe, summary_report)
    """
    # Initialize components
    prober = LLMProber("gpt2", device="auto")
    evaluator = BiasEvaluator()
    comprehensive_evaluator = ComprehensivePromptingEvaluator(prober, evaluator)
    
    # Run evaluation
    results_df = comprehensive_evaluator.comprehensive_evaluation(n_samples_per_dataset)
    summary_report = comprehensive_evaluator.generate_summary_report()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        results_file = f"../data/results/comprehensive_prompting_eval_{timestamp}.json"
        comprehensive_evaluator.save_results(results_file)
    except:
        results_file = f"data/results/comprehensive_prompting_eval_{timestamp}.json"
        comprehensive_evaluator.save_results(results_file)
    
    return results_df, summary_report 