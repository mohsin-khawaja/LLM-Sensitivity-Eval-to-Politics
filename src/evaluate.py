"""
Evaluation Metrics for Bias Analysis

This module provides comprehensive evaluation functions for measuring bias in language models,
including statistical tests, effect sizes, and advanced analysis metrics.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple, Dict, Optional, Any, Union
import re
import json
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings("ignore")

class BiasEvaluator:
    """
    Comprehensive bias evaluation toolkit for language model analysis.
    """
    
    def __init__(self):
        pass
    
    def compute_bias_score(self, matched: List[float], mismatched: List[float]) -> float:
        """
        Compute bias score as the difference between matched and mismatched conditions.
        
        Args:
            matched: List of surprisal values for matched condition
            mismatched: List of surprisal values for mismatched condition
            
        Returns:
            Bias score (mean_matched - mean_mismatched)
        """
        if len(matched) == 0 or len(mismatched) == 0:
            return 0.0
            
        mean_matched = np.mean(matched)
        mean_mismatched = np.mean(mismatched)
        
        return mean_matched - mean_mismatched
    
    def run_paired_ttest(self, matched: List[float], mismatched: List[float]) -> Tuple[float, float]:
        """
        Run paired t-test between matched and mismatched conditions.
        
        Args:
            matched: List of surprisal values for matched condition
            mismatched: List of surprisal values for mismatched condition
            
        Returns:
            Tuple of (t_statistic, p_value)
        """
        if len(matched) != len(mismatched):
            raise ValueError("Matched and mismatched lists must have the same length for paired t-test")
        
        if len(matched) < 2:
            return 0.0, 1.0
        
        # Remove infinite values
        matched_clean = [x for x in matched if np.isfinite(x)]
        mismatched_clean = [x for x in mismatched if np.isfinite(x)]
        
        if len(matched_clean) != len(mismatched_clean) or len(matched_clean) < 2:
            return 0.0, 1.0
        
        try:
            t_stat, p_value = stats.ttest_rel(matched_clean, mismatched_clean)
            return float(t_stat), float(p_value)
        except Exception:
            return 0.0, 1.0
    
    def compute_cohen_d(self, matched: List[float], mismatched: List[float]) -> float:
        """
        Compute Cohen's d effect size between matched and mismatched conditions.
        
        Args:
            matched: List of surprisal values for matched condition
            mismatched: List of surprisal values for mismatched condition
            
        Returns:
            Cohen's d effect size
        """
        if len(matched) == 0 or len(mismatched) == 0:
            return 0.0
        
        # Clean data
        matched_clean = [x for x in matched if np.isfinite(x)]
        mismatched_clean = [x for x in mismatched if np.isfinite(x)]
        
        if len(matched_clean) == 0 or len(mismatched_clean) == 0:
            return 0.0
        
        mean_matched = np.mean(matched_clean)
        mean_mismatched = np.mean(mismatched_clean)
        
        std_matched = np.std(matched_clean, ddof=1) if len(matched_clean) > 1 else 0
        std_mismatched = np.std(mismatched_clean, ddof=1) if len(mismatched_clean) > 1 else 0
        
        # Pooled standard deviation
        n1, n2 = len(matched_clean), len(mismatched_clean)
        pooled_std = np.sqrt(((n1 - 1) * std_matched**2 + (n2 - 1) * std_mismatched**2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        cohen_d = (mean_matched - mean_mismatched) / pooled_std
        return float(cohen_d)
    
    def measure_cot_complexity(self, traces: List[str]) -> float:
        """
        Measure the complexity of chain-of-thought reasoning traces.
        
        Args:
            traces: List of reasoning trace strings
            
        Returns:
            Average complexity score (based on length, reasoning words, etc.)
        """
        if not traces:
            return 0.0
        
        complexity_scores = []
        
        # Reasoning indicator words
        reasoning_words = [
            'because', 'since', 'therefore', 'thus', 'hence', 'consequently',
            'however', 'although', 'despite', 'nevertheless', 'furthermore',
            'moreover', 'additionally', 'first', 'second', 'finally',
            'step', 'analysis', 'consider', 'perspective', 'reasoning'
        ]
        
        for trace in traces:
            if not trace or not isinstance(trace, str):
                complexity_scores.append(0.0)
                continue
            
            # Basic metrics
            word_count = len(trace.split())
            sentence_count = len([s for s in trace.split('.') if s.strip()])
            
            # Reasoning complexity
            reasoning_count = sum(1 for word in reasoning_words if word in trace.lower())
            
            # Structural complexity (questions, lists, etc.)
            question_count = trace.count('?')
            list_indicators = trace.count('1.') + trace.count('2.') + trace.count('3.')
            
            # Combine metrics
            complexity = (
                word_count * 0.1 +  # Length component
                reasoning_count * 2.0 +  # Reasoning words
                question_count * 1.5 +  # Questions
                list_indicators * 1.0 +  # Structure
                sentence_count * 0.5  # Sentence complexity
            )
            
            complexity_scores.append(complexity)
        
        return float(np.mean(complexity_scores))
    
    def analyze_self_consistency(self, traces_per_example: Dict[str, List[str]]) -> float:
        """
        Analyze self-consistency across multiple reasoning traces per example.
        
        Args:
            traces_per_example: Dict mapping example IDs to lists of reasoning traces
            
        Returns:
            Average consistency score (0-1, higher = more consistent)
        """
        if not traces_per_example:
            return 0.0
        
        consistency_scores = []
        
        for example_id, traces in traces_per_example.items():
            if not traces or len(traces) < 2:
                consistency_scores.append(1.0)  # Single trace is perfectly consistent
                continue
            
            # Compare all pairs of traces
            pairwise_similarities = []
            
            for i in range(len(traces)):
                for j in range(i + 1, len(traces)):
                    similarity = self._compute_trace_similarity(traces[i], traces[j])
                    pairwise_similarities.append(similarity)
            
            if pairwise_similarities:
                consistency = np.mean(pairwise_similarities)
            else:
                consistency = 1.0
            
            consistency_scores.append(consistency)
        
        return float(np.mean(consistency_scores))
    
    def _compute_trace_similarity(self, trace1: str, trace2: str) -> float:
        """
        Compute similarity between two reasoning traces.
        
        Args:
            trace1: First reasoning trace
            trace2: Second reasoning trace
            
        Returns:
            Similarity score (0-1)
        """
        if not trace1 or not trace2:
            return 0.0
        
        # Tokenize and normalize
        words1 = set(trace1.lower().split())
        words2 = set(trace2.lower().split())
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 1.0
        
        return intersection / union
    
    def compute_comprehensive_metrics(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compute comprehensive bias metrics from results dataframe.
        
        Args:
            results_df: DataFrame with columns ['strategy', 'matched_surprisal', 'mismatched_surprisal']
            
        Returns:
            Dictionary of comprehensive metrics
        """
        metrics = {}
        
        strategies = results_df['strategy'].unique() if 'strategy' in results_df.columns else ['overall']
        
        for strategy in strategies:
            if 'strategy' in results_df.columns:
                strategy_data = results_df[results_df['strategy'] == strategy]
            else:
                strategy_data = results_df
            
            if 'matched_surprisal' in strategy_data.columns and 'mismatched_surprisal' in strategy_data.columns:
                matched = strategy_data['matched_surprisal'].dropna().tolist()
                mismatched = strategy_data['mismatched_surprisal'].dropna().tolist()
            else:
                # Fallback: try to infer from other columns
                matched = []
                mismatched = []
            
            if matched and mismatched:
                bias_score = self.compute_bias_score(matched, mismatched)
                t_stat, p_value = self.run_paired_ttest(matched, mismatched)
                cohen_d = self.compute_cohen_d(matched, mismatched)
                
                metrics[strategy] = {
                    'bias_score': bias_score,
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohen_d': cohen_d,
                    'n_examples': len(matched),
                    'mean_matched': np.mean(matched),
                    'mean_mismatched': np.mean(mismatched),
                    'std_matched': np.std(matched),
                    'std_mismatched': np.std(mismatched)
                }
        
        return metrics

# Standalone functions for backward compatibility and direct use
def compute_bias_score(matched: List[float], mismatched: List[float]) -> float:
    """Compute bias score as difference between matched and mismatched conditions."""
    evaluator = BiasEvaluator()
    return evaluator.compute_bias_score(matched, mismatched)

def run_paired_ttest(matched: List[float], mismatched: List[float]) -> Tuple[float, float]:
    """Run paired t-test between matched and mismatched conditions."""
    evaluator = BiasEvaluator()
    return evaluator.run_paired_ttest(matched, mismatched)

def compute_cohen_d(matched: List[float], mismatched: List[float]) -> float:
    """Compute Cohen's d effect size between matched and mismatched conditions."""
    evaluator = BiasEvaluator()
    return evaluator.compute_cohen_d(matched, mismatched)

def measure_cot_complexity(traces: List[str]) -> float:
    """Measure the complexity of chain-of-thought reasoning traces."""
    evaluator = BiasEvaluator()
    return evaluator.measure_cot_complexity(traces)

def analyze_self_consistency(traces_per_example: Dict[str, List[str]]) -> float:
    """Analyze self-consistency across multiple reasoning traces per example."""
    evaluator = BiasEvaluator()
    return evaluator.analyze_self_consistency(traces_per_example)

# Advanced evaluation utilities
class StatisticalAnalyzer:
    """
    Advanced statistical analysis for bias evaluation.
    """
    
    def __init__(self):
        pass
    
    def bootstrap_confidence_interval(self, data: List[float], n_bootstrap: int = 1000, 
                                    confidence: float = 0.95) -> Tuple[float, float]:
        """
        Compute bootstrap confidence interval for bias scores.
        
        Args:
            data: List of bias scores
            n_bootstrap: Number of bootstrap samples
            confidence: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if not data:
            return (0.0, 0.0)
        
        bootstrap_means = []
        data_array = np.array(data)
        
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data_array, size=len(data_array), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(bootstrap_means, lower_percentile)
        upper_bound = np.percentile(bootstrap_means, upper_percentile)
        
        return float(lower_bound), float(upper_bound)
    
    def effect_size_interpretation(self, cohen_d: float) -> str:
        """
        Interpret Cohen's d effect size.
        
        Args:
            cohen_d: Cohen's d value
            
        Returns:
            Interpretation string
        """
        abs_d = abs(cohen_d)
        
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def multiple_comparisons_correction(self, p_values: List[float], 
                                      method: str = "bonferroni") -> List[float]:
        """
        Apply multiple comparisons correction to p-values.
        
        Args:
            p_values: List of p-values
            method: Correction method ('bonferroni', 'holm', 'fdr')
            
        Returns:
            List of corrected p-values
        """
        if method == "bonferroni":
            return [min(p * len(p_values), 1.0) for p in p_values]
        elif method == "holm":
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            corrected = np.zeros(len(p_values))
            
            for i, idx in enumerate(sorted_indices):
                correction_factor = len(p_values) - i
                corrected[idx] = min(p_values[idx] * correction_factor, 1.0)
                
                # Ensure monotonicity
                if i > 0:
                    prev_idx = sorted_indices[i-1]
                    corrected[idx] = max(corrected[idx], corrected[prev_idx])
            
            return corrected.tolist()
        else:
            # Default: no correction
            return p_values

class MetricsReporter:
    """
    Generate comprehensive reports of bias evaluation metrics.
    """
    
    def __init__(self):
        self.evaluator = BiasEvaluator()
        self.analyzer = StatisticalAnalyzer()
    
    def generate_summary_report(self, metrics: Dict[str, Any]) -> str:
        """
        Generate a formatted summary report.
        
        Args:
            metrics: Dictionary of metrics from compute_comprehensive_metrics
            
        Returns:
            Formatted report string
        """
        report = "# Bias Evaluation Summary Report\n\n"
        
        for strategy, strategy_metrics in metrics.items():
            report += f"## Strategy: {strategy}\n\n"
            report += f"- **Bias Score (ΔSurprisal)**: {strategy_metrics['bias_score']:.4f}\n"
            report += f"- **T-statistic**: {strategy_metrics['t_statistic']:.4f}\n"
            report += f"- **P-value**: {strategy_metrics['p_value']:.4f}\n"
            report += f"- **Cohen's d**: {strategy_metrics['cohen_d']:.4f} ({self.analyzer.effect_size_interpretation(strategy_metrics['cohen_d'])})\n"
            report += f"- **N Examples**: {strategy_metrics['n_examples']}\n"
            report += f"- **Mean Matched**: {strategy_metrics['mean_matched']:.4f} (±{strategy_metrics['std_matched']:.4f})\n"
            report += f"- **Mean Mismatched**: {strategy_metrics['mean_mismatched']:.4f} (±{strategy_metrics['std_mismatched']:.4f})\n\n"
        
        return report
    
    def export_metrics_to_csv(self, metrics: Dict[str, Any], filepath: str):
        """
        Export metrics to CSV file.
        
        Args:
            metrics: Dictionary of metrics
            filepath: Output CSV file path
        """
        rows = []
        for strategy, strategy_metrics in metrics.items():
            row = {'strategy': strategy}
            row.update(strategy_metrics)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(filepath, index=False)
        print(f"✅ Metrics exported to {filepath}")

# Export main classes and functions
__all__ = [
    'BiasEvaluator',
    'StatisticalAnalyzer', 
    'MetricsReporter',
    'compute_bias_score',
    'run_paired_ttest',
    'compute_cohen_d',
    'measure_cot_complexity',
    'analyze_self_consistency'
] 