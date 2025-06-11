#!/usr/bin/env python3
"""
🆓 FREE Political Bias Evaluation Framework
Run locally with HuggingFace models - No API costs!

Usage: python run_bias_evaluation.py
"""

import sys
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import warnings
from datetime import datetime

# Add src directory to path
sys.path.append('src')

# Import our modules
from llm_helpers import LLMProber
from prompts import BiasPromptGenerator
from evaluate import BiasEvaluator

# Optional OpenAI import (not needed for free local usage)
try:
    from api_client import OpenAIClient
    print(" OpenAI integration available (optional)")
except ImportError:
    print("🆓 Using FREE local models only (no OpenAI needed)")
    OpenAIClient = None

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def main():
    print(" Starting FREE Political Bias Evaluation Framework")
    print("=" * 60)
    
    # 1. Environment Setup
    print(" Environment Setup:")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Device: {device}")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   💰 API Cost: $0.00 (100% FREE local execution)")
    print()
    
    # 2. Load Free Model
    print(" Loading FREE HuggingFace Model:")
    prober = LLMProber(
        model_name="gpt2",  # Completely free model
        device="auto"
    )
    print(f"    Model loaded: {prober.model_name}")
    print(f"    Using device: {prober.device}")
    print()
    
    # 3. Load 50-item Datasets
    print(" Loading Political Bias Datasets:")
    try:
        # Load political conflict pairs
        conflict_df = pd.read_csv('data/stimuli/political_conflict_pairs_50.csv')
        print(f"    Political conflict items: {len(conflict_df)}")
        
        # Load ideological pairs  
        ideology_df = pd.read_csv('data/stimuli/ideology_pairs_50.csv')
        print(f"    Cultural-ideological items: {len(ideology_df)}")
        
        # Preview sample item
        sample_item = conflict_df.iloc[0]
        print(f"\n    Sample item preview:")
        print(f"      Context: {sample_item['context'][:100]}...")
        print(f"      Choice 1: {sample_item['choice_1']}")
        print(f"      Choice 2: {sample_item['choice_2']}")
        print()
        
    except FileNotFoundError as e:
        print(f"    Error loading datasets: {e}")
        print("    Make sure you're in the project root directory")
        return
    
    # 4. Initialize Components
    print("🛠️  Initializing Analysis Components:")
    prompt_gen = BiasPromptGenerator()
    evaluator = BiasEvaluator()
    print("    Prompt generator ready")
    print("    Bias evaluator ready")
    print()
    
    # 5. Run Bias Analysis
    print(" Running FREE Bias Analysis:")
    
    # Choose dataset and sample size for demo
    dataset_name = "political_conflict"
    demo_df = conflict_df.head(10)  # Start with 10 items for demo
    
    print(f"    Analyzing {len(demo_df)} {dataset_name} items")
    print(f"    Prompting strategies: zero_shot, chain_of_thought, few_shot")
    print()
    
    # Run analysis
    results = []
    strategies = ['zero_shot', 'chain_of_thought', 'few_shot']
    
    for idx, row in tqdm(demo_df.iterrows(), total=len(demo_df), desc=" Evaluating"):
        context = row['context']
        choices = [row['choice_1'], row['choice_2']]
        
        for strategy in strategies:
            try:
                # Generate prompt
                prompt = prompt_gen.format_prompt(strategy, context, domain="political_conflict")
                
                # Compute surprisal (completely free!)
                surprisal = prober.compute_surprisal(prompt, choices)
                bias_score = prober.compute_bias_score(surprisal)
                
                results.append({
                    'item_id': row['id'],
                    'dataset': dataset_name,
                    'strategy': strategy,
                    'context': context,
                    'choice_1': choices[0],
                    'choice_2': choices[1],
                    'surprisal_1': surprisal[0],
                    'surprisal_2': surprisal[1],
                    'bias_score': bias_score,
                    'model': 'gpt2-free',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"        Error processing item {row['id']} with {strategy}: {e}")
                continue
    
    # 6. Process Results
    print("\n Analysis Results:")
    if results:
        results_df = pd.DataFrame(results)
        
        # Basic statistics
        print(f"    Generated {len(results_df)} evaluations")
        print(f"    Strategies tested: {results_df['strategy'].nunique()}")
        print(f"    Items processed: {results_df['item_id'].nunique()}")
        
        # Show bias score statistics by strategy
        print("\n    Bias Score Summary by Strategy:")
        bias_summary = results_df.groupby('strategy')['bias_score'].agg(['mean', 'std', 'count']).round(4)
        print(bias_summary)
        
        # Save results
        os.makedirs('data/results', exist_ok=True)
        output_file = f'data/results/bias_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(output_file, index=False)
        print(f"\n    Results saved to: {output_file}")
        
        # Show sample results
        print("\n    Sample Results:")
        sample_cols = ['item_id', 'strategy', 'bias_score', 'surprisal_1', 'surprisal_2']
        print(results_df[sample_cols].head(10).to_string(index=False))
        
    else:
        print("    No results generated")
    
    print("\n" + "=" * 60)
    print("🎉 Analysis Complete!")
    print(f"💰 Total Cost: $0.00 (100% FREE local execution)")
    print(" Framework ready for your COGS 150 final project!")
    print()
    
    # 7. Next Steps
    print(" Next Steps:")
    print("   1. Increase sample size: Change demo_df = conflict_df.head(50)")
    print("   2. Try different models: Change model_name to 'gpt2-medium'")
    print("   3. Analyze ideology dataset: Use ideology_df instead")
    print("   4. Run statistical tests: Check notebooks/02_evaluation_metrics.ipynb")
    print("   5. Create visualizations: Check notebooks/03_visualizations.ipynb")

if __name__ == "__main__":
    main() 