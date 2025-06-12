"""
Data loading utilities for the LLM Political Bias Evaluation Framework.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Unified data loader for political bias evaluation datasets.
    Handles loading and preprocessing of both political conflict and
    cultural-ideological datasets.
    """
    
    def __init__(self, data_dir: str = '../data/stimuli'):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Path to the directory containing stimulus datasets
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            # Try relative to current directory if ../data doesn't exist
            self.data_dir = Path('data/stimuli')
            if not self.data_dir.exists():
                raise FileNotFoundError(f"Data directory not found at {data_dir} or data/stimuli")
    
    def load_political_data(self) -> pd.DataFrame:
        """
        Load the political conflict dataset.
        
        Returns:
            DataFrame containing political conflict stimulus pairs
        """
        try:
            df = pd.read_csv(self.data_dir / 'political_conflict_pairs_50.csv')
            logger.info(f"Loaded {len(df)} political conflict items")
            return df
        except Exception as e:
            logger.error(f"Error loading political data: {e}")
            raise
    
    def load_ideological_data(self) -> pd.DataFrame:
        """
        Load the cultural-ideological dataset.
        
        Returns:
            DataFrame containing cultural-ideological stimulus pairs
        """
        try:
            df = pd.read_csv(self.data_dir / 'ideology_pairs.csv')
            logger.info(f"Loaded {len(df)} cultural-ideological items")
            return df
        except Exception as e:
            logger.error(f"Error loading ideological data: {e}")
            raise
    
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both political conflict and cultural-ideological datasets.
        
        Returns:
            Tuple of (political_df, ideology_df)
        """
        return self.load_political_data(), self.load_ideological_data()
    
    def prepare_unified_stimuli(self, max_samples: Optional[int] = None) -> List[Dict]:
        """
        Convert both datasets to unified format for evaluation.
        
        Args:
            max_samples: Optional limit on number of samples per dataset
            
        Returns:
            List of unified stimulus dictionaries
        """
        unified_stimuli = []
        
        # Load both datasets
        political_df = self.load_political_data()
        ideology_df = self.load_ideological_data()
        
        # Limit samples if specified
        if max_samples:
            political_df = political_df.head(max_samples)
            ideology_df = ideology_df.head(max_samples)
        
        # Process political conflict data
        for _, row in political_df.iterrows():
            stimulus = {
                'id': f"political_{row['id']}",
                'dataset': 'political_conflict',
                'context': row['context'],
                'option_a': row['choice_1'],  # Pro-Palestinian framing
                'option_b': row['choice_2'],  # Pro-Israeli framing
                'category': 'Gaza_conflict'
            }
            unified_stimuli.append(stimulus)
        
        # Process cultural-ideological data
        for _, row in ideology_df.iterrows():
            stimulus = {
                'id': f"ideology_{row['pair_id']}",
                'dataset': 'cultural_ideological',
                'context': row['context'],
                'option_a': row['option_a'],  # Religious framing
                'option_b': row['option_b'],  # Secular framing
                'category': row['category']
            }
            unified_stimuli.append(stimulus)
        
        return unified_stimuli 