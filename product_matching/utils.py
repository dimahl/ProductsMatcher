"""Utility classes and functions for product matching"""
import pandas as pd
from datetime import datetime

class Timer:
    """Helper class for timing operations"""
    def __init__(self):
        self.prev_metka = None
        self.reset()
    
    def reset(self):
        """Reset the timer"""
        self.prev_metka = pd.Timestamp.now()
        return self.prev_metka
    
    def delta(self):
        """Get time delta since last reset/init in seconds"""
        current = pd.Timestamp.now()
        delta = current - self.prev_metka
        self.prev_metka = current
        return delta.seconds

def safe_concat(dfs, **kwargs):
    """Safely concatenate DataFrames, handling None values"""
    valid_dfs = [df for df in dfs if df is not None and not df.empty]
    return pd.concat(valid_dfs, **kwargs) if valid_dfs else pd.DataFrame()

def safe_merge(left, right, **kwargs):
    """Safely merge DataFrames, handling None values"""
    if left is None or right is None or left.empty or right.empty:
        return pd.DataFrame()
    return pd.merge(left, right, **kwargs)
