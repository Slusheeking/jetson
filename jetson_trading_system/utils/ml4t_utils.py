"""
ML4Trading Utilities
Adapted from the proven ML4Trading repository components
Optimized for Jetson Orin performance
"""

import numpy as np
import pandas as pd
from typing import Generator, Tuple, Optional, Union, List
from sklearn.model_selection import BaseCrossValidator
from scipy.stats import spearmanr
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class MultipleTimeSeriesCV(BaseCrossValidator):
    """
    Time Series Cross-Validation for Multiple Assets
    Adapted from ML4Trading utils.py
    
    Generates tuples of train_idx, test_idx pairs
    Assumes the MultiIndex contains levels 'symbol' and 'date'
    Purges overlapping outcomes to prevent look-ahead bias
    """
    
    def __init__(self,
                 n_splits: int = 3,
                 train_period_length: int = 126,  # ~6 months of trading days
                 test_period_length: int = 21,   # ~1 month of trading days
                 lookahead: Optional[int] = None,
                 date_idx: str = 'date',
                 shuffle: bool = False):
        """
        Initialize time series cross-validator
        
        Args:
            n_splits: Number of cross-validation splits
            train_period_length: Number of periods in training set
            test_period_length: Number of periods in test set
            lookahead: Forward-looking periods to purge (prevents data leakage)
            date_idx: Name of the date index level
            shuffle: Whether to shuffle training indices
        """
        self.n_splits = n_splits
        self.lookahead = lookahead
        self.test_length = test_period_length
        self.train_length = train_period_length
        self.shuffle = shuffle
        self.date_idx = date_idx
        
        if self.lookahead is None:
            self.lookahead = 1  # Default to 1-day lookahead
    
    def split(self, X, y=None, groups=None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test splits for time series data
        
        Args:
            X: Feature matrix with MultiIndex (symbol, date)
            y: Target vector (optional)
            groups: Not used in time series split
            
        Yields:
            Tuple of (train_indices, test_indices)
        """
        # Get unique dates and sort in reverse order (most recent first)
        unique_dates = X.index.get_level_values(self.date_idx).unique()
        days = sorted(unique_dates, reverse=True)
        
        if len(days) < self.train_length + self.test_length + self.lookahead:
            raise ValueError(f"Insufficient data: need at least {self.train_length + self.test_length + self.lookahead} days, got {len(days)}")
        
        # Calculate split indices
        split_idx = []
        for i in range(self.n_splits):
            test_end_idx = i * self.test_length
            test_start_idx = test_end_idx + self.test_length
            train_end_idx = test_start_idx + self.lookahead - 1
            train_start_idx = train_end_idx + self.train_length + self.lookahead - 1
            
            # Ensure we don't go beyond available data
            if train_start_idx >= len(days):
                logger.warning(f"Split {i} exceeds available data, stopping at {i} splits")
                break
                
            split_idx.append([train_start_idx, train_end_idx, test_start_idx, test_end_idx])
        
        # Create DataFrame with reset index for easier filtering
        dates = X.reset_index()[[self.date_idx]]
        
        for train_start, train_end, test_start, test_end in split_idx:
            # Get training indices
            train_idx = dates[
                (dates[self.date_idx] > days[train_start]) & 
                (dates[self.date_idx] <= days[train_end])
            ].index
            
            # Get test indices
            test_idx = dates[
                (dates[self.date_idx] > days[test_start]) & 
                (dates[self.date_idx] <= days[test_end])
            ].index
            
            if self.shuffle:
                np.random.shuffle(list(train_idx))
            
            yield train_idx.to_numpy(), test_idx.to_numpy()
    
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Return the number of splitting iterations"""
        return self.n_splits

def information_coefficient(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Information Coefficient (IC)
    
    IC is the Spearman rank correlation between predictions and actual returns
    Used extensively in ML4Trading for evaluating signal quality
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        
    Returns:
        Information Coefficient (Spearman correlation)
    """
    try:
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() < 10:  # Need at least 10 observations
            return 0.0
            
        ic, p_value = spearmanr(y_true[mask], y_pred[mask])
        return ic if not np.isnan(ic) else 0.0
    except Exception as e:
        logger.warning(f"Error calculating IC: {e}")
        return 0.0

def ic_statistics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calculate comprehensive IC statistics
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        
    Returns:
        Dictionary with IC statistics
    """
    ic = information_coefficient(y_true, y_pred)
    
    # Calculate IC t-statistic
    n = len(y_true[~(np.isnan(y_true) | np.isnan(y_pred))])
    if n > 2:
        ic_std = np.sqrt((1 - ic**2) / (n - 2))
        ic_tstat = ic / ic_std if ic_std > 0 else 0.0
    else:
        ic_tstat = 0.0
    
    return {
        'ic': ic,
        'ic_tstat': ic_tstat,
        'observations': n,
        'abs_ic': abs(ic)
    }

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate directional accuracy (% of correct sign predictions)
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        
    Returns:
        Directional accuracy as percentage
    """
    try:
        # Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() < 10:
            return 0.0
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        # Calculate sign agreement
        correct_direction = np.sign(y_true_clean) == np.sign(y_pred_clean)
        return correct_direction.mean() * 100.0
        
    except Exception as e:
        logger.warning(f"Error calculating directional accuracy: {e}")
        return 0.0

def calculate_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate returns over specified periods
    
    Args:
        prices: Price series
        periods: Number of periods for return calculation
        
    Returns:
        Returns series
    """
    return prices.pct_change(periods=periods)

def calculate_log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """
    Calculate log returns over specified periods
    
    Args:
        prices: Price series
        periods: Number of periods for return calculation
        
    Returns:
        Log returns series
    """
    return np.log(prices / prices.shift(periods))

def winsorize_series(series: pd.Series, lower_pct: float = 0.01, upper_pct: float = 0.99) -> pd.Series:
    """
    Winsorize series by capping extreme values
    
    Args:
        series: Input series
        lower_pct: Lower percentile threshold
        upper_pct: Upper percentile threshold
        
    Returns:
        Winsorized series
    """
    lower_bound = series.quantile(lower_pct)
    upper_bound = series.quantile(upper_pct)
    
    return series.clip(lower=lower_bound, upper=upper_bound)

def standardize_series(series: pd.Series, method: str = 'zscore') -> pd.Series:
    """
    Standardize series using various methods
    
    Args:
        series: Input series
        method: 'zscore', 'minmax', or 'robust'
        
    Returns:
        Standardized series
    """
    if method == 'zscore':
        return (series - series.mean()) / series.std()
    elif method == 'minmax':
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'robust':
        median = series.median()
        mad = (series - median).abs().median()
        return (series - median) / mad
    else:
        raise ValueError(f"Unknown standardization method: {method}")

def rolling_rank(series: pd.Series, window: int, pct: bool = True) -> pd.Series:
    """
    Calculate rolling rank of series values
    
    Args:
        series: Input series
        window: Rolling window size
        pct: Return percentile ranks (0-1) instead of ranks
        
    Returns:
        Rolling rank series
    """
    return series.rolling(window=window).rank(pct=pct)

def cross_sectional_rank(df: pd.DataFrame, date_col: str = 'date', value_col: str = 'value') -> pd.DataFrame:
    """
    Calculate cross-sectional ranks within each date
    
    Args:
        df: DataFrame with date and value columns
        date_col: Name of date column
        value_col: Name of value column
        
    Returns:
        DataFrame with rank column added
    """
    df = df.copy()
    df['rank'] = df.groupby(date_col)[value_col].rank(pct=True)
    return df

def neutralize_by_factor(returns: pd.Series, factor: pd.Series, method: str = 'demean') -> pd.Series:
    """
    Neutralize returns by a factor (e.g., sector, market cap)
    
    Args:
        returns: Return series
        factor: Factor series (categorical)
        method: 'demean' or 'zscore'
        
    Returns:
        Neutralized returns
    """
    result = returns.copy()
    
    for factor_value in factor.unique():
        mask = factor == factor_value
        factor_returns = returns[mask]
        
        if method == 'demean':
            result[mask] = factor_returns - factor_returns.mean()
        elif method == 'zscore':
            result[mask] = (factor_returns - factor_returns.mean()) / factor_returns.std()
    
    return result

class PerformanceMetrics:
    """
    Calculate trading performance metrics
    Adapted from ML4Trading evaluation methods
    """
    
    @staticmethod
    def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
        """Calculate annualized Sharpe ratio"""
        excess_returns = returns - risk_free_rate / periods_per_year
        return np.sqrt(periods_per_year) * excess_returns.mean() / returns.std()
    
    @staticmethod
    def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        excess_returns = returns - risk_free_rate / periods_per_year
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std()
        
        if downside_std == 0:
            return np.inf if excess_returns.mean() > 0 else 0
        
        return np.sqrt(periods_per_year) * excess_returns.mean() / downside_std
    
    @staticmethod
    def max_drawdown(prices: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + prices.pct_change()).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = cumulative / rolling_max - 1
        return drawdown.min()
    
    @staticmethod
    def calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        annual_return = (1 + returns.mean()) ** periods_per_year - 1
        max_dd = abs(PerformanceMetrics.max_drawdown(returns))
        
        if max_dd == 0:
            return np.inf if annual_return > 0 else 0
        
        return annual_return / max_dd
    
    @staticmethod
    def win_rate(returns: pd.Series) -> float:
        """Calculate win rate (percentage of positive returns)"""
        return (returns > 0).mean() * 100
    
    @staticmethod
    def profit_factor(returns: pd.Series) -> float:
        """Calculate profit factor (gross profit / gross loss)"""
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        
        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 1.0
        
        return gross_profit / gross_loss

def format_time(t: float) -> str:
    """
    Return a formatted time string 'HH:MM:SS'
    based on a numeric time() value
    Adapted from ML4Trading utils.py
    """
    m, s = divmod(t, 60)
    h, m = divmod(m, 60)
    return f'{h:0>2.0f}:{m:0>2.0f}:{s:0>2.0f}'

def get_business_days(start_date: str, end_date: str) -> pd.DatetimeIndex:
    """
    Get business days between two dates
    
    Args:
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        
    Returns:
        DatetimeIndex of business days
    """
    return pd.bdate_range(start=start_date, end=end_date)

def align_data_to_trading_calendar(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Align data to trading calendar (business days only)
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        
    Returns:
        DataFrame aligned to business days
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Filter to business days only
    business_days_mask = df[date_col].dt.dayofweek < 5  # Monday=0, Friday=4
    return df[business_days_mask].reset_index(drop=True)

def create_forward_returns(prices_df: pd.DataFrame, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Create forward returns for multiple periods
    
    Args:
        prices_df: DataFrame with columns ['symbol', 'date', 'close']
        periods: List of forward periods to calculate
        
    Returns:
        DataFrame with forward returns columns
    """
    result_df = prices_df.copy()
    
    for period in periods:
        col_name = f'forward_return_{period}d'
        result_df[col_name] = result_df.groupby('symbol')['close'].pct_change(periods=-period)
    
    return result_df

# Jetson-specific optimizations
def optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage for Jetson Orin
    
    Args:
        df: Input DataFrame
        
    Returns:
        Memory-optimized DataFrame
    """
    df_optimized = df.copy()
    
    for col in df_optimized.columns:
        col_type = df_optimized[col].dtype
        
        if col_type == 'int64':
            # Try to downcast integers
            if df_optimized[col].min() >= 0:
                # Unsigned integers
                if df_optimized[col].max() < 255:
                    df_optimized[col] = df_optimized[col].astype('uint8')
                elif df_optimized[col].max() < 65535:
                    df_optimized[col] = df_optimized[col].astype('uint16')
                elif df_optimized[col].max() < 4294967295:
                    df_optimized[col] = df_optimized[col].astype('uint32')
            else:
                # Signed integers
                if df_optimized[col].min() > -128 and df_optimized[col].max() < 127:
                    df_optimized[col] = df_optimized[col].astype('int8')
                elif df_optimized[col].min() > -32768 and df_optimized[col].max() < 32767:
                    df_optimized[col] = df_optimized[col].astype('int16')
                elif df_optimized[col].min() > -2147483648 and df_optimized[col].max() < 2147483647:
                    df_optimized[col] = df_optimized[col].astype('int32')
        
        elif col_type == 'float64':
            # Try to downcast floats
            df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
    
    return df_optimized

# Wrapper class for backward compatibility
class ML4TradingUtils:
    """
    Utility class wrapper for ML4Trading functions
    Provides backward compatibility for imports
    """
    
    # Cross-validation
    MultipleTimeSeriesCV = MultipleTimeSeriesCV
    
    # IC calculation methods
    @staticmethod
    def information_coefficient(y_true, y_pred):
        return information_coefficient(y_true, y_pred)
    
    @staticmethod
    def ic_statistics(y_true, y_pred):
        return ic_statistics(y_true, y_pred)
    
    @staticmethod
    def directional_accuracy(y_true, y_pred):
        return directional_accuracy(y_true, y_pred)
    
    # Return calculation methods
    @staticmethod
    def calculate_returns(prices, periods=1):
        return calculate_returns(prices, periods)
    
    @staticmethod
    def calculate_log_returns(prices, periods=1):
        return calculate_log_returns(prices, periods)
    
    # Data preprocessing methods
    @staticmethod
    def winsorize_series(series, lower_pct=0.01, upper_pct=0.99):
        return winsorize_series(series, lower_pct, upper_pct)
    
    @staticmethod
    def standardize_series(series, method='zscore'):
        return standardize_series(series, method)
    
    @staticmethod
    def rolling_rank(series, window, pct=True):
        return rolling_rank(series, window, pct)
    
    @staticmethod
    def cross_sectional_rank(df, date_col='date', value_col='value'):
        return cross_sectional_rank(df, date_col, value_col)
    
    @staticmethod
    def neutralize_by_factor(returns, factor, method='demean'):
        return neutralize_by_factor(returns, factor, method)
    
    # Performance metrics class
    PerformanceMetrics = PerformanceMetrics
    
    # Utility methods
    @staticmethod
    def format_time(t):
        return format_time(t)
    
    @staticmethod
    def get_business_days(start_date, end_date):
        return get_business_days(start_date, end_date)
    
    @staticmethod
    def align_data_to_trading_calendar(df, date_col='date'):
        return align_data_to_trading_calendar(df, date_col)
    
    @staticmethod
    def create_forward_returns(prices_df, periods=[1, 5, 10, 20]):
        return create_forward_returns(prices_df, periods)
    
    @staticmethod
    def optimize_dataframe_memory(df):
        return optimize_dataframe_memory(df)