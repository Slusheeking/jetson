"""
Advanced Position Sizing for Jetson Trading System
ML4Trading-based position sizing with Kelly Criterion and risk parity
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import scipy.optimize as optimize

from ..config.jetson_settings import JetsonConfig
from ..config.trading_params import TradingConfig
from ..utils.logger import get_risk_logger
from ..utils.database import TradingDatabase

class SizingMethod(Enum):
    """Position sizing methods"""
    FIXED_DOLLAR = "fixed_dollar"
    FIXED_PERCENTAGE = "fixed_percentage"
    VOLATILITY_TARGET = "volatility_target"
    KELLY_CRITERION = "kelly_criterion"
    RISK_PARITY = "risk_parity"
    ML_CONFIDENCE = "ml_confidence"
    ADAPTIVE = "adaptive"

@dataclass
class SizingInput:
    """Input data for position sizing"""
    symbol: str
    current_price: float
    signal_probability: float
    signal_strength: float
    volatility: float
    expected_return: float
    portfolio_value: float
    current_position: float
    risk_free_rate: float = 0.02
    max_position_pct: float = 0.15
    target_volatility: float = 0.15

@dataclass
class SizingOutput:
    """Output from position sizing calculation"""
    recommended_shares: int
    recommended_value: float
    position_percentage: float
    sizing_method: SizingMethod
    risk_contribution: float
    expected_return: float
    expected_volatility: float
    kelly_fraction: float
    confidence_score: float
    reasoning: str

class PositionSizer:
    """
    Advanced position sizing system with multiple methodologies
    Optimized for ML4Trading signals and Jetson hardware constraints
    """
    
    def __init__(self, 
                 default_method: SizingMethod = SizingMethod.ADAPTIVE,
                 portfolio_target_vol: float = 0.15,
                 kelly_multiplier: float = 0.25,
                 min_position_value: float = 1000.0):
        """
        Initialize position sizer
        
        Args:
            default_method: Default sizing method
            portfolio_target_vol: Target portfolio volatility
            kelly_multiplier: Kelly fraction multiplier for safety
            min_position_value: Minimum position value
        """
        self.default_method = default_method
        self.portfolio_target_vol = portfolio_target_vol
        self.kelly_multiplier = kelly_multiplier
        self.min_position_value = min_position_value
        
        self.logger = get_risk_logger()
        self.db_manager = TradingDatabase()
        
        # Sizing parameters
        self.sizing_params = {
            'lookback_days': 252,
            'min_observations': 30,
            'confidence_threshold': 0.6,
            'volatility_floor': 0.05,
            'volatility_ceiling': 1.0,
            'return_floor': -0.5,
            'return_ceiling': 2.0,
            'correlation_window': 60,
            'rebalance_threshold': 0.1
        }
        
        # Performance tracking
        self.sizing_history = []
        self.performance_metrics = {}
        
        self.logger.info(f"PositionSizer initialized with method: {default_method.value}")
    
    def calculate_position_size(self, 
                              sizing_input: SizingInput,
                              method: SizingMethod = None) -> SizingOutput:
        """
        Calculate optimal position size using specified method
        
        Args:
            sizing_input: Input parameters for sizing
            method: Sizing method (uses default if None)
            
        Returns:
            Position sizing recommendation
        """
        try:
            method = method or self.default_method
            
            # Validate inputs
            if not self._validate_inputs(sizing_input):
                return self._create_zero_position(sizing_input, method, "Invalid inputs")
            
            # Calculate based on method
            if method == SizingMethod.FIXED_DOLLAR:
                result = self._fixed_dollar_sizing(sizing_input)
            elif method == SizingMethod.FIXED_PERCENTAGE:
                result = self._fixed_percentage_sizing(sizing_input)
            elif method == SizingMethod.VOLATILITY_TARGET:
                result = self._volatility_target_sizing(sizing_input)
            elif method == SizingMethod.KELLY_CRITERION:
                result = self._kelly_criterion_sizing(sizing_input)
            elif method == SizingMethod.RISK_PARITY:
                result = self._risk_parity_sizing(sizing_input)
            elif method == SizingMethod.ML_CONFIDENCE:
                result = self._ml_confidence_sizing(sizing_input)
            elif method == SizingMethod.ADAPTIVE:
                result = self._adaptive_sizing(sizing_input)
            else:
                result = self._fixed_percentage_sizing(sizing_input)
            
            # Apply position limits and constraints
            result = self._apply_constraints(result, sizing_input)
            
            # Record sizing decision
            self._record_sizing_decision(sizing_input, result)
            
            self.logger.info(f"Position size for {sizing_input.symbol}: {result.recommended_shares} shares (${result.recommended_value:,.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {sizing_input.symbol}: {e}")
            return self._create_zero_position(sizing_input, method, f"Error: {e}")
    
    def _validate_inputs(self, sizing_input: SizingInput) -> bool:
        """Validate sizing input parameters"""
        try:
            # Check required fields
            if sizing_input.current_price <= 0:
                return False
            if sizing_input.portfolio_value <= 0:
                return False
            if not (0 <= sizing_input.signal_probability <= 1):
                return False
            if sizing_input.volatility < 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _fixed_dollar_sizing(self, sizing_input: SizingInput) -> SizingOutput:
        """Fixed dollar amount position sizing"""
        fixed_amount = TradingConfig.DEFAULT_POSITION_SIZE
        shares = int(fixed_amount / sizing_input.current_price)
        value = shares * sizing_input.current_price
        
        return SizingOutput(
            recommended_shares=shares,
            recommended_value=value,
            position_percentage=value / sizing_input.portfolio_value,
            sizing_method=SizingMethod.FIXED_DOLLAR,
            risk_contribution=value * sizing_input.volatility / np.sqrt(252),
            expected_return=sizing_input.expected_return,
            expected_volatility=sizing_input.volatility,
            kelly_fraction=0.0,
            confidence_score=0.5,
            reasoning=f"Fixed ${fixed_amount:,.0f} position size"
        )
    
    def _fixed_percentage_sizing(self, sizing_input: SizingInput) -> SizingOutput:
        """Fixed percentage of portfolio position sizing"""
        target_pct = 0.05  # 5% default
        target_value = sizing_input.portfolio_value * target_pct
        shares = int(target_value / sizing_input.current_price)
        actual_value = shares * sizing_input.current_price
        
        return SizingOutput(
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_value / sizing_input.portfolio_value,
            sizing_method=SizingMethod.FIXED_PERCENTAGE,
            risk_contribution=actual_value * sizing_input.volatility / np.sqrt(252),
            expected_return=sizing_input.expected_return,
            expected_volatility=sizing_input.volatility,
            kelly_fraction=0.0,
            confidence_score=0.5,
            reasoning=f"Fixed {target_pct:.1%} of portfolio"
        )
    
    def _volatility_target_sizing(self, sizing_input: SizingInput) -> SizingOutput:
        """Volatility targeting position sizing"""
        # Target position volatility contribution
        target_position_vol = self.portfolio_target_vol * 0.3  # 30% of portfolio vol budget
        
        # Calculate position size to achieve target volatility
        daily_vol = sizing_input.volatility / np.sqrt(252)
        target_value = sizing_input.portfolio_value * target_position_vol / daily_vol
        
        shares = int(target_value / sizing_input.current_price)
        actual_value = shares * sizing_input.current_price
        
        return SizingOutput(
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_value / sizing_input.portfolio_value,
            sizing_method=SizingMethod.VOLATILITY_TARGET,
            risk_contribution=actual_value * sizing_input.volatility / np.sqrt(252),
            expected_return=sizing_input.expected_return,
            expected_volatility=sizing_input.volatility,
            kelly_fraction=0.0,
            confidence_score=0.7,
            reasoning=f"Targeting {target_position_vol:.1%} position volatility"
        )
    
    def _kelly_criterion_sizing(self, sizing_input: SizingInput) -> SizingOutput:
        """Kelly Criterion position sizing"""
        try:
            # Kelly formula: f = (bp - q) / b
            # Where: b = odds received, p = probability of win, q = probability of loss
            
            win_prob = sizing_input.signal_probability
            loss_prob = 1 - win_prob
            
            # Estimate win/loss ratios from expected return and volatility
            if sizing_input.expected_return <= 0:
                kelly_fraction = 0.0
            else:
                # Simplified Kelly calculation
                excess_return = sizing_input.expected_return - sizing_input.risk_free_rate
                kelly_fraction = excess_return / (sizing_input.volatility ** 2)
                
                # Apply probability adjustment
                kelly_fraction *= win_prob
                
                # Apply safety multiplier
                kelly_fraction *= self.kelly_multiplier
            
            # Convert to position value
            target_value = sizing_input.portfolio_value * abs(kelly_fraction)
            shares = int(target_value / sizing_input.current_price)
            actual_value = shares * sizing_input.current_price
            
            return SizingOutput(
                recommended_shares=shares,
                recommended_value=actual_value,
                position_percentage=actual_value / sizing_input.portfolio_value,
                sizing_method=SizingMethod.KELLY_CRITERION,
                risk_contribution=actual_value * sizing_input.volatility / np.sqrt(252),
                expected_return=sizing_input.expected_return,
                expected_volatility=sizing_input.volatility,
                kelly_fraction=kelly_fraction,
                confidence_score=win_prob,
                reasoning=f"Kelly fraction: {kelly_fraction:.3f} (safety-adjusted)"
            )
            
        except Exception as e:
            self.logger.error(f"Kelly criterion calculation error: {e}")
            return self._fixed_percentage_sizing(sizing_input)
    
    def _risk_parity_sizing(self, sizing_input: SizingInput) -> SizingOutput:
        """Risk parity position sizing"""
        # Inverse volatility weighting for risk parity
        vol_floor = max(sizing_input.volatility, self.sizing_params['volatility_floor'])
        
        # Calculate weight inversely proportional to volatility
        risk_weight = 1.0 / vol_floor
        
        # Normalize to reasonable position size (assume average portfolio volatility)
        avg_portfolio_vol = 0.15
        normalized_weight = risk_weight * avg_portfolio_vol / 10  # Scale factor
        
        target_value = sizing_input.portfolio_value * min(normalized_weight, sizing_input.max_position_pct)
        shares = int(target_value / sizing_input.current_price)
        actual_value = shares * sizing_input.current_price
        
        return SizingOutput(
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_value / sizing_input.portfolio_value,
            sizing_method=SizingMethod.RISK_PARITY,
            risk_contribution=actual_value * sizing_input.volatility / np.sqrt(252),
            expected_return=sizing_input.expected_return,
            expected_volatility=sizing_input.volatility,
            kelly_fraction=0.0,
            confidence_score=0.6,
            reasoning=f"Risk parity weight: {normalized_weight:.3f}"
        )
    
    def _ml_confidence_sizing(self, sizing_input: SizingInput) -> SizingOutput:
        """ML confidence-based position sizing"""
        # Base size on signal strength and probability
        base_pct = 0.08  # 8% base allocation
        
        # Confidence multiplier based on signal probability
        confidence_multiplier = sizing_input.signal_probability
        
        # Strength multiplier based on signal strength
        strength_multiplier = sizing_input.signal_strength
        
        # Combined multiplier with diminishing returns
        combined_multiplier = np.sqrt(confidence_multiplier * strength_multiplier)
        
        target_pct = base_pct * combined_multiplier
        target_value = sizing_input.portfolio_value * target_pct
        
        shares = int(target_value / sizing_input.current_price)
        actual_value = shares * sizing_input.current_price
        
        return SizingOutput(
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_value / sizing_input.portfolio_value,
            sizing_method=SizingMethod.ML_CONFIDENCE,
            risk_contribution=actual_value * sizing_input.volatility / np.sqrt(252),
            expected_return=sizing_input.expected_return,
            expected_volatility=sizing_input.volatility,
            kelly_fraction=0.0,
            confidence_score=sizing_input.signal_probability,
            reasoning=f"ML confidence: {confidence_multiplier:.3f}, strength: {strength_multiplier:.3f}"
        )
    
    def _adaptive_sizing(self, sizing_input: SizingInput) -> SizingOutput:
        """Adaptive sizing combining multiple methods"""
        # Calculate sizes using different methods
        methods = [
            SizingMethod.VOLATILITY_TARGET,
            SizingMethod.KELLY_CRITERION,
            SizingMethod.ML_CONFIDENCE
        ]
        
        results = []
        for method in methods:
            if method == SizingMethod.VOLATILITY_TARGET:
                result = self._volatility_target_sizing(sizing_input)
            elif method == SizingMethod.KELLY_CRITERION:
                result = self._kelly_criterion_sizing(sizing_input)
            elif method == SizingMethod.ML_CONFIDENCE:
                result = self._ml_confidence_sizing(sizing_input)
            
            results.append(result)
        
        # Weight methods based on signal quality
        weights = [
            0.3,  # Volatility target
            0.3 if sizing_input.expected_return > 0 else 0.1,  # Kelly (higher weight if positive expected return)
            0.4 if sizing_input.signal_probability > 0.6 else 0.2  # ML confidence (higher weight if high confidence)
        ]
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average of position sizes
        weighted_shares = sum(w * r.recommended_shares for w, r in zip(weights, results))
        shares = int(weighted_shares)
        actual_value = shares * sizing_input.current_price
        
        # Weighted average of other metrics
        weighted_confidence = sum(w * r.confidence_score for w, r in zip(weights, results))
        
        return SizingOutput(
            recommended_shares=shares,
            recommended_value=actual_value,
            position_percentage=actual_value / sizing_input.portfolio_value,
            sizing_method=SizingMethod.ADAPTIVE,
            risk_contribution=actual_value * sizing_input.volatility / np.sqrt(252),
            expected_return=sizing_input.expected_return,
            expected_volatility=sizing_input.volatility,
            kelly_fraction=results[1].kelly_fraction,  # From Kelly method
            confidence_score=weighted_confidence,
            reasoning=f"Adaptive blend: Vol {weights[0]:.1%}, Kelly {weights[1]:.1%}, ML {weights[2]:.1%}"
        )
    
    def _apply_constraints(self, result: SizingOutput, sizing_input: SizingInput) -> SizingOutput:
        """Apply position size constraints and limits"""
        original_shares = result.recommended_shares
        
        # Minimum position size
        min_shares = int(self.min_position_value / sizing_input.current_price)
        if result.recommended_shares > 0 and result.recommended_shares < min_shares:
            result.recommended_shares = 0
            result.recommended_value = 0
            result.reasoning += f" | Below minimum ${self.min_position_value:,.0f}"
        
        # Maximum position percentage
        max_value = sizing_input.portfolio_value * sizing_input.max_position_pct
        if result.recommended_value > max_value:
            result.recommended_shares = int(max_value / sizing_input.current_price)
            result.recommended_value = result.recommended_shares * sizing_input.current_price
            result.reasoning += f" | Capped at {sizing_input.max_position_pct:.1%}"
        
        # Update position percentage
        result.position_percentage = result.recommended_value / sizing_input.portfolio_value
        
        # Update risk contribution
        result.risk_contribution = result.recommended_value * sizing_input.volatility / np.sqrt(252)
        
        if result.recommended_shares != original_shares:
            self.logger.info(f"Position size adjusted from {original_shares} to {result.recommended_shares} shares")
        
        return result
    
    def _create_zero_position(self, sizing_input: SizingInput, method: SizingMethod, reason: str) -> SizingOutput:
        """Create zero position output"""
        return SizingOutput(
            recommended_shares=0,
            recommended_value=0.0,
            position_percentage=0.0,
            sizing_method=method,
            risk_contribution=0.0,
            expected_return=0.0,
            expected_volatility=0.0,
            kelly_fraction=0.0,
            confidence_score=0.0,
            reasoning=reason
        )
    
    def _record_sizing_decision(self, sizing_input: SizingInput, result: SizingOutput):
        """Record sizing decision for analysis"""
        record = {
            'timestamp': datetime.now().isoformat(),
            'symbol': sizing_input.symbol,
            'sizing_input': sizing_input.__dict__,
            'sizing_output': result.__dict__
        }
        
        self.sizing_history.append(record)
        
        # Keep only recent history
        if len(self.sizing_history) > 1000:
            self.sizing_history = self.sizing_history[-1000:]
    
    def get_portfolio_allocation(self, 
                               symbols: List[str],
                               signals: Dict[str, Dict[str, float]],
                               portfolio_value: float,
                               current_positions: Dict[str, float] = None) -> Dict[str, SizingOutput]:
        """
        Calculate optimal portfolio allocation across multiple symbols
        
        Args:
            symbols: List of symbols to consider
            signals: Dictionary of signal data per symbol
            portfolio_value: Total portfolio value
            current_positions: Current position sizes
            
        Returns:
            Dictionary of sizing recommendations per symbol
        """
        try:
            current_positions = current_positions or {}
            allocations = {}
            
            # Get market data for all symbols
            market_data = {}
            for symbol in symbols:
                try:
                    data = self._get_symbol_data(symbol)
                    if data:
                        market_data[symbol] = data
                except Exception as e:
                    self.logger.warning(f"Could not get data for {symbol}: {e}")
            
            # Calculate individual allocations
            for symbol in symbols:
                if symbol not in signals or symbol not in market_data:
                    continue
                
                signal_data = signals[symbol]
                symbol_data = market_data[symbol]
                
                # Create sizing input
                sizing_input = SizingInput(
                    symbol=symbol,
                    current_price=symbol_data['price'],
                    signal_probability=signal_data.get('probability', 0.5),
                    signal_strength=signal_data.get('strength', 0.5),
                    volatility=symbol_data['volatility'],
                    expected_return=signal_data.get('expected_return', 0.0),
                    portfolio_value=portfolio_value,
                    current_position=current_positions.get(symbol, 0.0)
                )
                
                allocation = self.calculate_position_size(sizing_input)
                allocations[symbol] = allocation
            
            # Normalize allocations if total exceeds portfolio
            total_allocation = sum(a.recommended_value for a in allocations.values())
            if total_allocation > portfolio_value * 0.95:  # Max 95% allocation
                scale_factor = (portfolio_value * 0.95) / total_allocation
                
                for symbol, allocation in allocations.items():
                    allocation.recommended_shares = int(allocation.recommended_shares * scale_factor)
                    allocation.recommended_value = allocation.recommended_shares * market_data[symbol]['price']
                    allocation.position_percentage = allocation.recommended_value / portfolio_value
                    allocation.reasoning += f" | Scaled by {scale_factor:.3f}"
            
            return allocations
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio allocation: {e}")
            return {}
    
    def _get_symbol_data(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get symbol data for sizing calculations"""
        try:
            # Get recent price data
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
            
            price_data = self.db_manager.get_market_data(symbol, start_date, end_date)
            if price_data is None or price_data.empty:
                return None
            
            # Calculate metrics
            returns = price_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            current_price = price_data['close'].iloc[-1]
            
            return {
                'price': current_price,
                'volatility': max(volatility, self.sizing_params['volatility_floor']),
                'returns': returns.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting symbol data for {symbol}: {e}")
            return None
    
    def get_sizing_performance(self) -> Dict[str, Any]:
        """Get performance metrics for sizing decisions"""
        if not self.sizing_history:
            return {}
        
        try:
            # Convert to DataFrame for analysis
            df = pd.DataFrame([
                {
                    'timestamp': h['timestamp'],
                    'symbol': h['symbol'],
                    'method': h['sizing_output']['sizing_method'],
                    'shares': h['sizing_output']['recommended_shares'],
                    'value': h['sizing_output']['recommended_value'],
                    'position_pct': h['sizing_output']['position_percentage'],
                    'confidence': h['sizing_output']['confidence_score']
                }
                for h in self.sizing_history
            ])
            
            # Calculate metrics
            total_decisions = len(df)
            avg_position_size = df['value'].mean()
            avg_position_pct = df['position_pct'].mean()
            method_distribution = df['method'].value_counts().to_dict()
            avg_confidence = df['confidence'].mean()
            
            return {
                'total_decisions': total_decisions,
                'avg_position_size': avg_position_size,
                'avg_position_percentage': avg_position_pct,
                'method_distribution': method_distribution,
                'avg_confidence': avg_confidence,
                'recent_decisions': self.sizing_history[-10:]
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating sizing performance: {e}")
            return {}

# Example usage
if __name__ == "__main__":
    print("--- Running PositionSizer Demo ---")
    
    # Initialize the sizer
    sizer = PositionSizer()
    
    # --- Test Case 1: High Confidence Buy Signal ---
    print("\n--- Test Case 1: High Confidence Buy Signal (AAPL) ---")
    sizing_input_1 = SizingInput(
        symbol="AAPL",
        current_price=175.0,
        signal_probability=0.75,
        signal_strength=0.8,
        volatility=0.22,
        expected_return=0.15,
        portfolio_value=100000.0,
        current_position=0.0
    )
    
    # Calculate size using the default adaptive method
    result_1 = sizer.calculate_position_size(sizing_input_1)
    print(f"Method: {result_1.sizing_method.value}")
    print(f"Recommended position: {result_1.recommended_shares} shares (${result_1.recommended_value:,.2f})")
    print(f"Reasoning: {result_1.reasoning}")
    
    # --- Test Case 2: Low Volatility, Moderate Signal ---
    print("\n--- Test Case 2: Low Volatility, Moderate Signal (MSFT) ---")
    sizing_input_2 = SizingInput(
        symbol="MSFT",
        current_price=350.0,
        signal_probability=0.60,
        signal_strength=0.65,
        volatility=0.15, # Lower volatility
        expected_return=0.10,
        portfolio_value=100000.0,
        current_position=0.0
    )
    
    # Test a specific method: Volatility Targeting
    result_2 = sizer.calculate_position_size(sizing_input_2, method=SizingMethod.VOLATILITY_TARGET)
    print(f"Method: {result_2.sizing_method.value}")
    print(f"Recommended position: {result_2.recommended_shares} shares (${result_2.recommended_value:,.2f})")
    print(f"Reasoning: {result_2.reasoning}")

    # --- Test Case 3: High Volatility, Risky Bet ---
    print("\n--- Test Case 3: High Volatility, Risky Bet (TSLA) ---")
    sizing_input_3 = SizingInput(
        symbol="TSLA",
        current_price=250.0,
        signal_probability=0.55,
        signal_strength=0.9, # High strength but low prob
        volatility=0.45, # High volatility
        expected_return=0.25,
        portfolio_value=100000.0,
        current_position=0.0
    )

    # Test Kelly Criterion
    result_3 = sizer.calculate_position_size(sizing_input_3, method=SizingMethod.KELLY_CRITERION)
    print(f"Method: {result_3.sizing_method.value}")
    print(f"Recommended position: {result_3.recommended_shares} shares (${result_3.recommended_value:,.2f})")
    print(f"Reasoning: {result_3.reasoning}")
    
    # --- Test Case 4: Full Portfolio Allocation ---
    print("\n--- Test Case 4: Full Portfolio Allocation (Multiple Stocks) ---")
    signals_data = {
        "AAPL": {"probability": 0.7, "strength": 0.8, "expected_return": 0.15},
        "MSFT": {"probability": 0.6, "strength": 0.65, "expected_return": 0.10},
        "GOOGL": {"probability": 0.65, "strength": 0.7, "expected_return": 0.12},
    }
    
    # get_portfolio_allocation requires live data, so we run a simplified version here
    # In a live run, it would fetch from the database
    print("Demonstrating multi-symbol allocation concept (requires live DB for full run)")
    
    
    # --- Test Case 5: Sizing with an existing position (rebalancing) ---
    print("\n--- Test Case 5: Rebalancing an Existing Position ---")
    sizing_input_5 = SizingInput(
        symbol="AAPL",
        current_price=180.0, # Price moved up
        signal_probability=0.70,
        signal_strength=0.75,
        volatility=0.23,
        expected_return=0.13,
        portfolio_value=110000.0, # Portfolio grew
        current_position=result_1.recommended_value # From first test case
    )
    
    result_5 = sizer.calculate_position_size(sizing_input_5)
    print("Original position was:", result_1.recommended_shares, "shares")
    print(f"New Recommended position: {result_5.recommended_shares} shares (${result_5.recommended_value:,.2f})")
    print(f"Change in shares: {result_5.recommended_shares - result_1.recommended_shares}")
    print(f"Reasoning: {result_5.reasoning}")

    print("\n--- PositionSizer Demo Complete ---")
