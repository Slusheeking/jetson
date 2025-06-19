"""
Risk Management System for Jetson Trading
Advanced risk controls and position sizing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timedelta
import threading
from dataclasses import dataclass, asdict
from enum import Enum

from ..config.jetson_settings import JetsonConfig
from ..config.trading_params import TradingConfig
from ..utils.logger import get_risk_logger
from ..utils.database import TradingDatabase

class RiskLevel(Enum):
    """Risk levels for positions and portfolio"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class RiskMetrics:
    """Container for risk metrics"""
    portfolio_value: float
    total_exposure: float
    leverage: float
    var_1d: float  # Value at Risk 1 day
    var_5d: float  # Value at Risk 5 days
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_spy: float
    concentration_risk: float
    sector_exposure: Dict[str, float]
    risk_level: RiskLevel
    violation_count: int
    last_updated: datetime

@dataclass
class PositionRisk:
    """Risk metrics for individual position"""
    symbol: str
    position_size: float
    market_value: float
    unrealized_pnl: float
    daily_var: float
    volatility: float
    beta: float
    correlation_spy: float
    max_position_size: float
    risk_score: float
    risk_level: RiskLevel
    days_held: int
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None

class RiskManager:
    """
    Comprehensive risk management system
    Implements portfolio-level and position-level risk controls
    """
    
    def __init__(self, 
                 initial_capital: float = 25000.0,
                 max_portfolio_risk: float = 0.02,  # 2% daily VaR
                 max_position_risk: float = 0.01,   # 1% per position
                 max_leverage: float = 1.0):
        """
        Initialize risk manager
        
        Args:
            initial_capital: Starting capital
            max_portfolio_risk: Maximum portfolio daily VaR
            max_position_risk: Maximum position daily VaR
            max_leverage: Maximum allowed leverage
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_portfolio_risk = max_portfolio_risk
        self.max_position_risk = max_position_risk
        self.max_leverage = max_leverage
        
        self.logger = get_risk_logger()
        self.db_manager = TradingDatabase()
        
        # Risk state
        self.positions = {}  # symbol -> PositionRisk
        self.portfolio_metrics = None
        self.risk_violations = []
        self.circuit_breaker_active = False
        self.daily_loss_limit = initial_capital * 0.05  # 5% daily loss limit
        
        # Threading
        self.risk_lock = threading.Lock()
        
        # Risk parameters
        self.risk_params = {
            'confidence_level': 0.95,  # VaR confidence level
            'lookback_days': 252,      # Days for volatility calculation
            'rebalance_threshold': 0.1, # Portfolio rebalance threshold
            'correlation_threshold': 0.7, # High correlation threshold
            'concentration_limit': 0.15,  # Max 15% in single position
            'sector_limit': 0.30,         # Max 30% in single sector
            'max_drawdown_limit': 0.20,   # Max 20% drawdown
        }
        
        # Market data for risk calculations
        self.market_data = {}  # symbol -> price data
        self.spy_data = None   # SPY data for beta calculation
        
        self.logger.info(f"RiskManager initialized with ${initial_capital:,.2f} capital")
    
    def update_position(self, 
                       symbol: str,
                       quantity: float,
                       price: float,
                       action: str = "update") -> bool:
        """
        Update position and recalculate risk metrics
        
        Args:
            symbol: Stock symbol
            quantity: Position quantity (shares)
            price: Current price
            action: Action type (buy, sell, update)
            
        Returns:
            True if position update is within risk limits
        """
        try:
            with self.risk_lock:
                # Calculate position metrics
                market_value = abs(quantity) * price
                
                # Get position risk metrics
                position_risk = self._calculate_position_risk(symbol, quantity, price)
                
                if position_risk is None:
                    self.logger.error(f"Cannot calculate risk for {symbol}")
                    return False
                
                # Check position-level risk limits
                if not self._check_position_limits(position_risk):
                    self.logger.warning(f"Position risk limits exceeded for {symbol}")
                    return False
                
                # Update position
                self.positions[symbol] = position_risk
                
                # Update portfolio metrics
                self._update_portfolio_metrics()
                
                # Check portfolio-level risk limits
                if not self._check_portfolio_limits():
                    self.logger.warning("Portfolio risk limits exceeded")
                    return False
                
                self.logger.info(f"Position updated: {symbol} {quantity}@${price:.2f} (Risk: {position_risk.risk_level.value})")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating position {symbol}: {e}")
            return False
    
    def _calculate_position_risk(self, 
                               symbol: str,
                               quantity: float,
                               price: float) -> Optional[PositionRisk]:
        """Calculate comprehensive risk metrics for a position"""
        try:
            # Get market data for volatility calculation
            market_data = self._get_market_data(symbol)
            if market_data is None:
                return None
            
            # Calculate returns and volatility
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate beta vs SPY
            beta = self._calculate_beta(symbol, returns)
            
            # Calculate correlation with SPY
            correlation_spy = self._calculate_correlation_spy(returns)
            
            # Calculate Value at Risk
            daily_var = abs(quantity) * price * volatility / np.sqrt(252) * 1.96  # 95% confidence
            
            # Calculate risk score (0-1 scale)
            volatility_score = min(volatility / 0.5, 1.0)  # Normalize by 50% vol
            concentration_score = min(abs(quantity) * price / self.current_capital / 0.1, 1.0)
            beta_score = min(abs(beta) / 2.0, 1.0)
            
            risk_score = (volatility_score + concentration_score + beta_score) / 3
            
            # Determine risk level
            if risk_score >= 0.8:
                risk_level = RiskLevel.CRITICAL
            elif risk_score >= 0.6:
                risk_level = RiskLevel.HIGH
            elif risk_score >= 0.4:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            # Calculate maximum position size based on risk
            max_position_value = self.current_capital * self.max_position_risk / (volatility / np.sqrt(252))
            max_position_size = max_position_value / price
            
            # Calculate unrealized P&L (simplified)
            unrealized_pnl = 0.0  # Would need entry price for accurate calculation
            
            # Calculate days held (simplified)
            days_held = 0  # Would need position entry date
            
            return PositionRisk(
                symbol=symbol,
                position_size=quantity,
                market_value=abs(quantity) * price,
                unrealized_pnl=unrealized_pnl,
                daily_var=daily_var,
                volatility=volatility,
                beta=beta,
                correlation_spy=correlation_spy,
                max_position_size=max_position_size,
                risk_score=risk_score,
                risk_level=risk_level,
                days_held=days_held
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating position risk for {symbol}: {e}")
            return None
    
    def _get_market_data(self, symbol: str, days: int = 252) -> Optional[pd.DataFrame]:
        """Get market data for risk calculations"""
        try:
            if symbol not in self.market_data:
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                
                data = self.db_manager.get_price_data(symbol, start_date, end_date)
                if data is not None and not data.empty:
                    self.market_data[symbol] = data
                else:
                    return None
            
            return self.market_data[symbol]
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def _calculate_beta(self, symbol: str, returns: pd.Series) -> float:
        """Calculate beta vs SPY"""
        try:
            # Get SPY data
            if self.spy_data is None:
                spy_data = self._get_market_data('SPY')
                if spy_data is not None:
                    self.spy_data = spy_data['close'].pct_change().dropna()
            
            if self.spy_data is None:
                return 1.0  # Default beta
            
            # Align dates
            aligned_data = pd.concat([returns, self.spy_data], axis=1, join='inner')
            if len(aligned_data) < 20:
                return 1.0
            
            stock_returns = aligned_data.iloc[:, 0]
            market_returns = aligned_data.iloc[:, 1]
            
            # Calculate beta
            covariance = np.cov(stock_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            
            beta = covariance / market_variance if market_variance > 0 else 1.0
            
            return beta
            
        except Exception as e:
            self.logger.error(f"Error calculating beta for {symbol}: {e}")
            return 1.0
    
    def _calculate_correlation_spy(self, returns: pd.Series) -> float:
        """Calculate correlation with SPY"""
        try:
            if self.spy_data is None:
                return 0.0
            
            # Align dates
            aligned_data = pd.concat([returns, self.spy_data], axis=1, join='inner')
            if len(aligned_data) < 20:
                return 0.0
            
            correlation = aligned_data.iloc[:, 0].corr(aligned_data.iloc[:, 1])
            return correlation if not np.isnan(correlation) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation: {e}")
            return 0.0
    
    def _check_position_limits(self, position_risk: PositionRisk) -> bool:
        """Check if position is within risk limits"""
        violations = []
        
        # Check position size limit
        position_pct = position_risk.market_value / self.current_capital
        if position_pct > self.risk_params['concentration_limit']:
            violations.append(f"Position concentration {position_pct:.1%} exceeds limit {self.risk_params['concentration_limit']:.1%}")
        
        # Check daily VaR limit
        var_pct = position_risk.daily_var / self.current_capital
        if var_pct > self.max_position_risk:
            violations.append(f"Position VaR {var_pct:.1%} exceeds limit {self.max_position_risk:.1%}")
        
        # Check risk level
        if position_risk.risk_level == RiskLevel.CRITICAL:
            violations.append("Position risk level is CRITICAL")
        
        if violations:
            self.risk_violations.extend(violations)
            self.logger.warning(f"Position limit violations for {position_risk.symbol}: {violations}")
            return False
        
        return True
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level risk metrics"""
        try:
            if not self.positions:
                return
            
            # Calculate portfolio values
            total_value = sum(pos.market_value for pos in self.positions.values())
            total_var = np.sqrt(sum(pos.daily_var ** 2 for pos in self.positions.values()))
            
            # Calculate leverage
            leverage = total_value / self.current_capital
            
            # Calculate concentration risk
            max_position_pct = max(pos.market_value / self.current_capital for pos in self.positions.values()) if self.positions else 0
            
            # Calculate sector exposure (simplified - would need sector mapping)
            sector_exposure = {"Technology": 0.5}  # Placeholder
            
            # Determine portfolio risk level
            portfolio_var_pct = total_var / self.current_capital
            
            if portfolio_var_pct >= self.max_portfolio_risk * 1.5:
                risk_level = RiskLevel.CRITICAL
            elif portfolio_var_pct >= self.max_portfolio_risk:
                risk_level = RiskLevel.HIGH
            elif portfolio_var_pct >= self.max_portfolio_risk * 0.5:
                risk_level = RiskLevel.MEDIUM
            else:
                risk_level = RiskLevel.LOW
            
            self.portfolio_metrics = RiskMetrics(
                portfolio_value=self.current_capital,
                total_exposure=total_value,
                leverage=leverage,
                var_1d=total_var,
                var_5d=total_var * np.sqrt(5),
                max_drawdown=0.0,  # Would need historical tracking
                sharpe_ratio=0.0,  # Would need return history
                sortino_ratio=0.0, # Would need return history
                beta=np.mean([pos.beta for pos in self.positions.values()]) if self.positions else 1.0,
                correlation_spy=np.mean([pos.correlation_spy for pos in self.positions.values()]) if self.positions else 0.0,
                concentration_risk=max_position_pct,
                sector_exposure=sector_exposure,
                risk_level=risk_level,
                violation_count=len(self.risk_violations),
                last_updated=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {e}")
    
    def _check_portfolio_limits(self) -> bool:
        """Check if portfolio is within risk limits"""
        if self.portfolio_metrics is None:
            return True
        
        violations = []
        
        # Check leverage limit
        if self.portfolio_metrics.leverage > self.max_leverage:
            violations.append(f"Leverage {self.portfolio_metrics.leverage:.2f} exceeds limit {self.max_leverage}")
        
        # Check portfolio VaR limit
        var_pct = self.portfolio_metrics.var_1d / self.current_capital
        if var_pct > self.max_portfolio_risk:
            violations.append(f"Portfolio VaR {var_pct:.1%} exceeds limit {self.max_portfolio_risk:.1%}")
        
        # Check concentration risk
        if self.portfolio_metrics.concentration_risk > self.risk_params['concentration_limit']:
            violations.append(f"Concentration risk {self.portfolio_metrics.concentration_risk:.1%} exceeds limit {self.risk_params['concentration_limit']:.1%}")
        
        if violations:
            self.risk_violations.extend(violations)
            self.logger.warning(f"Portfolio limit violations: {violations}")
            return False
        
        return True
    
    def check_trade_approval(self, 
                           symbol: str,
                           quantity: float,
                           price: float,
                           action: str) -> Tuple[bool, str]:
        """
        Check if a trade is approved by risk management
        
        Args:
            symbol: Stock symbol
            quantity: Trade quantity
            price: Trade price
            action: Trade action (buy/sell)
            
        Returns:
            Tuple of (approved, reason)
        """
        try:
            # Check circuit breaker
            if self.circuit_breaker_active:
                return False, "Circuit breaker active"
            
            # Calculate new position after trade
            current_position = self.positions.get(symbol, None)
            if current_position:
                new_quantity = current_position.position_size + quantity
            else:
                new_quantity = quantity
            
            # Simulate position risk
            position_risk = self._calculate_position_risk(symbol, new_quantity, price)
            if position_risk is None:
                return False, "Cannot calculate position risk"
            
            # Check position limits
            if not self._check_position_limits(position_risk):
                return False, "Position would exceed risk limits"
            
            # Check portfolio impact
            trade_value = abs(quantity) * price
            if trade_value > self.current_capital * 0.1:  # Max 10% per trade
                return False, f"Trade size ${trade_value:,.2f} exceeds 10% of capital"
            
            return True, "Trade approved"
            
        except Exception as e:
            self.logger.error(f"Error checking trade approval: {e}")
            return False, f"Risk check error: {e}"
    
    def activate_circuit_breaker(self, reason: str):
        """Activate emergency circuit breaker"""
        self.circuit_breaker_active = True
        self.logger.critical(f"CIRCUIT BREAKER ACTIVATED: {reason}")
        
        # Log to database
        self.db_manager.log_system_event("CIRCUIT_BREAKER", reason)
    
    def deactivate_circuit_breaker(self):
        """Deactivate circuit breaker"""
        self.circuit_breaker_active = False
        self.logger.info("Circuit breaker deactivated")
    
    def get_risk_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'portfolio_metrics': asdict(self.portfolio_metrics) if self.portfolio_metrics else None,
                'positions': {symbol: asdict(pos) for symbol, pos in self.positions.items()},
                'risk_violations': self.risk_violations[-10:],  # Last 10 violations
                'circuit_breaker_active': self.circuit_breaker_active,
                'risk_parameters': self.risk_params,
                'capital': {
                    'initial': self.initial_capital,
                    'current': self.current_capital,
                    'daily_loss_limit': self.daily_loss_limit
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
            return {}
    
    def update_capital(self, new_capital: float):
        """Update current capital amount"""
        self.current_capital = new_capital
        self.logger.info(f"Capital updated to ${new_capital:,.2f}")
    
    def reset_daily_limits(self):
        """Reset daily risk limits (call at start of trading day)"""
        self.risk_violations = []
        self.logger.info("Daily risk limits reset")
    
    def get_position_size_recommendation(self, 
                                       symbol: str,
                                       signal_strength: float,
                                       price: float) -> Tuple[int, str]:
        """
        Get recommended position size based on risk and signal strength
        
        Args:
            symbol: Stock symbol
            signal_strength: Signal confidence (0-1)
            price: Current price
            
        Returns:
            Tuple of (recommended_shares, reasoning)
        """
        try:
            # Base position size on risk budget
            base_risk_budget = self.current_capital * self.max_position_risk
            
            # Get volatility for sizing
            market_data = self._get_market_data(symbol)
            if market_data is None:
                return 0, "No market data available"
            
            returns = market_data['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            # Calculate position size based on volatility
            daily_vol = volatility / np.sqrt(252)
            max_shares_by_risk = int(base_risk_budget / (price * daily_vol * 1.96))
            
            # Adjust by signal strength
            adjusted_shares = int(max_shares_by_risk * signal_strength)
            
            # Apply concentration limit
            max_shares_by_concentration = int(self.current_capital * self.risk_params['concentration_limit'] / price)
            
            # Take minimum
            recommended_shares = min(adjusted_shares, max_shares_by_concentration)
            
            reasoning = f"Risk-based: {max_shares_by_risk}, Signal-adjusted: {adjusted_shares}, Concentration-limited: {max_shares_by_concentration}"
            
            return max(0, recommended_shares), reasoning
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return 0, f"Error: {e}"

# Example usage
if __name__ == "__main__":
    import json

    print("--- Running RiskManager Demo in Live Mode ---")

    # This demo uses the live TradingDatabase.
    # Ensure your database has data for 'AAPL' and 'SPY' for the last year.
    risk_manager = RiskManager(initial_capital=100000)

    print("\n1. Testing Trade Approval for a new position...")
    # This will fetch live data from the DB for risk calculation
    approved, reason = risk_manager.check_trade_approval("AAPL", 50, 150.0, "buy")
    print(f"Trade Approved: {approved}, Reason: {reason}")

    if approved:
        print("\n2. Updating position and recalculating risk...")
        update_success = risk_manager.update_position("AAPL", 50, 150.0, "buy")
        if update_success:
            print("Position for AAPL updated successfully.")
        else:
            print("Failed to update position for AAPL.")

    print("\n3. Generating Risk Report...")
    report = risk_manager.get_risk_report()
    # Use json for pretty printing the nested dictionary
    print(json.dumps(report, indent=2, default=str))

    print("\n4. Testing Position Size Recommendation...")
    recommended_shares, reasoning = risk_manager.get_position_size_recommendation("AAPL", 0.75, 155.0)
    print(f"Recommended Shares: {recommended_shares}, Reasoning: {reasoning}")
    
    print("\n5. Testing Circuit Breaker...")
    risk_manager.activate_circuit_breaker("Manual test")
    print(f"Circuit Breaker Active: {risk_manager.circuit_breaker_active}")
    approved, reason = risk_manager.check_trade_approval("AAPL", 10, 156.0, "buy")
    print(f"Trade Approved while breaker is active: {approved}, Reason: {reason}")
    risk_manager.deactivate_circuit_breaker()
    print(f"Circuit Breaker Active after deactivation: {risk_manager.circuit_breaker_active}")

    print("\n--- Demo Finished ---")
