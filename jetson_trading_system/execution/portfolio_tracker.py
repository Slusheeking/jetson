"""
Portfolio Tracking System for Jetson Trading
Real-time portfolio monitoring and performance tracking
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import threading
import time

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingConfig
from jetson_trading_system.utils.logger import get_trading_logger
from jetson_trading_system.utils.database import TradingDatabase

@dataclass
class Position:
    """Individual position data"""
    symbol: str
    quantity: int
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    total_pnl: float
    day_change: float
    day_change_pct: float
    position_pct: float
    entry_date: datetime
    last_updated: datetime

@dataclass
class PortfolioSummary:
    """Portfolio summary metrics"""
    total_value: float
    cash_balance: float
    equity_value: float
    day_change: float
    day_change_pct: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    total_pnl: float
    position_count: int
    largest_position_pct: float
    sector_allocation: Dict[str, float]
    last_updated: datetime

@dataclass
class PerformanceMetrics:
    """Portfolio performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    consecutive_wins: int
    consecutive_losses: int

class PortfolioTracker:
    """
    Comprehensive portfolio tracking and performance monitoring
    Optimized for real-time updates on Jetson hardware
    """
    
    def __init__(self, 
                 initial_cash: float = 25000.0,
                 benchmark_symbol: str = "SPY"):
        """
        Initialize portfolio tracker
        
        Args:
            initial_cash: Starting cash amount
            benchmark_symbol: Benchmark for performance comparison
        """
        self.initial_cash = initial_cash
        self.benchmark_symbol = benchmark_symbol
        
        self.logger = get_trading_logger()
        self.db_manager = TradingDatabase()
        
        # Portfolio state
        self.positions = {}  # symbol -> Position
        self.cash_balance = initial_cash
        self.portfolio_history = []  # Historical portfolio values
        self.trade_history = []      # Historical trades
        
        # Performance tracking
        self.daily_returns = []
        self.benchmark_returns = []
        self.drawdown_history = []
        self.high_water_mark = initial_cash
        
        # Metrics cache
        self.portfolio_summary = None
        self.performance_metrics = None
        self.last_update_time = None
        
        # Threading
        self.update_lock = threading.Lock()
        
        # Market data cache
        self.market_prices = {}  # symbol -> current_price
        self.previous_closes = {}  # symbol -> previous_close
        
        self.logger.info(f"PortfolioTracker initialized with ${initial_cash:,.2f}")
    
    def update_position(self, 
                       symbol: str,
                       quantity_change: int,
                       price: float,
                       trade_type: str = "buy") -> bool:
        """
        Update position based on trade execution
        
        Args:
            symbol: Stock symbol
            quantity_change: Change in position (positive for buy, negative for sell)
            price: Execution price
            trade_type: Type of trade ("buy" or "sell")
            
        Returns:
            True if update successful
        """
        try:
            with self.update_lock:
                current_time = datetime.now()
                
                # Get or create position
                if symbol in self.positions:
                    position = self.positions[symbol]
                else:
                    position = Position(
                        symbol=symbol,
                        quantity=0,
                        avg_cost=0.0,
                        current_price=price,
                        market_value=0.0,
                        unrealized_pnl=0.0,
                        realized_pnl=0.0,
                        total_pnl=0.0,
                        day_change=0.0,
                        day_change_pct=0.0,
                        position_pct=0.0,
                        entry_date=current_time,
                        last_updated=current_time
                    )
                
                # Calculate trade value
                trade_value = abs(quantity_change) * price
                
                if trade_type == "buy" and quantity_change > 0:
                    # Buy trade
                    if position.quantity > 0:
                        # Adding to existing position - calculate new avg cost
                        total_cost = (position.quantity * position.avg_cost) + trade_value
                        new_quantity = position.quantity + quantity_change
                        position.avg_cost = total_cost / new_quantity
                    else:
                        # New position
                        position.avg_cost = price
                        position.entry_date = current_time
                    
                    position.quantity += quantity_change
                    self.cash_balance -= trade_value
                    
                elif trade_type == "sell" and quantity_change < 0:
                    # Sell trade
                    if position.quantity == 0:
                        self.logger.error(f"Cannot sell {symbol}: no position exists")
                        return False
                    
                    quantity_sold = abs(quantity_change)
                    if quantity_sold > position.quantity:
                        self.logger.error(f"Cannot sell {quantity_sold} shares of {symbol}: only {position.quantity} available")
                        return False
                    
                    # Calculate realized P&L
                    cost_basis = quantity_sold * position.avg_cost
                    sale_proceeds = trade_value
                    realized_pnl = sale_proceeds - cost_basis
                    
                    position.quantity -= quantity_sold
                    position.realized_pnl += realized_pnl
                    self.cash_balance += trade_value
                    
                    # Remove position if fully closed
                    if position.quantity == 0:
                        del self.positions[symbol]
                        
                        # Record trade in history
                        self._record_trade(symbol, quantity_change, price, realized_pnl, current_time)
                        return True
                
                # Update position metrics
                position.current_price = price
                position.market_value = position.quantity * price
                position.unrealized_pnl = position.market_value - (position.quantity * position.avg_cost)
                position.total_pnl = position.unrealized_pnl + position.realized_pnl
                position.last_updated = current_time
                
                # Store updated position
                self.positions[symbol] = position
                
                # Record trade in history
                realized_pnl = position.realized_pnl if trade_type == "sell" else 0.0
                self._record_trade(symbol, quantity_change, price, realized_pnl, current_time)
                
                # Update portfolio metrics
                self._update_portfolio_metrics()
                
                self.logger.info(f"Position updated: {symbol} {quantity_change:+d} @ ${price:.2f}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error updating position for {symbol}: {e}")
            return False
    
    def update_market_prices(self, market_data: Dict[str, float]):
        """
        Update current market prices for all positions
        
        Args:
            market_data: Dictionary of symbol -> current_price
        """
        try:
            with self.update_lock:
                updated_positions = []
                
                for symbol, position in self.positions.items():
                    if symbol in market_data:
                        new_price = market_data[symbol]
                        old_price = position.current_price
                        
                        # Update position with new price
                        position.current_price = new_price
                        position.market_value = position.quantity * new_price
                        position.unrealized_pnl = position.market_value - (position.quantity * position.avg_cost)
                        position.total_pnl = position.unrealized_pnl + position.realized_pnl
                        
                        # Calculate day change
                        if symbol in self.previous_closes:
                            prev_close = self.previous_closes[symbol]
                            position.day_change = new_price - prev_close
                            position.day_change_pct = (position.day_change / prev_close) * 100 if prev_close > 0 else 0.0
                        
                        position.last_updated = datetime.now()
                        updated_positions.append(symbol)
                
                # Update market price cache
                self.market_prices.update(market_data)
                
                # Update portfolio metrics
                self._update_portfolio_metrics()
                
                self.logger.info(f"Market prices updated for {len(updated_positions)} positions")
                
        except Exception as e:
            self.logger.error(f"Error updating market prices: {e}")
    
    def update_positions(self):
        """Update all positions with current market data"""
        try:
            # Get current market prices for all positions
            symbols = list(self.positions.keys())
            if not symbols:
                return
            
            # In real implementation, would fetch from market data feed
            # For now, simulate with slight price movements
            market_data = {}
            for symbol in symbols:
                current_price = self.positions[symbol].current_price
                # Simulate ±0.5% random movement
                import random
                price_change = random.uniform(-0.005, 0.005)
                new_price = current_price * (1 + price_change)
                market_data[symbol] = new_price
            
            self.update_market_prices(market_data)
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _update_portfolio_metrics(self):
        """Update portfolio-level metrics"""
        try:
            current_time = datetime.now()
            
            # Calculate portfolio totals
            total_equity_value = sum(pos.market_value for pos in self.positions.values())
            total_portfolio_value = total_equity_value + self.cash_balance
            
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            total_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
            total_pnl = total_unrealized_pnl + total_realized_pnl
            
            # Calculate day change
            day_change = total_portfolio_value - self.initial_cash  # Simplified
            day_change_pct = (day_change / self.initial_cash) * 100 if self.initial_cash > 0 else 0.0
            
            # Update position percentages
            for position in self.positions.values():
                position.position_pct = (position.market_value / total_portfolio_value) * 100 if total_portfolio_value > 0 else 0.0
            
            # Find largest position
            largest_position_pct = max([pos.position_pct for pos in self.positions.values()]) if self.positions else 0.0
            
            # Sector allocation (simplified)
            sector_allocation = self._calculate_sector_allocation()
            
            # Create portfolio summary
            self.portfolio_summary = PortfolioSummary(
                total_value=total_portfolio_value,
                cash_balance=self.cash_balance,
                equity_value=total_equity_value,
                day_change=day_change,
                day_change_pct=day_change_pct,
                total_unrealized_pnl=total_unrealized_pnl,
                total_realized_pnl=total_realized_pnl,
                total_pnl=total_pnl,
                position_count=len(self.positions),
                largest_position_pct=largest_position_pct,
                sector_allocation=sector_allocation,
                last_updated=current_time
            )
            
            # Update portfolio history
            self.portfolio_history.append({
                'timestamp': current_time,
                'total_value': total_portfolio_value,
                'cash': self.cash_balance,
                'equity': total_equity_value,
                'pnl': total_pnl
            })
            
            # Keep only recent history (last 1000 points)
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
            
            # Update high water mark and drawdown
            if total_portfolio_value > self.high_water_mark:
                self.high_water_mark = total_portfolio_value
            
            current_drawdown = (self.high_water_mark - total_portfolio_value) / self.high_water_mark
            self.drawdown_history.append({
                'timestamp': current_time,
                'value': total_portfolio_value,
                'drawdown': current_drawdown
            })
            
            self.last_update_time = current_time
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio metrics: {e}")
    
    def _calculate_sector_allocation(self) -> Dict[str, float]:
        """Calculate sector allocation (simplified)"""
        try:
            # Simplified sector mapping
            sector_map = {
                'AAPL': 'Technology',
                'MSFT': 'Technology',
                'GOOGL': 'Technology',
                'TSLA': 'Consumer Discretionary',
                'SPY': 'Diversified',
                'QQQ': 'Technology',
                'IWM': 'Diversified',
                'VIX': 'Volatility'
            }
            
            sector_values = {}
            total_value = self.get_total_value()
            
            for symbol, position in self.positions.items():
                sector = sector_map.get(symbol, 'Other')
                sector_values[sector] = sector_values.get(sector, 0) + position.market_value
            
            # Convert to percentages
            sector_allocation = {}
            for sector, value in sector_values.items():
                sector_allocation[sector] = (value / total_value) * 100 if total_value > 0 else 0.0
            
            return sector_allocation
            
        except Exception as e:
            self.logger.error(f"Error calculating sector allocation: {e}")
            return {}
    
    def _record_trade(self, 
                     symbol: str,
                     quantity: int,
                     price: float,
                     realized_pnl: float,
                     timestamp: datetime):
        """Record trade in history"""
        try:
            trade_record = {
                'timestamp': timestamp,
                'symbol': symbol,
                'quantity': quantity,
                'price': price,
                'value': abs(quantity) * price,
                'side': 'buy' if quantity > 0 else 'sell',
                'realized_pnl': realized_pnl
            }
            
            self.trade_history.append(trade_record)
            
            # Keep only recent trades (last 1000)
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-1000:]
                
        except Exception as e:
            self.logger.error(f"Error recording trade: {e}")
    
    def calculate_performance_metrics(self) -> PerformanceMetrics:
        """Calculate comprehensive performance metrics"""
        try:
            if len(self.portfolio_history) < 2:
                # Not enough data for calculations
                return PerformanceMetrics(
                    total_return=0.0, annualized_return=0.0, volatility=0.0,
                    sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                    calmar_ratio=0.0, win_rate=0.0, profit_factor=0.0,
                    avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
                    consecutive_wins=0, consecutive_losses=0
                )
            
            # Calculate returns
            portfolio_values = [h['total_value'] for h in self.portfolio_history]
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
            
            # Basic metrics
            total_return = (portfolio_values[-1] / self.initial_cash - 1) * 100
            
            # Annualized return
            days = (self.portfolio_history[-1]['timestamp'] - self.portfolio_history[0]['timestamp']).days
            annualized_return = ((portfolio_values[-1] / self.initial_cash) ** (365.25 / max(days, 1)) - 1) * 100 if days > 0 else 0.0
            
            # Volatility (annualized)
            volatility = np.std(returns) * np.sqrt(252) * 100 if len(returns) > 1 else 0.0
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            excess_returns = returns - (risk_free_rate / 252)
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252) if np.std(excess_returns) > 0 else 0.0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            downside_deviation = np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            sortino_ratio = np.mean(excess_returns) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0.0
            
            # Maximum drawdown
            max_drawdown = max([d['drawdown'] for d in self.drawdown_history]) * 100 if self.drawdown_history else 0.0
            
            # Calmar ratio
            calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0.0
            
            # Trade-based metrics
            winning_trades = [t for t in self.trade_history if t['realized_pnl'] > 0]
            losing_trades = [t for t in self.trade_history if t['realized_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(self.trade_history) * 100 if self.trade_history else 0.0
            
            avg_win = np.mean([t['realized_pnl'] for t in winning_trades]) if winning_trades else 0.0
            avg_loss = np.mean([t['realized_pnl'] for t in losing_trades]) if losing_trades else 0.0
            
            profit_factor = abs(sum([t['realized_pnl'] for t in winning_trades])) / abs(sum([t['realized_pnl'] for t in losing_trades])) if losing_trades else float('inf')
            
            largest_win = max([t['realized_pnl'] for t in winning_trades]) if winning_trades else 0.0
            largest_loss = min([t['realized_pnl'] for t in losing_trades]) if losing_trades else 0.0
            
            # Consecutive wins/losses
            consecutive_wins, consecutive_losses = self._calculate_consecutive_trades()
            
            self.performance_metrics = PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                max_drawdown=max_drawdown,
                calmar_ratio=calmar_ratio,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win=avg_win,
                avg_loss=avg_loss,
                largest_win=largest_win,
                largest_loss=largest_loss,
                consecutive_wins=consecutive_wins,
                consecutive_losses=consecutive_losses
            )
            
            return self.performance_metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {e}")
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                calmar_ratio=0.0, win_rate=0.0, profit_factor=0.0,
                avg_win=0.0, avg_loss=0.0, largest_win=0.0, largest_loss=0.0,
                consecutive_wins=0, consecutive_losses=0
            )
    
    def _calculate_consecutive_trades(self) -> Tuple[int, int]:
        """Calculate maximum consecutive wins and losses"""
        try:
            if not self.trade_history:
                return 0, 0
            
            max_consecutive_wins = 0
            max_consecutive_losses = 0
            current_wins = 0
            current_losses = 0
            
            for trade in self.trade_history:
                if trade['realized_pnl'] > 0:
                    current_wins += 1
                    current_losses = 0
                    max_consecutive_wins = max(max_consecutive_wins, current_wins)
                elif trade['realized_pnl'] < 0:
                    current_losses += 1
                    current_wins = 0
                    max_consecutive_losses = max(max_consecutive_losses, current_losses)
                else:
                    current_wins = 0
                    current_losses = 0
            
            return max_consecutive_wins, max_consecutive_losses
            
        except Exception as e:
            self.logger.error(f"Error calculating consecutive trades: {e}")
            return 0, 0
    
    # Public interface methods
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for specific symbol"""
        return self.positions.get(symbol)
    
    def get_position_size(self, symbol: str) -> int:
        """Get position size for symbol"""
        position = self.positions.get(symbol)
        return position.quantity if position else 0
    
    def get_all_positions(self) -> Dict[str, int]:
        """Get all positions as symbol -> quantity mapping"""
        return {symbol: pos.quantity for symbol, pos in self.positions.items()}
    
    def get_position_count(self) -> int:
        """Get number of open positions"""
        return len(self.positions)
    
    def get_total_value(self) -> float:
        """Get total portfolio value"""
        equity_value = sum(pos.market_value for pos in self.positions.values())
        return equity_value + self.cash_balance
    
    def get_equity_value(self) -> float:
        """Get total equity value"""
        return sum(pos.market_value for pos in self.positions.values())
    
    def get_cash_balance(self) -> float:
        """Get current cash balance"""
        return self.cash_balance
    
    def get_daily_pnl(self) -> float:
        """Get today's P&L"""
        if self.portfolio_summary:
            return self.portfolio_summary.day_change
        return 0.0
    
    def get_total_pnl(self) -> float:
        """Get total P&L"""
        if self.portfolio_summary:
            return self.portfolio_summary.total_pnl
        return 0.0
    
    def get_portfolio_summary(self) -> Optional[PortfolioSummary]:
        """Get current portfolio summary"""
        return self.portfolio_summary
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics"""
        return self.calculate_performance_metrics()
    
    def get_recent_trades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent trades"""
        return self.trade_history[-limit:] if self.trade_history else []
    
    def get_portfolio_history(self, days: int = 30) -> List[Dict[str, Any]]:
        """Get portfolio history for specified number of days"""
        if not self.portfolio_history:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=days)
        return [h for h in self.portfolio_history if h['timestamp'] >= cutoff_date]
    
    def export_portfolio_data(self) -> Dict[str, Any]:
        """Export comprehensive portfolio data"""
        return {
            'summary': asdict(self.portfolio_summary) if self.portfolio_summary else None,
            'positions': {symbol: asdict(pos) for symbol, pos in self.positions.items()},
            'performance': asdict(self.get_performance_metrics()),
            'recent_trades': self.get_recent_trades(50),
            'portfolio_history': self.get_portfolio_history(90),
            'cash_balance': self.cash_balance,
            'last_updated': self.last_update_time.isoformat() if self.last_update_time else None
        }

# Example usage
if __name__ == "__main__":
    import time
    
    print("--- Running PortfolioTracker Demo ---")
    
    # 1. Initialize portfolio tracker
    tracker = PortfolioTracker(initial_cash=100000.0)

    print("\n1. Initial State:")
    summary = tracker.get_portfolio_summary()
    print(f"Initial Portfolio Value: ${tracker.get_total_value():,.2f}")
    print(f"Initial Cash: ${tracker.get_cash_balance():,.2f}")

    # 2. Simulate a buy trade
    print("\n2. Simulating BUY 10 AAPL @ $150.00")
    tracker.update_position("AAPL", 10, 150.0, "buy")
    print(f"Cash after buy: ${tracker.get_cash_balance():,.2f}")
    print(f"AAPL Position Size: {tracker.get_position_size('AAPL')}")
    aapl_pos = tracker.get_position('AAPL')
    print(f"AAPL Avg Cost: ${aapl_pos.avg_cost:.2f}")

    # 3. Simulate another buy trade
    print("\n3. Simulating BUY 20 MSFT @ $300.00")
    tracker.update_position("MSFT", 20, 300.0, "buy")
    summary = tracker.get_portfolio_summary()
    print(f"Portfolio Value after buys: ${summary.total_value:,.2f}")
    
    # 4. Update market prices (price appreciation)
    print("\n4. Updating market prices (AAPL -> $155, MSFT -> $310)")
    tracker.update_market_prices({"AAPL": 155.0, "MSFT": 310.0})
    summary = tracker.get_portfolio_summary()
    print(f"Portfolio Value after price update: ${summary.total_value:,.2f}")
    print(f"Total Unrealized P&L: ${summary.total_unrealized_pnl:,.2f}")
    
    aapl_pos = tracker.get_position('AAPL')
    print(f"AAPL Unrealized P&L: ${aapl_pos.unrealized_pnl:.2f}")
    assert abs(aapl_pos.unrealized_pnl - 50.0) < 1e-9 # 10 * (155-150)

    # 5. Simulate a sell trade (partial)
    print("\n5. Simulating SELL 5 MSFT @ $310.00")
    tracker.update_position("MSFT", -5, 310.0, "sell")
    summary = tracker.get_portfolio_summary()
    msft_pos = tracker.get_position('MSFT')
    print(f"Cash after sell: ${tracker.get_cash_balance():,.2f}")
    print(f"MSFT Position Size: {msft_pos.quantity}")
    print(f"Total Realized P&L: ${summary.total_realized_pnl:,.2f}")
    assert abs(summary.total_realized_pnl - 50.0) < 1e-9 # 5 * (310-300)

    # 6. Simulate some time passing with random price walks
    print("\n6. Simulating market for 5 steps...")
    for i in range(5):
        tracker.update_positions()
        summary = tracker.get_portfolio_summary()
        print(f"  Step {i+1}: Portfolio Value: ${summary.total_value:,.2f}, P&L: ${summary.total_pnl:,.2f}")
        time.sleep(0.1)

    # 7. Get final performance metrics
    print("\n7. Final Performance Metrics:")
    performance = tracker.get_performance_metrics()
    print(f"Total Return: {performance.total_return:.2f}%")
    print(f"Volatility: {performance.volatility:.2f}%")
    print(f"Sharpe Ratio: {performance.sharpe_ratio:.2f}")
    print(f"Max Drawdown: {performance.max_drawdown:.2f}%")
    print(f"Win Rate: {performance.win_rate:.1f}%")
    
    # 8. Export final portfolio state
    print("\n8. Exporting portfolio data:")
    export_data = tracker.export_portfolio_data()
    print(f"Final Total Value: ${export_data['summary']['total_value']:,.2f}")
    print(f"Final Position Count: {len(export_data['positions'])}")

    print("\n--- PortfolioTracker Demo Complete ---")
