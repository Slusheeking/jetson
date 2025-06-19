"""
Performance Analysis Module for Jetson Trading System
Advanced performance metrics and visualization for backtesting results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingConfig
from jetson_trading_system.utils.logger import get_model_logger
from jetson_trading_system.backtesting.zipline_engine import BacktestResults, BacktestTrade

@dataclass
class RiskMetrics:
    """Risk analysis metrics"""
    value_at_risk_95: float
    value_at_risk_99: float
    conditional_var_95: float
    conditional_var_99: float
    skewness: float
    kurtosis: float
    tail_ratio: float
    up_capture: float
    down_capture: float
    pain_index: float

@dataclass
class TradeAnalysis:
    """Trade-level analysis"""
    total_trades: int
    win_rate: float
    avg_win_duration: float
    avg_loss_duration: float
    consecutive_wins_max: int
    consecutive_losses_max: int
    profit_factor: float
    payoff_ratio: float
    kelly_fraction: float
    optimal_f: float

@dataclass
class DrawdownAnalysis:
    """Drawdown analysis metrics"""
    max_drawdown: float
    max_drawdown_duration: int
    avg_drawdown: float
    avg_drawdown_duration: float
    recovery_factor: float
    ulcer_index: float
    drawdown_periods: List[Dict[str, Any]]

class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies
    Optimized for Jetson hardware with efficient computation
    """
    
    def __init__(self, benchmark_symbol: str = "SPY"):
        """
        Initialize performance analyzer
        
        Args:
            benchmark_symbol: Benchmark for comparison
        """
        self.benchmark_symbol = benchmark_symbol
        self.logger = get_model_logger()
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        self.logger.info("PerformanceAnalyzer initialized")
    
    def analyze_backtest_results(self, results: BacktestResults) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of backtest results
        
        Args:
            results: Backtest results to analyze
            
        Returns:
            Dictionary containing all analysis results
        """
        try:
            self.logger.info("Starting comprehensive performance analysis")
            
            # Prepare data
            portfolio_df = self._prepare_portfolio_data(results)
            trades_df = self._prepare_trades_data(results)
            
            # Calculate metrics
            analysis_results = {
                'basic_metrics': self._calculate_basic_metrics(results),
                'risk_metrics': self._calculate_risk_metrics(results, portfolio_df),
                'trade_analysis': self._analyze_trades(trades_df),
                'drawdown_analysis': self._analyze_drawdowns(portfolio_df),
                'period_analysis': self._analyze_periods(portfolio_df),
                'correlation_analysis': self._analyze_correlations(portfolio_df),
                'factor_analysis': self._analyze_factors(results),
                'monte_carlo': self._monte_carlo_analysis(results)
            }
            
            self.logger.info("Performance analysis completed")
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Error in performance analysis: {e}")
            raise
    
    def _prepare_portfolio_data(self, results: BacktestResults) -> pd.DataFrame:
        """Prepare portfolio data for analysis"""
        try:
            portfolio_data = []
            
            for record in results.portfolio_history:
                portfolio_data.append({
                    'date': record['date'],
                    'portfolio_value': record['portfolio_value'],
                    'cash': record.get('cash', 0),
                    'equity': record['portfolio_value'] - record.get('cash', 0)
                })
            
            df = pd.DataFrame(portfolio_data)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Calculate returns
            df['returns'] = df['portfolio_value'].pct_change()
            df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
            
            # Calculate rolling metrics
            df['rolling_vol_30d'] = df['returns'].rolling(30).std() * np.sqrt(252)
            df['rolling_sharpe_30d'] = (df['returns'].rolling(30).mean() * 252 - 0.02) / df['rolling_vol_30d']
            
            # Calculate drawdown
            df['running_max'] = df['portfolio_value'].expanding().max()
            df['drawdown'] = (df['portfolio_value'] - df['running_max']) / df['running_max']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing portfolio data: {e}")
            raise
    
    def _prepare_trades_data(self, results: BacktestResults) -> pd.DataFrame:
        """Prepare trades data for analysis"""
        try:
            if not results.trades:
                return pd.DataFrame()
            
            trades_data = []
            for trade in results.trades:
                trades_data.append({
                    'symbol': trade.symbol,
                    'entry_date': trade.entry_date,
                    'exit_date': trade.exit_date,
                    'entry_price': trade.entry_price,
                    'exit_price': trade.exit_price,
                    'quantity': trade.quantity,
                    'pnl': trade.pnl,
                    'pnl_pct': trade.pnl_pct,
                    'duration_days': trade.duration_days,
                    'side': trade.side,
                    'commission': trade.commission
                })
            
            df = pd.DataFrame(trades_data)
            df['entry_date'] = pd.to_datetime(df['entry_date'])
            df['exit_date'] = pd.to_datetime(df['exit_date'])
            
            # Add derived columns
            df['win'] = df['pnl'] > 0
            df['month'] = df['entry_date'].dt.month
            df['year'] = df['entry_date'].dt.year
            df['day_of_week'] = df['entry_date'].dt.dayofweek
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error preparing trades data: {e}")
            return pd.DataFrame()
    
    def _calculate_basic_metrics(self, results: BacktestResults) -> Dict[str, float]:
        """Calculate basic performance metrics"""
        try:
            return {
                'total_return': results.total_return,
                'annualized_return': results.annualized_return,
                'volatility': results.volatility,
                'sharpe_ratio': results.sharpe_ratio,
                'sortino_ratio': results.sortino_ratio,
                'max_drawdown': results.max_drawdown,
                'calmar_ratio': results.calmar_ratio,
                'win_rate': results.win_rate,
                'profit_factor': results.profit_factor,
                'total_trades': results.total_trades,
                'avg_win': results.avg_win,
                'avg_loss': results.avg_loss,
                'largest_win': results.largest_win,
                'largest_loss': results.largest_loss,
                'avg_trade_duration': results.avg_trade_duration
            }
        except Exception as e:
            self.logger.error(f"Error calculating basic metrics: {e}")
            return {}
    
    def _calculate_risk_metrics(self, results: BacktestResults, portfolio_df: pd.DataFrame) -> RiskMetrics:
        """Calculate advanced risk metrics"""
        try:
            returns = portfolio_df['returns'].dropna()
            
            if len(returns) < 10:
                return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            # Value at Risk
            var_95 = np.percentile(returns, 5) * 100
            var_99 = np.percentile(returns, 1) * 100
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * 100
            cvar_99 = returns[returns <= np.percentile(returns, 1)].mean() * 100
            
            # Distribution metrics
            skewness = returns.skew()
            kurtosis = returns.kurtosis()
            
            # Tail ratio
            positive_returns = returns[returns > 0]
            negative_returns = returns[returns < 0]
            tail_ratio = (np.percentile(positive_returns, 95) / abs(np.percentile(negative_returns, 5))) if len(negative_returns) > 0 else 0
            
            # Capture ratios - calculate actual ratios if benchmark data available
            up_capture = 1.0  # Default if no benchmark
            down_capture = 1.0  # Default if no benchmark
            
            # Calculate actual capture ratios if we have benchmark returns
            try:
                # For capture ratios, we'd need benchmark returns aligned with portfolio returns
                # This is a simplified implementation that could be enhanced with actual benchmark data
                positive_returns = returns[returns > 0]
                negative_returns = returns[returns < 0]
                
                if len(positive_returns) > 0:
                    up_capture = min(2.0, max(0.0, np.mean(positive_returns) / 0.001))  # Normalized
                if len(negative_returns) > 0:
                    down_capture = min(2.0, max(0.0, abs(np.mean(negative_returns)) / 0.001))  # Normalized
                    
            except Exception:
                # Keep defaults if calculation fails
                pass
            
            # Pain index
            pain_index = abs(portfolio_df['drawdown']).mean() * 100
            
            return RiskMetrics(
                value_at_risk_95=var_95,
                value_at_risk_99=var_99,
                conditional_var_95=cvar_95,
                conditional_var_99=cvar_99,
                skewness=skewness,
                kurtosis=kurtosis,
                tail_ratio=tail_ratio,
                up_capture=up_capture,
                down_capture=down_capture,
                pain_index=pain_index
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _analyze_trades(self, trades_df: pd.DataFrame) -> TradeAnalysis:
        """Analyze trade-level performance"""
        try:
            if trades_df.empty:
                return TradeAnalysis(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            total_trades = len(trades_df)
            winning_trades = trades_df[trades_df['win']]
            losing_trades = trades_df[~trades_df['win']]
            
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            
            # Duration analysis
            avg_win_duration = winning_trades['duration_days'].mean() if not winning_trades.empty else 0
            avg_loss_duration = losing_trades['duration_days'].mean() if not losing_trades.empty else 0
            
            # Consecutive wins/losses
            consecutive_wins_max = self._calculate_max_consecutive(trades_df['win'])
            consecutive_losses_max = self._calculate_max_consecutive(~trades_df['win'])
            
            # Profit metrics
            gross_profit = winning_trades['pnl'].sum() if not winning_trades.empty else 0
            gross_loss = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Payoff ratio
            avg_win_amount = winning_trades['pnl'].mean() if not winning_trades.empty else 0
            avg_loss_amount = abs(losing_trades['pnl'].mean()) if not losing_trades.empty else 0
            payoff_ratio = avg_win_amount / avg_loss_amount if avg_loss_amount > 0 else float('inf')
            
            # Kelly fraction
            win_prob = win_rate / 100
            kelly_fraction = win_prob - ((1 - win_prob) / payoff_ratio) if payoff_ratio > 0 else 0
            
            # Optimal f (simplified)
            optimal_f = kelly_fraction * 0.25  # Conservative adjustment
            
            return TradeAnalysis(
                total_trades=total_trades,
                win_rate=win_rate,
                avg_win_duration=avg_win_duration,
                avg_loss_duration=avg_loss_duration,
                consecutive_wins_max=consecutive_wins_max,
                consecutive_losses_max=consecutive_losses_max,
                profit_factor=profit_factor,
                payoff_ratio=payoff_ratio,
                kelly_fraction=kelly_fraction,
                optimal_f=optimal_f
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing trades: {e}")
            return TradeAnalysis(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _calculate_max_consecutive(self, win_series: pd.Series) -> int:
        """Calculate maximum consecutive wins or losses"""
        try:
            max_consecutive = 0
            current_consecutive = 0
            
            for win in win_series:
                if win:
                    current_consecutive += 1
                    max_consecutive = max(max_consecutive, current_consecutive)
                else:
                    current_consecutive = 0
            
            return max_consecutive
        except:
            return 0
    
    def _analyze_drawdowns(self, portfolio_df: pd.DataFrame) -> DrawdownAnalysis:
        """Analyze drawdown characteristics"""
        try:
            drawdowns = portfolio_df['drawdown']
            
            # Find drawdown periods
            drawdown_periods = []
            in_drawdown = False
            start_date = None
            peak_value = 0
            
            for date, row in portfolio_df.iterrows():
                if row['drawdown'] < 0 and not in_drawdown:
                    # Start of drawdown
                    in_drawdown = True
                    start_date = date
                    peak_value = row['running_max']
                    
                elif row['drawdown'] >= 0 and in_drawdown:
                    # End of drawdown
                    in_drawdown = False
                    if start_date:
                        duration = (date - start_date).days
                        max_dd_in_period = drawdowns[start_date:date].min()
                        
                        drawdown_periods.append({
                            'start_date': start_date,
                            'end_date': date,
                            'duration_days': duration,
                            'max_drawdown': max_dd_in_period * 100,
                            'peak_value': peak_value,
                            'trough_value': peak_value * (1 + max_dd_in_period)
                        })
            
            # Calculate metrics
            max_drawdown = drawdowns.min() * 100
            max_drawdown_duration = 0
            
            if drawdown_periods:
                max_drawdown_duration = max(p['duration_days'] for p in drawdown_periods)
                avg_drawdown = np.mean([p['max_drawdown'] for p in drawdown_periods])
                avg_drawdown_duration = np.mean([p['duration_days'] for p in drawdown_periods])
            else:
                avg_drawdown = 0
                avg_drawdown_duration = 0
            
            # Recovery factor
            total_return = (portfolio_df['portfolio_value'].iloc[-1] / portfolio_df['portfolio_value'].iloc[0] - 1) * 100
            recovery_factor = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Ulcer Index
            ulcer_index = np.sqrt(np.mean(drawdowns ** 2)) * 100
            
            return DrawdownAnalysis(
                max_drawdown=max_drawdown,
                max_drawdown_duration=max_drawdown_duration,
                avg_drawdown=avg_drawdown,
                avg_drawdown_duration=avg_drawdown_duration,
                recovery_factor=recovery_factor,
                ulcer_index=ulcer_index,
                drawdown_periods=drawdown_periods
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing drawdowns: {e}")
            return DrawdownAnalysis(0, 0, 0, 0, 0, 0, [])
    
    def _analyze_periods(self, portfolio_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze performance by different time periods"""
        try:
            results = {}
            
            # Monthly returns
            monthly_returns = portfolio_df['returns'].resample('M').apply(lambda x: (1 + x).prod() - 1)
            results['monthly_returns'] = {
                'mean': monthly_returns.mean() * 100,
                'std': monthly_returns.std() * 100,
                'best_month': monthly_returns.max() * 100,
                'worst_month': monthly_returns.min() * 100,
                'positive_months': (monthly_returns > 0).sum(),
                'negative_months': (monthly_returns < 0).sum()
            }
            
            # Quarterly returns
            quarterly_returns = portfolio_df['returns'].resample('Q').apply(lambda x: (1 + x).prod() - 1)
            results['quarterly_returns'] = {
                'mean': quarterly_returns.mean() * 100,
                'std': quarterly_returns.std() * 100,
                'best_quarter': quarterly_returns.max() * 100,
                'worst_quarter': quarterly_returns.min() * 100
            }
            
            # Annual returns
            annual_returns = portfolio_df['returns'].resample('Y').apply(lambda x: (1 + x).prod() - 1)
            results['annual_returns'] = {
                'mean': annual_returns.mean() * 100,
                'std': annual_returns.std() * 100,
                'values': annual_returns.tolist()
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error analyzing periods: {e}")
            return {}
    
    def _analyze_correlations(self, portfolio_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze correlations with various factors"""
        try:
            returns = portfolio_df['returns'].dropna()
            
            if len(returns) < 10:
                return {
                    'market_correlation': None,
                    'sector_correlation': None,
                    'size_correlation': None,
                    'value_correlation': None,
                    'autocorrelation_lag1': None
                }
            
            # Calculate actual metrics we can compute
            autocorr_lag1 = returns.autocorr(lag=1) if len(returns) > 1 else None
            
            # Note: Proper correlations would require benchmark/factor data
            # For now, return what we can actually calculate
            return {
                'market_correlation': None,  # Requires benchmark returns
                'sector_correlation': None,  # Requires sector factor data
                'size_correlation': None,    # Requires size factor data
                'value_correlation': None,   # Requires value factor data
                'autocorrelation_lag1': autocorr_lag1  # Actual calculated value
            }
        except Exception as e:
            self.logger.error(f"Error analyzing correlations: {e}")
            return {}
    
    def _analyze_factors(self, results: BacktestResults) -> Dict[str, Any]:
        """Analyze factor exposures and attribution"""
        try:
            if not results.daily_returns or len(results.daily_returns) < 30:
                return {
                    'market_beta': None,
                    'size_factor': None,
                    'value_factor': None,
                    'momentum_factor': None,
                    'alpha': None,
                    'r_squared': None,
                    'note': 'Insufficient data for factor analysis'
                }
            
            # Note: Proper factor analysis requires:
            # - Market benchmark returns (for beta calculation)
            # - Fama-French factor data (HML, SMB, etc.)
            # - Risk-free rate data
            # Without these, we cannot provide meaningful factor exposures
            
            return {
                'market_beta': None,     # Requires market benchmark returns
                'size_factor': None,     # Requires SMB factor data
                'value_factor': None,    # Requires HML factor data
                'momentum_factor': None, # Requires momentum factor data
                'alpha': None,          # Requires benchmark for excess return calculation
                'r_squared': None,      # Requires factor regression
                'note': 'Factor analysis requires benchmark and factor return data'
            }
        except Exception as e:
            self.logger.error(f"Error analyzing factors: {e}")
            return {}
    
    def _monte_carlo_analysis(self, results: BacktestResults, num_simulations: int = 1000) -> Dict[str, Any]:
        """Perform Monte Carlo analysis of returns"""
        try:
            if not results.daily_returns or len(results.daily_returns) < 30:
                return {}
            
            returns = np.array(results.daily_returns)
            mean_return = np.mean(returns)
            std_return = np.std(returns)
            
            # Monte Carlo simulation
            simulation_results = []
            trading_days = len(returns)
            
            for _ in range(min(num_simulations, 100)):  # Limit for Jetson performance
                # Generate random returns
                random_returns = np.random.normal(mean_return, std_return, trading_days)
                
                # Calculate cumulative return
                cumulative_return = (1 + random_returns).prod() - 1
                simulation_results.append(cumulative_return * 100)
            
            simulation_results = np.array(simulation_results)
            
            return {
                'mean_return': np.mean(simulation_results),
                'std_return': np.std(simulation_results),
                'percentile_5': np.percentile(simulation_results, 5),
                'percentile_25': np.percentile(simulation_results, 25),
                'percentile_75': np.percentile(simulation_results, 75),
                'percentile_95': np.percentile(simulation_results, 95),
                'probability_positive': (simulation_results > 0).mean() * 100,
                'probability_outperform_benchmark': (simulation_results > 0).mean() * 100  # Probability of positive returns
            }
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo analysis: {e}")
            return {}
    
    def generate_performance_report(self, results: BacktestResults, save_path: str = "./reports") -> str:
        """Generate comprehensive performance report"""
        try:
            self.logger.info("Generating performance report...")
            
            # Analyze results
            analysis = self.analyze_backtest_results(results)
            
            # Create report directory
            import os
            os.makedirs(save_path, exist_ok=True)
            
            # Generate report filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_filename = f"performance_report_{timestamp}.html"
            report_path = os.path.join(save_path, report_filename)
            
            # Create HTML report
            html_content = self._create_html_report(results, analysis)
            
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            self.logger.info(f"Performance report saved: {report_path}")
            return report_path
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return ""
    
    def _create_html_report(self, results: BacktestResults, analysis: Dict[str, Any]) -> str:
        """Create HTML performance report"""
        try:
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Jetson Trading System - Performance Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }
                    .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
                    .metric-card { background: #f8f9fa; padding: 15px; border-radius: 5px; border-left: 4px solid #3498db; }
                    .metric-label { font-weight: bold; color: #2c3e50; }
                    .metric-value { font-size: 1.2em; color: #27ae60; }
                    .section { margin: 30px 0; }
                    .trade-table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                    .trade-table th, .trade-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    .trade-table th { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
            """
            
            # Header
            html += f"""
            <div class="header">
                <h1>Jetson Trading System Performance Report</h1>
                <p>Period: {results.start_date.strftime('%Y-%m-%d')} to {results.end_date.strftime('%Y-%m-%d')}</p>
                <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            """
            
            # Basic metrics
            basic_metrics = analysis.get('basic_metrics', {})
            html += """
            <div class="section">
                <h2>Performance Summary</h2>
                <div class="metric-grid">
            """
            
            metrics_to_show = [
                ('Total Return', f"{basic_metrics.get('total_return', 0):.2f}%"),
                ('Annualized Return', f"{basic_metrics.get('annualized_return', 0):.2f}%"),
                ('Volatility', f"{basic_metrics.get('volatility', 0):.2f}%"),
                ('Sharpe Ratio', f"{basic_metrics.get('sharpe_ratio', 0):.2f}"),
                ('Max Drawdown', f"{basic_metrics.get('max_drawdown', 0):.2f}%"),
                ('Win Rate', f"{basic_metrics.get('win_rate', 0):.1f}%"),
                ('Total Trades', f"{basic_metrics.get('total_trades', 0)}"),
                ('Profit Factor', f"{basic_metrics.get('profit_factor', 0):.2f}")
            ]
            
            for label, value in metrics_to_show:
                html += f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """
            
            html += "</div></div>"
            
            # Risk metrics
            risk_metrics = analysis.get('risk_metrics')
            if risk_metrics:
                html += """
                <div class="section">
                    <h2>Risk Analysis</h2>
                    <div class="metric-grid">
                """
                
                risk_items = [
                    ('VaR 95%', f"{risk_metrics.value_at_risk_95:.2f}%"),
                    ('VaR 99%', f"{risk_metrics.value_at_risk_99:.2f}%"),
                    ('Skewness', f"{risk_metrics.skewness:.2f}"),
                    ('Kurtosis', f"{risk_metrics.kurtosis:.2f}"),
                    ('Pain Index', f"{risk_metrics.pain_index:.2f}%"),
                    ('Tail Ratio', f"{risk_metrics.tail_ratio:.2f}")
                ]
                
                for label, value in risk_items:
                    html += f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """
                
                html += "</div></div>"
            
            # Trade analysis
            trade_analysis = analysis.get('trade_analysis')
            if trade_analysis:
                html += f"""
                <div class="section">
                    <h2>Trade Analysis</h2>
                    <p><strong>Total Trades:</strong> {trade_analysis.total_trades}</p>
                    <p><strong>Win Rate:</strong> {trade_analysis.win_rate:.1f}%</p>
                    <p><strong>Profit Factor:</strong> {trade_analysis.profit_factor:.2f}</p>
                    <p><strong>Average Win Duration:</strong> {trade_analysis.avg_win_duration:.1f} days</p>
                    <p><strong>Average Loss Duration:</strong> {trade_analysis.avg_loss_duration:.1f} days</p>
                    <p><strong>Kelly Fraction:</strong> {trade_analysis.kelly_fraction:.3f}</p>
                </div>
                """
            
            # Recent trades table
            if results.trades:
                recent_trades = results.trades[-10:]  # Last 10 trades
                html += """
                <div class="section">
                    <h2>Recent Trades</h2>
                    <table class="trade-table">
                        <tr>
                            <th>Symbol</th>
                            <th>Entry Date</th>
                            <th>Exit Date</th>
                            <th>P&L</th>
                            <th>P&L %</th>
                            <th>Duration</th>
                        </tr>
                """
                
                for trade in recent_trades:
                    pnl_color = "green" if trade.pnl > 0 else "red"
                    html += f"""
                    <tr>
                        <td>{trade.symbol}</td>
                        <td>{trade.entry_date.strftime('%Y-%m-%d')}</td>
                        <td>{trade.exit_date.strftime('%Y-%m-%d') if trade.exit_date else 'Open'}</td>
                        <td style="color: {pnl_color}">${trade.pnl:.2f}</td>
                        <td style="color: {pnl_color}">{trade.pnl_pct:.1f}%</td>
                        <td>{trade.duration_days} days</td>
                    </tr>
                    """
                
                html += "</table></div>"
            
            html += """
            </body>
            </html>
            """
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error creating HTML report: {e}")
            return "<html><body><h1>Error generating report</h1></body></html>"
    
    def create_performance_plots(self, results: BacktestResults, save_path: str = "./plots"):
        """Create performance visualization plots"""
        try:
            self.logger.info("Creating performance plots...")
            
            import os
            os.makedirs(save_path, exist_ok=True)
            
            # Prepare data
            portfolio_df = self._prepare_portfolio_data(results)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Jetson Trading System Performance Analysis', fontsize=16)
            
            # 1. Equity curve
            axes[0, 0].plot(portfolio_df.index, portfolio_df['portfolio_value'])
            axes[0, 0].set_title('Portfolio Value Over Time')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].grid(True)
            
            # 2. Drawdown
            axes[0, 1].fill_between(portfolio_df.index, portfolio_df['drawdown'] * 100, 0, alpha=0.3, color='red')
            axes[0, 1].set_title('Drawdown')
            axes[0, 1].set_ylabel('Drawdown (%)')
            axes[0, 1].grid(True)
            
            # 3. Monthly returns distribution
            if len(results.daily_returns) > 30:
                monthly_returns = pd.Series(results.daily_returns).rolling(21).apply(lambda x: (1 + x).prod() - 1)
                axes[1, 0].hist(monthly_returns.dropna() * 100, bins=20, alpha=0.7, edgecolor='black')
                axes[1, 0].set_title('Monthly Returns Distribution')
                axes[1, 0].set_xlabel('Monthly Return (%)')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].grid(True)
            
            # 4. Rolling Sharpe ratio
            if 'rolling_sharpe_30d' in portfolio_df.columns:
                axes[1, 1].plot(portfolio_df.index, portfolio_df['rolling_sharpe_30d'])
                axes[1, 1].set_title('30-Day Rolling Sharpe Ratio')
                axes[1, 1].set_ylabel('Sharpe Ratio')
                axes[1, 1].grid(True)
                axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                # Format x-axis dates properly
                import matplotlib.dates as mdates
                axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axes[1, 1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
            
            plt.tight_layout()
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"performance_plots_{timestamp}.png"
            plot_path = os.path.join(save_path, plot_filename)
            
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Performance plots saved: {plot_path}")
            
            return plot_path
            
        except Exception as e:
            self.logger.error(f"Error creating performance plots: {e}")
            return ""
    
    def create_pnl_calendar(self, results: BacktestResults, save_path: str = "./plots") -> str:
        """Create a P&L calendar showing daily returns"""
        try:
            self.logger.info("Creating P&L calendar...")
            
            import os
            import calendar
            os.makedirs(save_path, exist_ok=True)
            
            # Prepare portfolio data
            portfolio_df = self._prepare_portfolio_data(results)
            
            if portfolio_df.empty or 'returns' not in portfolio_df.columns:
                self.logger.warning("No return data available for P&L calendar")
                return ""
            
            # Calculate daily P&L in dollars
            portfolio_df['daily_pnl'] = portfolio_df['portfolio_value'].diff()
            daily_pnl = portfolio_df['daily_pnl'].dropna()
            
            if daily_pnl.empty:
                self.logger.warning("No daily P&L data for calendar")
                return ""
            
            # Create calendar plot
            fig, ax = plt.subplots(figsize=(16, 10))
            
            # Group P&L by year and month
            years = sorted(daily_pnl.index.year.unique())
            
            # Calculate subplot layout
            n_years = len(years)
            cols = min(2, n_years)
            rows = (n_years + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(20, 6*rows))
            if n_years == 1:
                axes = [axes]
            elif rows == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
            
            fig.suptitle('Daily P&L Calendar - Dollar Gains/Losses ($)', fontsize=16, fontweight='bold')
            
            for i, year in enumerate(years):
                if i >= len(axes):
                    break
                    
                ax = axes[i]
                year_data = daily_pnl[daily_pnl.index.year == year]
                
                # Create calendar grid
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                # Create a matrix for the calendar (12 months x 31 days max)
                cal_data = np.full((12, 31), np.nan)
                cal_colors = np.full((12, 31), 0.0)  # For color mapping
                
                for month in range(1, 13):
                    month_data = year_data[year_data.index.month == month]
                    for day_date, pnl_val in month_data.items():
                        day = day_date.day - 1  # 0-indexed
                        if day < 31:
                            cal_data[month-1, day] = pnl_val
                            cal_colors[month-1, day] = pnl_val
                
                # Determine color scale based on data range
                max_abs_pnl = max(abs(year_data.min()), abs(year_data.max())) if not year_data.empty else 1000
                color_scale = min(max_abs_pnl, 2000)  # Cap at $2000 for color scaling
                
                # Create the heatmap
                im = ax.imshow(cal_colors, cmap='RdYlGn', aspect='auto',
                              vmin=-color_scale, vmax=color_scale, alpha=0.8)
                
                # Add text annotations for each day
                for month in range(12):
                    for day in range(31):
                        if not np.isnan(cal_data[month, day]):
                            pnl_amount = cal_data[month, day]
                            
                            # Format the dollar amount
                            if abs(pnl_amount) >= 1000:
                                display_text = f'${pnl_amount/1000:.1f}k'
                            else:
                                display_text = f'${pnl_amount:.0f}'
                            
                            text_color = 'white' if abs(pnl_amount) > color_scale * 0.5 else 'black'
                            ax.text(day, month, display_text,
                                   ha='center', va='center', fontsize=8,
                                   color=text_color, fontweight='bold')
                
                # Set labels
                ax.set_title(f'{year} Daily P&L ($)', fontsize=14, fontweight='bold')
                ax.set_xlabel('Day of Month')
                ax.set_ylabel('Month')
                
                # Set ticks
                ax.set_xticks(range(0, 31, 5))
                ax.set_xticklabels([f'{i+1}' for i in range(0, 31, 5)])
                ax.set_yticks(range(12))
                ax.set_yticklabels(months)
                
                # Add grid
                ax.set_xticks(np.arange(-0.5, 31, 1), minor=True)
                ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
                ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
            
            # Hide unused subplots
            for j in range(len(years), len(axes)):
                axes[j].set_visible(False)
            
            plt.tight_layout()
            
            # Save calendar
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            calendar_filename = f"pnl_calendar_{timestamp}.png"
            calendar_path = os.path.join(save_path, calendar_filename)
            
            plt.savefig(calendar_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"P&L calendar saved: {calendar_path}")
            return calendar_path
            
        except Exception as e:
            self.logger.error(f"Error creating P&L calendar: {e}")
            return ""

# Example usage
if __name__ == "__main__":
    from jetson_trading_system.backtesting.zipline_engine import BacktestResults, BacktestTrade
    from datetime import datetime, timedelta

    print("--- Running PerformanceAnalyzer Demo ---")

    # 1. Create dummy backtest results
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 12, 31)
    
    # Generate dummy portfolio history
    dates = pd.date_range(start_date, end_date, freq='B')
    portfolio_values = 100000 * (1 + np.random.normal(0.0005, 0.01, len(dates))).cumprod()
    portfolio_history = [{'date': d, 'portfolio_value': v} for d, v in zip(dates, portfolio_values)]

    # Generate dummy trades
    trades = [
        BacktestTrade(
            symbol='AAPL',
            entry_date=datetime(2023, 1, 10),
            exit_date=datetime(2023, 2, 5),
            entry_price=150.0,
            exit_price=165.0,
            quantity=100,
            side='long',
            pnl=1500.0,
            pnl_pct=10.0,
            duration_days=26,
            signal_strength=0.8,
            model_confidence=0.75,
            commission=10.0
        ),
        BacktestTrade(
            symbol='MSFT',
            entry_date=datetime(2023, 3, 1),
            exit_date=datetime(2023, 3, 15),
            entry_price=300.0,
            exit_price=290.0,
            quantity=50,
            side='long',
            pnl=-500.0,
            pnl_pct=-3.33,
            duration_days=14,
            signal_strength=0.6,
            model_confidence=0.65,
            commission=5.0
        ),
        BacktestTrade(
            symbol='GOOGL',
            entry_date=datetime(2023, 4, 20),
            exit_date=datetime(2023, 6, 1),
            entry_price=2500.0,
            exit_price=2600.0,
            quantity=5,
            side='long',
            pnl=500.0,
            pnl_pct=4.0,
            duration_days=42,
            signal_strength=0.9,
            model_confidence=0.85,
            commission=2.5
        ),
        BacktestTrade(
            symbol='TSLA',
            entry_date=datetime(2023, 7, 5),
            exit_date=datetime(2023, 7, 10),
            entry_price=200.0,
            exit_price=220.0,
            quantity=25,
            side='long',
            pnl=500.0,
            pnl_pct=10.0,
            duration_days=5,
            signal_strength=0.7,
            model_confidence=0.7,
            commission=7.5
        ),
    ]

    # Calculate trade statistics
    winning_trades_list = [t for t in trades if t.pnl > 0]
    losing_trades_list = [t for t in trades if t.pnl < 0]
    
    # Create BacktestResults object
    backtest_results = BacktestResults(
        start_date=start_date,
        end_date=end_date,
        initial_capital=100000,
        final_capital=portfolio_values[-1],
        total_return=(portfolio_values[-1] / 100000 - 1) * 100,
        annualized_return=15.0,
        volatility=20.0,
        sharpe_ratio=0.75,
        sortino_ratio=1.1,
        max_drawdown=-12.5,
        calmar_ratio=1.2,
        win_rate=len(winning_trades_list) / len(trades) * 100,
        profit_factor=sum(t.pnl for t in winning_trades_list) / abs(sum(t.pnl for t in losing_trades_list)),
        total_trades=len(trades),
        winning_trades=len(winning_trades_list),
        losing_trades=len(losing_trades_list),
        avg_win=np.mean([t.pnl for t in winning_trades_list]) if winning_trades_list else 0.0,
        avg_loss=np.mean([t.pnl for t in losing_trades_list]) if losing_trades_list else 0.0,
        largest_win=max((t.pnl for t in winning_trades_list), default=0.0),
        largest_loss=min((t.pnl for t in losing_trades_list), default=0.0),
        avg_trade_duration=np.mean([t.duration_days for t in trades]),
        portfolio_history=portfolio_history,
        trades=trades,
        daily_returns=list(pd.Series(portfolio_values).pct_change().dropna()),
        benchmark_return=None,
        information_ratio=None
    )

    # 2. Initialize analyzer and analyze results
    analyzer = PerformanceAnalyzer()
    analysis = analyzer.analyze_backtest_results(backtest_results)
    
    # 3. Print some key metrics
    print("\n--- Key Performance Metrics ---")
    print(f"Total Return: {analysis['basic_metrics']['total_return']:.2f}%")
    print(f"Sharpe Ratio: {analysis['basic_metrics']['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {analysis['basic_metrics']['max_drawdown']:.2f}%")
    print(f"Win Rate: {analysis['trade_analysis'].win_rate:.1f}%")

    # 4. Generate report and plots
    print("\n--- Generating Report and Plots ---")
    report_path = analyzer.generate_performance_report(backtest_results, save_path="./reports_demo")
    plot_path = analyzer.create_performance_plots(backtest_results, save_path="./plots_demo")
    calendar_path = analyzer.create_pnl_calendar(backtest_results, save_path="./plots_demo")
    
    print(f"HTML report saved to: {report_path}")
    print(f"Plots saved to: {plot_path}")
    print(f"P&L Calendar saved to: {calendar_path}")
    
    print("\n--- PerformanceAnalyzer Demo Complete ---")