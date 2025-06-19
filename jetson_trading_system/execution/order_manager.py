"""
Order Management System for Jetson Trading
Handles order execution, tracking, and broker integration
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import uuid
import json

from jetson_trading_system.config.jetson_settings import JetsonConfig
from jetson_trading_system.config.trading_params import TradingConfig
from jetson_trading_system.utils.logger import get_trading_logger
from jetson_trading_system.utils.database import TradingDatabase

class OrderStatus(Enum):
    """Order status types"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderSide(Enum):
    """Order sides"""
    BUY = "buy"
    SELL = "sell"

@dataclass
class Order:
    """Order data structure"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: Optional[float]
    stop_price: Optional[float]
    status: OrderStatus
    filled_quantity: int
    avg_fill_price: float
    submitted_time: datetime
    updated_time: datetime
    notes: str
    broker_order_id: Optional[str] = None
    commission: float = 0.0
    total_value: float = 0.0

@dataclass
class Fill:
    """Trade fill data structure"""
    fill_id: str
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    commission: float
    value: float

class OrderManager:
    """
    Comprehensive order management system
    Handles both paper trading and live broker integration
    """
    
    def __init__(self, 
                 paper_trading: bool = True,
                 broker_name: str = "alpaca",
                 commission_per_share: float = 0.0):
        """
        Initialize order manager
        
        Args:
            paper_trading: Use paper trading mode
            broker_name: Broker to use for live trading
            commission_per_share: Commission per share
        """
        self.paper_trading = paper_trading
        self.broker_name = broker_name
        self.commission_per_share = commission_per_share
        
        self.logger = get_trading_logger()
        self.db_manager = TradingDatabase()
        
        # Order tracking
        self.orders = {}  # order_id -> Order
        self.fills = {}   # fill_id -> Fill
        self.pending_orders = {}  # order_id -> Order
        
        # Paper trading state
        self.paper_positions = {}  # symbol -> quantity
        self.paper_cash = 25000.0  # Starting cash
        self.paper_market_data = {}  # symbol -> current_price
        
        # Statistics
        self.order_stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': 0.0,
            'total_commission': 0.0
        }
        
        # Threading
        self.order_lock = threading.Lock()
        self.update_thread = None
        self.shutdown_event = threading.Event()
        
        # Broker connection (placeholder)
        self.broker_connected = False
        
        # Initialize
        self._initialize_paper_trading()
        if not paper_trading:
            self._initialize_broker_connection()
        
        self.logger.info(f"OrderManager initialized - Paper trading: {paper_trading}")
    
    def _initialize_paper_trading(self):
        """Initialize paper trading environment"""
        try:
            # Load paper trading state from database if exists
            self.paper_positions = self._load_paper_positions()
            self.paper_cash = self._load_paper_cash()
            
            # Start price update thread for paper trading
            if self.paper_trading:
                self.update_thread = threading.Thread(target=self._paper_trading_update_loop, daemon=True)
                self.update_thread.start()
            
            self.logger.info(f"Paper trading initialized - Cash: ${self.paper_cash:,.2f}")
            
        except Exception as e:
            self.logger.error(f"Error initializing paper trading: {e}")
    
    def _initialize_broker_connection(self):
        """Initialize broker connection for live trading"""
        try:
            # Placeholder for broker initialization
            # In real implementation, this would connect to Alpaca API
            self.broker_connected = True
            self.logger.info(f"Broker connection initialized: {self.broker_name}")
            
        except Exception as e:
            self.logger.error(f"Error connecting to broker: {e}")
            self.broker_connected = False
    
    def is_connected(self) -> bool:
        """Check if order manager is ready for trading"""
        if self.paper_trading:
            return True
        return self.broker_connected
    
    def place_market_buy_order(self, 
                             symbol: str,
                             quantity: int,
                             notes: str = "") -> Optional[str]:
        """
        Place market buy order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            notes: Order notes
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            if quantity <= 0:
                self.logger.error(f"Invalid quantity for buy order: {quantity}")
                return None
            
            # Create order
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=None,
                stop_price=None,
                status=OrderStatus.PENDING,
                filled_quantity=0,
                avg_fill_price=0.0,
                submitted_time=datetime.now(),
                updated_time=datetime.now(),
                notes=notes
            )
            
            # Submit order
            if self._submit_order(order):
                with self.order_lock:
                    self.orders[order_id] = order
                    if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                        self.pending_orders[order_id] = order
                
                self.order_stats['total_orders'] += 1
                self.logger.info(f"Market buy order placed: {symbol} {quantity} shares (ID: {order_id})")
                return order_id
            else:
                self.logger.error(f"Failed to submit buy order: {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing market buy order: {e}")
            return None
    
    def place_market_sell_order(self, 
                              symbol: str,
                              quantity: int,
                              notes: str = "") -> Optional[str]:
        """
        Place market sell order
        
        Args:
            symbol: Stock symbol
            quantity: Number of shares
            notes: Order notes
            
        Returns:
            Order ID if successful, None otherwise
        """
        try:
            if quantity <= 0:
                self.logger.error(f"Invalid quantity for sell order: {quantity}")
                return None
            
            # Check position for sell order
            current_position = self.get_position(symbol)
            if current_position < quantity:
                self.logger.error(f"Insufficient position for sell: {symbol} (have: {current_position}, need: {quantity})")
                return None
            
            # Create order
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=quantity,
                price=None,
                stop_price=None,
                status=OrderStatus.PENDING,
                filled_quantity=0,
                avg_fill_price=0.0,
                submitted_time=datetime.now(),
                updated_time=datetime.now(),
                notes=notes
            )
            
            # Submit order
            if self._submit_order(order):
                with self.order_lock:
                    self.orders[order_id] = order
                    if order.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED]:
                        self.pending_orders[order_id] = order
                
                self.order_stats['total_orders'] += 1
                self.logger.info(f"Market sell order placed: {symbol} {quantity} shares (ID: {order_id})")
                return order_id
            else:
                self.logger.error(f"Failed to submit sell order: {symbol}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error placing market sell order: {e}")
            return None
    
    def place_limit_order(self, 
                         symbol: str,
                         side: OrderSide,
                         quantity: int,
                         limit_price: float,
                         notes: str = "") -> Optional[str]:
        """Place limit order"""
        try:
            order_id = self._generate_order_id()
            order = Order(
                order_id=order_id,
                symbol=symbol,
                side=side,
                order_type=OrderType.LIMIT,
                quantity=quantity,
                price=limit_price,
                stop_price=None,
                status=OrderStatus.PENDING,
                filled_quantity=0,
                avg_fill_price=0.0,
                submitted_time=datetime.now(),
                updated_time=datetime.now(),
                notes=notes
            )
            
            if self._submit_order(order):
                with self.order_lock:
                    self.orders[order_id] = order
                    self.pending_orders[order_id] = order
                
                self.order_stats['total_orders'] += 1
                self.logger.info(f"Limit {side.value} order placed: {symbol} {quantity}@${limit_price:.2f} (ID: {order_id})")
                return order_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error placing limit order: {e}")
            return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            with self.order_lock:
                if order_id not in self.orders:
                    self.logger.error(f"Order not found: {order_id}")
                    return False
                
                order = self.orders[order_id]
                
                if order.status not in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]:
                    self.logger.warning(f"Cannot cancel order in status: {order.status}")
                    return False
                
                # Cancel with broker or paper trading
                if self._cancel_order_with_broker(order):
                    order.status = OrderStatus.CANCELLED
                    order.updated_time = datetime.now()
                    
                    # Remove from pending orders
                    if order_id in self.pending_orders:
                        del self.pending_orders[order_id]
                    
                    self.order_stats['cancelled_orders'] += 1
                    self.logger.info(f"Order cancelled: {order_id}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error cancelling order: {e}")
            return False
    
    def _submit_order(self, order: Order) -> bool:
        """Submit order to broker or paper trading system"""
        try:
            if self.paper_trading:
                return self._submit_paper_order(order)
            else:
                return self._submit_broker_order(order)
                
        except Exception as e:
            self.logger.error(f"Error submitting order: {e}")
            return False
    
    def _submit_paper_order(self, order: Order) -> bool:
        """Submit order to paper trading system"""
        try:
            # Get current market price
            current_price = self._get_paper_market_price(order.symbol)
            if current_price is None:
                order.status = OrderStatus.REJECTED
                self.logger.error(f"No market price available for {order.symbol}")
                return False
            
            # For market orders, fill immediately
            if order.order_type == OrderType.MARKET:
                fill_price = current_price
                
                # Check if we have enough cash for buy orders
                if order.side == OrderSide.BUY:
                    required_cash = order.quantity * fill_price
                    if self.paper_cash < required_cash:
                        order.status = OrderStatus.REJECTED
                        self.logger.error(f"Insufficient cash for order: ${required_cash:,.2f} required, ${self.paper_cash:,.2f} available")
                        return False
                
                # Execute the fill
                self._execute_paper_fill(order, fill_price, order.quantity)
                return True
            
            # For limit orders, mark as submitted (would be filled later based on price)
            else:
                order.status = OrderStatus.SUBMITTED
                return True
                
        except Exception as e:
            self.logger.error(f"Error in paper order submission: {e}")
            return False
    
    def _submit_broker_order(self, order: Order) -> bool:
        """Submit order to live broker"""
        try:
            # Placeholder for broker API integration
            # In real implementation, this would call Alpaca API
            
            # Simulate broker submission
            order.status = OrderStatus.SUBMITTED
            order.broker_order_id = f"broker_{order.order_id}"
            
            # For demo, immediately fill market orders
            if order.order_type == OrderType.MARKET:
                # Simulate immediate fill with slight slippage
                fill_price = 150.0  # Placeholder price
                self._execute_broker_fill(order, fill_price, order.quantity)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error submitting broker order: {e}")
            return False
    
    def _execute_paper_fill(self, order: Order, fill_price: float, fill_quantity: int):
        """Execute fill in paper trading"""
        try:
            # Calculate commission
            commission = fill_quantity * self.commission_per_share
            
            # Create fill record
            fill_id = self._generate_fill_id()
            fill = Fill(
                fill_id=fill_id,
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=fill_quantity,
                price=fill_price,
                timestamp=datetime.now(),
                commission=commission,
                value=fill_quantity * fill_price
            )
            
            # Update order
            order.filled_quantity += fill_quantity
            order.avg_fill_price = fill_price  # Simplified
            order.commission += commission
            order.total_value = order.filled_quantity * order.avg_fill_price
            order.updated_time = datetime.now()
            
            if order.filled_quantity >= order.quantity:
                order.status = OrderStatus.FILLED
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
            
            # Update paper trading state
            if order.side == OrderSide.BUY:
                self.paper_cash -= (fill_quantity * fill_price + commission)
                self.paper_positions[order.symbol] = self.paper_positions.get(order.symbol, 0) + fill_quantity
            else:  # SELL
                self.paper_cash += (fill_quantity * fill_price - commission)
                self.paper_positions[order.symbol] = self.paper_positions.get(order.symbol, 0) - fill_quantity
            
            # Store fill
            self.fills[fill_id] = fill
            
            # Update statistics
            if order.status == OrderStatus.FILLED:
                self.order_stats['filled_orders'] += 1
                if order.order_id in self.pending_orders:
                    del self.pending_orders[order.order_id]
            
            self.order_stats['total_volume'] += fill.value
            self.order_stats['total_commission'] += commission
            
            self.logger.info(f"Paper fill executed: {order.symbol} {fill_quantity}@${fill_price:.2f} (Order: {order.order_id})")
            
        except Exception as e:
            self.logger.error(f"Error executing paper fill: {e}")
    
    def _execute_broker_fill(self, order: Order, fill_price: float, fill_quantity: int):
        """Execute fill from broker"""
        # Similar to paper fill but would integrate with broker fill notifications
        self._execute_paper_fill(order, fill_price, fill_quantity)
    
    def _cancel_order_with_broker(self, order: Order) -> bool:
        """Cancel order with broker"""
        if self.paper_trading:
            return True  # Paper orders can always be cancelled
        else:
            # Placeholder for broker cancellation
            return True
    
    def _get_paper_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for paper trading"""
        try:
            # In real implementation, this would fetch from market data feed
            # For now, use a placeholder price with some randomness
            base_prices = {
                'AAPL': 150.0,
                'MSFT': 300.0,
                'GOOGL': 2500.0,
                'TSLA': 200.0,
                'SPY': 400.0,
                'QQQ': 350.0,
                'IWM': 180.0,
                'VIX': 20.0
            }
            
            base_price = base_prices.get(symbol, 100.0)
            
            # Add some random movement (±2%)
            import random
            price_change = random.uniform(-0.02, 0.02)
            current_price = base_price * (1 + price_change)
            
            self.paper_market_data[symbol] = current_price
            return current_price
            
        except Exception as e:
            self.logger.error(f"Error getting market price for {symbol}: {e}")
            return None
    
    def _paper_trading_update_loop(self):
        """Update loop for paper trading order processing"""
        try:
            while not self.shutdown_event.is_set():
                try:
                    # Update pending limit orders
                    self._update_pending_limit_orders()
                    
                    # Sleep briefly
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Error in paper trading update loop: {e}")
                    time.sleep(5)
                    
        except Exception as e:
            self.logger.error(f"Paper trading update loop error: {e}")
    
    def _update_pending_limit_orders(self):
        """Check and fill pending limit orders based on market prices"""
        try:
            with self.order_lock:
                orders_to_process = list(self.pending_orders.values())
            
            for order in orders_to_process:
                if order.order_type != OrderType.LIMIT:
                    continue
                
                current_price = self._get_paper_market_price(order.symbol)
                if current_price is None:
                    continue
                
                # Check if limit order should be filled
                should_fill = False
                if order.side == OrderSide.BUY and current_price <= order.price:
                    should_fill = True
                elif order.side == OrderSide.SELL and current_price >= order.price:
                    should_fill = True
                
                if should_fill:
                    self._execute_paper_fill(order, order.price, order.quantity)
                    
        except Exception as e:
            self.logger.error(f"Error updating pending limit orders: {e}")
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID"""
        return self.orders.get(order_id)
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a symbol"""
        return [order for order in self.orders.values() if order.symbol == symbol]
    
    def get_pending_orders(self) -> List[Order]:
        """Get all pending orders"""
        return list(self.pending_orders.values())
    
    def get_position(self, symbol: str) -> int:
        """Get current position for symbol"""
        if self.paper_trading:
            return self.paper_positions.get(symbol, 0)
        else:
            # Would query broker for live positions
            return 0
    
    def get_cash_balance(self) -> float:
        """Get current cash balance"""
        if self.paper_trading:
            return self.paper_cash
        else:
            # Would query broker for live cash balance
            return 0.0
    
    def get_order_statistics(self) -> Dict[str, Any]:
        """Get order statistics"""
        return self.order_stats.copy()
    
    def get_recent_fills(self, limit: int = 10) -> List[Fill]:
        """Get recent fills"""
        fills = sorted(self.fills.values(), key=lambda f: f.timestamp, reverse=True)
        return fills[:limit]
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        return f"ORD_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    def _generate_fill_id(self) -> str:
        """Generate unique fill ID"""
        return f"FILL_{uuid.uuid4().hex[:8]}_{int(time.time())}"
    
    def _load_paper_positions(self) -> Dict[str, int]:
        """Load paper trading positions from database"""
        try:
            # Placeholder - would load from database
            return {}
        except Exception as e:
            self.logger.error(f"Error loading paper positions: {e}")
            return {}
    
    def _load_paper_cash(self) -> float:
        """Load paper trading cash from database"""
        try:
            # Placeholder - would load from database
            return 25000.0
        except Exception as e:
            self.logger.error(f"Error loading paper cash: {e}")
            return 25000.0
    
    def save_state(self):
        """Save paper trading state to database"""
        try:
            # Placeholder - would save to database
            self.logger.info("Paper trading state saved")
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")
    
    def shutdown(self):
        """Shutdown order manager"""
        try:
            self.shutdown_event.set()
            
            if self.update_thread and self.update_thread.is_alive():
                self.update_thread.join(timeout=5)
            
            # Save state
            if self.paper_trading:
                self.save_state()
            
            self.logger.info("OrderManager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")

# Example usage
if __name__ == "__main__":
    print("--- Running OrderManager Demo (Paper Trading) ---")
    
    # Initialize order manager in paper trading mode
    order_manager = OrderManager(paper_trading=True)
    
    try:
        # 1. Check initial state
        print("\n1. Initial State:")
        print(f"Is Connected: {order_manager.is_connected()}")
        print(f"Initial Cash: ${order_manager.get_cash_balance():,.2f}")
        print(f"Initial AAPL Position: {order_manager.get_position('AAPL')}")

        # 2. Place a market buy order
        print("\n2. Placing Market Buy Order for 10 AAPL...")
        buy_order_id = order_manager.place_market_buy_order("AAPL", 10, "Initial test buy")
        time.sleep(1) # Allow order to process
        
        if buy_order_id:
            buy_order = order_manager.get_order(buy_order_id)
            print(f"Buy Order ID: {buy_order_id}, Status: {buy_order.status.value}")
            print(f"Cash after buy: ${order_manager.get_cash_balance():,.2f}")
            print(f"AAPL Position after buy: {order_manager.get_position('AAPL')}")
            assert order_manager.get_position('AAPL') == 10
        else:
            print("Buy order failed to place.")

        # 3. Place a limit sell order (will not fill immediately)
        print("\n3. Placing Limit Sell Order for 5 AAPL at a high price...")
        limit_sell_id = order_manager.place_limit_order("AAPL", OrderSide.SELL, 5, 999.0, "High limit sell")
        time.sleep(1)

        if limit_sell_id:
            limit_order = order_manager.get_order(limit_sell_id)
            print(f"Limit Sell Order ID: {limit_sell_id}, Status: {limit_order.status.value}")
            pending_orders = order_manager.get_pending_orders()
            print(f"Pending orders: {[o.order_id for o in pending_orders]}")
            assert limit_sell_id in [o.order_id for o in pending_orders]
        else:
            print("Limit sell order failed to place.")

        # 4. Cancel the limit order
        print("\n4. Cancelling the limit sell order...")
        cancelled = order_manager.cancel_order(limit_sell_id)
        time.sleep(1)
        
        cancelled_order = order_manager.get_order(limit_sell_id)
        print(f"Order cancelled: {cancelled}, New Status: {cancelled_order.status.value}")
        pending_orders_after_cancel = order_manager.get_pending_orders()
        print(f"Pending orders after cancel: {[o.order_id for o in pending_orders_after_cancel]}")
        assert cancelled_order.status == OrderStatus.CANCELLED

        # 5. Place a market sell order
        print("\n5. Placing Market Sell Order for 3 AAPL...")
        sell_order_id = order_manager.place_market_sell_order("AAPL", 3, "Partial sell")
        time.sleep(1)

        if sell_order_id:
            sell_order = order_manager.get_order(sell_order_id)
            print(f"Sell Order ID: {sell_order_id}, Status: {sell_order.status.value}")
            print(f"Cash after sell: ${order_manager.get_cash_balance():,.2f}")
            print(f"AAPL Position after sell: {order_manager.get_position('AAPL')}")
            assert order_manager.get_position('AAPL') == 7
        else:
            print("Sell order failed to place.")

        # 6. Check recent fills
        print("\n6. Recent Fills:")
        recent_fills = order_manager.get_recent_fills()
        for fill in recent_fills:
            print(f"  - Fill ID: {fill.fill_id}, Side: {fill.side.value}, Qty: {fill.quantity}, Price: ${fill.price:.2f}")

        # 7. Check order statistics
        print("\n7. Final Order Statistics:")
        stats = order_manager.get_order_statistics()
        print(json.dumps(stats, indent=2))
        assert stats['total_orders'] == 3
        assert stats['filled_orders'] == 2

    except Exception as e:
        print(f"\nAn error occurred during the demo: {e}")
    finally:
        # Shutdown the order manager
        order_manager.shutdown()
        print("\n--- OrderManager Demo Complete ---")
