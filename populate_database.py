#!/usr/bin/env python3
"""
Script to populate the database with historical market data
Uses Polygon API to fetch real data for backtesting
"""

import os
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from jetson_trading_system.utils.database import TradingDatabase
from jetson_trading_system.data.polygon_client import PolygonClient
from jetson_trading_system.features.technical_indicators import TechnicalIndicators

async def populate_market_data():
    """Populate database with historical market data"""
    
    # Initialize components
    db = TradingDatabase()
    ta_calculator = TechnicalIndicators()
    
    # Get API key
    api_key = os.getenv('POLYGON_API_KEY')
    if not api_key:
        print("❌ POLYGON_API_KEY not found in .env file")
        return False
    
    print(f"🔑 Using Polygon API key: {api_key[:8]}...")
    
    # Symbols to fetch data for - separate stocks from indices
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'SPY', 'QQQ', 'IWM']
    index_symbols = ['VIX']  # VIX is an index
    all_symbols = stock_symbols + index_symbols
    
    # Date range for historical data - Get 2+ years for ML model training
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=850)).strftime('%Y-%m-%d')  # ~2.3 years (~600 trading days)
    
    print(f"📅 Fetching data from {start_date} to {end_date}")
    print(f"📊 Symbols: {', '.join(all_symbols)}")
    
    async with PolygonClient(api_key=api_key) as client:
        total_records = 0
        
        for i, symbol in enumerate(all_symbols, 1):
            print(f"\n[{i}/{len(all_symbols)}] Fetching data for {symbol}...")
            
            try:
                # Determine if symbol is an index
                is_index = symbol in index_symbols
                
                # Fetch historical bars
                df = await client.get_historical_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timespan="day",
                    is_index=is_index
                )
                
                if df is not None and not df.empty:
                    print(f"  📈 Received {len(df)} records")
                    
                    # Prepare data for database - fix timestamp format
                    df_prepared = df.copy()
                    
                    # Ensure timestamp is datetime, then convert to string format for SQLite
                    if 'timestamp' in df_prepared.columns:
                        # Convert to datetime if it's not already
                        df_prepared['timestamp'] = pd.to_datetime(df_prepared['timestamp'])
                        # Convert to string format for SQLite
                        df_prepared['timestamp'] = df_prepared['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
                    
                    df_prepared['symbol'] = symbol
                    
                    # Ensure numeric columns are properly formatted
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
                    for col in numeric_cols:
                        if col in df_prepared.columns:
                            df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce').fillna(0)
                    
                    # Insert market data
                    rows_inserted = db.insert_market_data(df_prepared)
                    print(f"  💾 Inserted {rows_inserted} records into database")
                    
                    # Calculate and store technical indicators
                    if len(df_prepared) >= 50:  # Need enough data for indicators
                        print(f"  🔧 Calculating technical indicators...")
                        try:
                            # Prepare data with proper datetime index for technical indicators
                            df_for_indicators = df_prepared.copy()
                            # Set timestamp column as index and ensure it's datetime
                            df_for_indicators['timestamp'] = pd.to_datetime(df_for_indicators['timestamp'])
                            df_for_indicators = df_for_indicators.set_index('timestamp')
                            
                            indicators = ta_calculator.calculate_all_indicators(df_for_indicators)
                            
                            if indicators is not None and not indicators.empty:
                                # Store indicators in database
                                for timestamp, row in indicators.iterrows():
                                    indicator_dict = {}
                                    for col in indicators.columns:
                                        if pd.notna(row[col]) and str(row[col]).lower() not in ['nan', 'inf', '-inf']:
                                            try:
                                                indicator_dict[col] = float(row[col])
                                            except (ValueError, TypeError):
                                                continue
                                    
                                    if indicator_dict:
                                        # Ensure timestamp is properly formatted
                                        timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S') if hasattr(timestamp, 'strftime') else str(timestamp)
                                        db.insert_technical_indicators(
                                            symbol=symbol,
                                            timestamp=timestamp_str,
                                            indicators=indicator_dict
                                        )
                                
                                print(f"  📊 Stored technical indicators")
                            else:
                                print(f"  ⚠️  No technical indicators calculated")
                        except Exception as e:
                            print(f"  ⚠️  Error calculating technical indicators: {e}")
                    
                    total_records += len(df)
                    
                else:
                    print(f"  ⚠️  No data received for {symbol}")
                    
            except Exception as e:
                print(f"  ❌ Error fetching data for {symbol}: {e}")
                continue
            
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)
        
        print(f"\n✅ Database population completed!")
        print(f"📊 Total records inserted: {total_records:,}")
        
        # Verify data
        print(f"\n🔍 Verifying data...")
        for symbol in all_symbols:
            data = db.get_market_data(symbol=symbol, limit=5)
            if not data.empty:
                # timestamp is now the index, not a column
                latest_date = data.index.max().strftime('%Y-%m-%d') if hasattr(data.index.max(), 'strftime') else str(data.index.max())
                print(f"  ✅ {symbol}: {len(data)} records found (latest: {latest_date})")
            else:
                print(f"  ❌ {symbol}: No data found")
        
        return True

async def main():
    """Main function"""
    print("🚀 Starting database population with real market data...")
    print("=" * 60)
    
    try:
        success = await populate_market_data()
        
        if success:
            print("\n" + "=" * 60)
            print("🎉 Database population completed successfully!")
            print("You can now run the backtesting engine with real data.")
        else:
            print("\n" + "=" * 60)
            print("❌ Database population failed.")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())