import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def get_top_symbols(exchange_id='binance', limit=50, quote_currency='USDT'):
    exchange = getattr(ccxt, exchange_id)()
    exchange.load_markets()
    markets = exchange.fetch_tickers()

    # Filter for quote currency and active markets
    symbols_with_vol = []
    for symbol, ticker in markets.items():
        if symbol.endswith(f'/{quote_currency}') and ticker['quoteVolume']:
            symbols_with_vol.append({
                'symbol': symbol,
                'quoteVolume': ticker['quoteVolume']
            })

    # Sort by volume
    symbols_with_vol.sort(key=lambda x: x['quoteVolume'], reverse=True)

    # Check for issuance duration (at least 6 months)
    eligible_symbols = []
    six_months_ago = (datetime.now() - timedelta(days=180)).timestamp() * 1000

    print(f"Checking issuance duration for top {limit*2} high-volume symbols...")
    for item in symbols_with_vol[:limit*2]: # Check more than limit in case some don't meet criteria
        symbol = item['symbol']
        try:
            # Fetch the first candle to estimate listing date
            first_candle = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=1, since=0)
            if first_candle and first_candle[0][0] < six_months_ago:
                eligible_symbols.append(symbol)
                if len(eligible_symbols) >= limit:
                    break
        except Exception as e:
            print(f"Error checking listing date for {symbol}: {e}")
        time.sleep(0.1) # Rate limit protection

    return eligible_symbols

if __name__ == "__main__":
    try:
        top_50 = get_top_symbols(limit=5) # Test with 5
        print(f"Eligible top symbols: {top_50}")
    except Exception as e:
        print(f"Error fetching symbols: {e}")
