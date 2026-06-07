import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta

def get_top_symbols(exchange_id='mexc', limit=50, quote_currency='USDT'):
    exchange = getattr(ccxt, exchange_id)()
    exchange.load_markets()
    markets = exchange.fetch_tickers()

    symbols_with_vol = []
    for symbol, ticker in markets.items():
        if (f'/{quote_currency}' in symbol or f'-{quote_currency}' in symbol) and ticker['quoteVolume']:
            symbols_with_vol.append({
                'symbol': symbol,
                'quoteVolume': ticker['quoteVolume']
            })

    symbols_with_vol.sort(key=lambda x: x['quoteVolume'], reverse=True)

    eligible_symbols = []
    # Use a fixed date 6 months before "now" (which is May 2026 in this environment)
    # 1780790400000 is June 7, 2026
    six_months_ago_ts = 1780790400000 - (180 * 24 * 3600 * 1000)

    print(f"Checking issuance duration for top symbols on {exchange_id}...")
    for item in symbols_with_vol:
        symbol = item['symbol']
        try:
            # Check if it existed 6 months ago
            first_candle = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=1, since=six_months_ago_ts)
            if first_candle and first_candle[0][0] <= six_months_ago_ts:
                eligible_symbols.append(symbol)
                if len(eligible_symbols) >= limit:
                    break
        except Exception as e:
            pass
        time.sleep(0.05)

    return eligible_symbols
