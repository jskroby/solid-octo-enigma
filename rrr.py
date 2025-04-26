import ccxt
import time
import threading
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.table import Table
import config

console = Console()
start_time = time.time()
total_profit = 0.0
total_trades = 0
stop_event = threading.Event()
lock = threading.Lock()

# === EXCHANGE SETUP ===
def create_exchange():
    return ccxt.gateio({
        'apiKey': config.API_KEY,
        'secret': config.API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'defaultSettle': 'usdt',
            'recvWindow': 10000,
            'adjustForTimeDifference': True
        }
    })

# === MARKET FUNCTIONS ===
def fetch_volatility(ex, symbol, timeframe='5m', lookback=24):
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=lookback)
        closes = np.array([c[4] for c in ohlcv])
        returns = np.diff(np.log(closes))
        return np.std(returns) * np.sqrt(288)
    except Exception:
        return None

def analyze_symbol_trend(ex, symbol):
    try:
        o = ex.fetch_ohlcv(symbol, '15m', limit=20)
        closes = np.array([x[4] for x in o])
        change = (closes[-1] / closes[0] - 1)
        if change > 0.002:
            return 'bullish', change
        if change < -0.002:
            return 'bearish', abs(change)
    except Exception:
        pass
    return 'neutral', 0

# === SYMBOL SELECTION ===
def get_best_symbols(ex):
    try:
        markets = ex.load_markets()
        futs = [s for s in markets if s.endswith('/USDT')]
        tickers = ex.fetch_tickers(futs)
        candidates = []
        for sym, t in tickers.items():
            last = t.get('last')
            ask = t.get('ask')
            bid = t.get('bid')
            if last is None or ask is None or bid is None:
                continue
            spread = (ask - bid) / last
            if spread > config.MAX_SPREAD_PERCENT:
                continue
            vol = fetch_volatility(ex, sym)
            if vol is None or not (config.VOL_LOW <= vol <= config.VOL_HIGH):
                continue
            trend, strength = analyze_symbol_trend(ex, sym)
            score = (1 - spread / config.MAX_SPREAD_PERCENT) * 0.5 + strength * 0.5
            candidates.append((sym, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:config.MAX_SYMBOLS]]
    except Exception as e:
        console.print(f"[red]Error selecting symbols: {e}")
        return []

# === GRID STRATEGY ===
def dynamic_spacing(volatility):
    return config.BASE_SPACING_PERCENT * (1 + (volatility or 0) / config.VOL_HIGH) / 100

def place_grid(ex, symbol):
    orders = []
    try:
        ticker = ex.fetch_ticker(symbol)
        price = ticker['last']
        vol = fetch_volatility(ex, symbol)
        spacing = dynamic_spacing(vol)
        levels = [price * (1 + (i - config.FRACTAL_LEVELS) * spacing)
                  for i in range(config.FRACTAL_LEVELS * 2)]
        size = (config.INITIAL_DEPOSIT - config.SAFETY_BUFFER) / price / len(levels)
        for p in levels:
            side = 'buy' if p < price else 'sell'
            try:
                o = ex.create_limit_order(symbol, side, size, p)
                orders.append(o)
            except Exception:
                pass
    except Exception as e:
        console.print(f"[yellow]Grid placement error for {symbol}: {e}")
    return orders

# === MONITORING & MANAGEMENT ===
def monitor_symbol(ex, symbol):
    global total_profit, total_trades
    orders = place_grid(ex, symbol)
    while not stop_event.is_set():
        try:
            open_ids = [o['id'] for o in ex.fetch_open_orders(symbol)]
            for o in orders:
                if o['id'] not in open_ids and o.get('filled', 0) > 0:
                    avg = o.get('average', o['price'])
                    filled = o.get('filled', 0)
                    profit = (o['price'] - avg) * filled if o['side'] == 'sell' else (avg - o['price']) * filled
                    with lock:
                        total_profit += profit
                        total_trades += 1
                    console.print(f"[green]{symbol} {o['side']} profit {profit:.4f}[/green]")
            time.sleep(1)
        except Exception as e:
            console.print(f"[red]Error in monitor {symbol}: {e}")
            time.sleep(5)

def profit_monitor():
    with Live(console=console, refresh_per_second=1) as live:
        while not stop_event.is_set():
            elapsed = time.time() - start_time
            cps = (total_profit / elapsed) if elapsed > 0 else 0
            table = Table()
            table.add_column("Total Profit", justify="right")
            table.add_column("CPS ($/s)", justify="right")
            table.add_row(f"${total_profit:.2f}", f"{cps:.4f}")
            live.update(table)
            time.sleep(1)

# === MAIN ===
def main():
    ex = create_exchange()
    symbols = get_best_symbols(ex)
    console.print(f"[cyan]Trading Symbols: {symbols}[/cyan]")
    threads = []
    for s in symbols:
        t = threading.Thread(target=monitor_symbol, args=(ex, s), daemon=True)
        threads.append(t)
        t.start()
    t = threading.Thread(target=profit_monitor, daemon=True)
    threads.append(t)
    t.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        console.print("[yellow]Shutting down...[/yellow]")

if __name__ == '__main__':
    main()
