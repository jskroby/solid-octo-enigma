# hydra_monolith_teroid_live.py
# 🛸 LIVE Teroided Monolith Bot for Gate.io Futures
# 🚀 Immortal, Self-Healing, Self-Compounding Profit Engine (Environmental Variables Edition)

import ccxt
import os
import time
import threading
import numpy as np
import random
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.live import Live

# === LIVE GATE.IO ENVIRONMENTAL API KEYS ===
API_KEY = os.getenv('GATEIO_API_KEY')
API_SECRET = os.getenv('GATEIO_API_SECRET')

# === HYDRA SETTINGS ===
MAX_SYMBOLS             = 10    # Max symbols active
LEVERAGE                = 20    # Safe leverage
GRID_SPACING_PERCENT    = 0.15  # 0.15% spacing
TARGET_GRID_PROFIT      = 0.003 # 0.3% target profit
STOP_LOSS_THRESHOLD     = -0.005 # -0.5% stoploss
REFRESH_INTERVAL        = 900   # Refresh grids every 15 minutes
COMPOUND_PERCENT        = 0.90  # 90% reinvest profits

# === VOLATILITY FILTERS ===
MAX_VOLATILITY_THRESHOLD = 0.08
MIN_VOLATILITY_THRESHOLD = 0.004

# === SYSTEM STATE ===
console = Console()
thread_lock = threading.Lock()
active_grids = {}
symbol_data = {}
start_time = None
total_profit = 0.0
total_trades = 0
total_locked_profit = 0.0

# === CORE FUNCTIONS ===
def create_exchange():
    exchange = ccxt.gateio({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'defaultSettle': 'usdt',
        }
    })
    exchange.urls['api']['swap'] = 'https://api.gateio.ws/api/v4'
    return exchange

def safe_fetch(method, *args, **kwargs):
    try:
        return method(*args, **kwargs)
    except Exception as e:
        console.print(f"[red]API Error: {e}")
        return None

def get_symbol_volatility(ex, symbol, timeframe='5m', lookback=12):
    ohlcv = safe_fetch(ex.fetch_ohlcv, symbol, timeframe, limit=lookback)
    if not ohlcv:
        return None
    closes = np.array([c[4] for c in ohlcv])
    returns = np.diff(np.log(closes))
    return np.std(returns) * np.sqrt(288)

def get_best_symbols(ex, max_symbols=5):
    symbols = []
    try:
        markets = safe_fetch(ex.load_markets)
        if not markets:
            return []
        candidates = []
        usdtm = [s for s in markets if s.endswith('/USDT:USDT') and markets[s]['active']]
        tickers = safe_fetch(ex.fetch_tickers, usdtm[:50])
        if not tickers:
            return []

        for sym, tk in tickers.items():
            last = tk['last']
            if last < 0.0001 or last > 100000:
                continue
            spread = (tk['ask'] - tk['bid']) / last if last else np.inf
            vol_usd = tk.get('quoteVolume', 0)
            if vol_usd < 150_000:
                continue
            vol = get_symbol_volatility(ex, sym)
            if vol is None or vol < MIN_VOLATILITY_THRESHOLD or vol > MAX_VOLATILITY_THRESHOLD or spread > 0.007:
                continue
            score = (vol / MAX_VOLATILITY_THRESHOLD) * (1 - spread * 200)
            candidates.append({
                'symbol': sym,
                'volatility': vol,
                'spread': spread,
                'volume': vol_usd,
                'last': last,
                'score': score
            })

        candidates.sort(key=lambda x: -x['score'])
        for c in candidates[:max_symbols]:
            symbols.append(c['symbol'])
            symbol_data[c['symbol']] = {
                'volatility': c['volatility'],
                'price': c['last'],
                'spread': c['spread'],
                'volume': c['volume']
            }

    except Exception as e:
        console.print(f"[red]Error scanning symbols: {e}")
    return symbols

def calculate_grid_levels(price, num_levels, spacing_pct):
    half = num_levels // 2
    return sorted([price * (1 + (i * spacing_pct / 100)) for i in range(-half, half + 1)])

def place_grid_orders(ex, symbol, grid_levels, position_size_usd, leverage):
    orders = []
    try:
        safe_fetch(ex.set_leverage, leverage, symbol)
        size_usd = position_size_usd / len(grid_levels)
        for idx, lvl in enumerate(grid_levels):
            side = 'buy' if idx % 2 == 0 else 'sell'
            amount = (size_usd / lvl) * leverage
            o = safe_fetch(ex.create_limit_order, symbol, side, amount, lvl, {
                'timeInForce': 'GTC',
                'marginMode': 'cross'
            })
            if o:
                orders.append({
                    'id': o['id'],
                    'side': side,
                    'price': lvl,
                    'amount': amount,
                    'status': 'open',
                    'symbol': symbol
                })
                console.print(f"[green]Placed {side.upper()} {amount:.5f} @ {lvl:.5f}")
    except Exception as e:
        console.print(f"[red]Grid placement error: {e}")
    return orders

def monitor_grid(ex, symbol, grid_orders, position_size_usd):
    global total_profit, total_trades, total_locked_profit
    with thread_lock:
        active_grids[symbol] = {'orders': grid_orders, 'profit': 0.0, 'start': time.time()}
    while symbol in active_grids:
        try:
            for order in list(grid_orders):
                fetched = safe_fetch(ex.fetch_order, order['id'], order['symbol'])
                if fetched and fetched['status'] == 'closed':
                    pnl = (fetched['price'] - order['price']) * fetched['filled']
                    if order['side'] == 'sell':
                        pnl = -pnl
                    with thread_lock:
                        active_grids[symbol]['profit'] += pnl
                        total_profit += pnl
                        total_trades += 1
                    grid_orders.remove(order)

            current_profit = active_grids[symbol]['profit']

            if current_profit >= position_size_usd * TARGET_GRID_PROFIT:
                locked_profit = current_profit
                compound_amount = locked_profit * COMPOUND_PERCENT
                console.print(f"[cyan]LOCKED: {symbol} +${locked_profit:.4f} — Reinvesting ${compound_amount:.4f}")
                for o in grid_orders:
                    safe_fetch(ex.cancel_order, o['id'], symbol)
                with thread_lock:
                    total_locked_profit += locked_profit
                    active_grids.pop(symbol)
                threading.Thread(target=setup_grid_for_symbol, args=(ex, symbol, compound_amount), daemon=True).start()
                return

            if current_profit <= position_size_usd * STOP_LOSS_THRESHOLD:
                console.print(f"[red]STOPLOSS: {symbol} {current_profit:.4f}")
                for o in grid_orders:
                    safe_fetch(ex.cancel_order, o['id'], symbol)
                with thread_lock:
                    active_grids.pop(symbol)
                return

            time.sleep(5 + random.random() * 2)

        except Exception as e:
            console.print(f"[red]Monitor error {symbol}: {e}")
            time.sleep(10)

def display_dashboard():
    while True:
        with console.status("[bold green]Hydra Monolith: Breathing Profits..."):
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Symbol")
            table.add_column("Uptime (min)")
            table.add_column("Profit ($)")
            table.add_column("CPS ($/s)")
            table.add_column("Reinvested ($)")

            for symbol, metrics in active_grids.items():
                uptime = (time.time() - metrics['start']) / 60
                profit = metrics['profit']
                cps = profit / max(1, uptime * 60)
                table.add_row(
                    symbol,
                    f"{uptime:.2f}",
                    f"{profit:.4f}",
                    f"{cps:.5f}",
                    f"{profit * COMPOUND_PERCENT:.4f}"
                )
            
            global_cps = total_profit / max(1, (time.time() - start_time))
            console.clear()
            console.print(Panel.fit(table, title="🛡 HYDRA TEROIDED MONOLITH — LIVE"))
            console.print(f"[cyan]Total Profit: ${total_profit:.4f} | Total Locked: ${total_locked_profit:.4f} | Global CPS: ${global_cps:.5f}/s")
            
            time.sleep(10)

def setup_grid_for_symbol(ex, symbol, position_size_usd):
    try:
        price = safe_fetch(ex.fetch_ticker, symbol)['last']
        levels = calculate_grid_levels(price, 21, GRID_SPACING_PERCENT)
        orders = place_grid_orders(ex, symbol, levels, position_size_usd, LEVERAGE)
        threading.Thread(target=monitor_grid, args=(ex, symbol, orders, position_size_usd), daemon=True).start()
    except Exception as e:
        console.print(f"[red]Setup error {symbol}: {e}")

# === HYDRA MONOLITH LIVE ENGINE ===
if __name__ == "__main__":
    ex = create_exchange()
    start_time = time.time()

    console.print("[bold cyan]🚀 Launching LIVE TEROIDED MONOLITH...")
    threading.Thread(target=display_dashboard, daemon=True).start()

    while True:
        try:
            usdt_balance = safe_fetch(ex.fetch_balance, {'type': 'swap'})
            if not usdt_balance:
                console.print("[red]Balance fetch failed. Retrying...")
                time.sleep(10)
                continue
            
            free_balance = usdt_balance['USDT']['free']
            position_size_usd = (free_balance / MAX_SYMBOLS) * 0.9

            needed_symbols = MAX_SYMBOLS - len(active_grids)
            if needed_symbols > 0:
                symbols = get_best_symbols(ex, needed_symbols)
                for sym in symbols:
                    if sym not in active_grids:
                        setup_grid_for_symbol(ex, sym, position_size_usd)

            if free_balance > position_size_usd * 1.2:
                console.print("[yellow]🛠 More margin detected — expanding grids...")
                symbols = get_best_symbols(ex, 1)
                for sym in symbols:
                    if sym not in active_grids:
                        setup_grid_for_symbol(ex, sym, position_size_usd)

            time.sleep(REFRESH_INTERVAL)

        except Exception as e:
            console.print(f"[red]Global loop error: {e}")
            time.sleep(30)
