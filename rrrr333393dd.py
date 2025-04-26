# === IMPORTS ===
import ccxt.async_support as ccxt
import asyncio
import numpy as np
import os
import time
import threading
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from datetime import datetime

# === SETUP ===
API_KEY = os.getenv("GATEIO_API_KEY")
API_SECRET = os.getenv("GATEIO_API_SECRET")
SAFETY_BUFFER = 30.0
TARGET_PROFIT_RATE = 0.03  # 3 cents per second

console = Console()
thread_lock = threading.Lock()
start_time = None
total_profit = 0
total_trades = 0
profit_rate = 0
active_grids = {}

# === PARAMETERS ===
class GAParams:
    def __init__(self):
        self.MAX_SYMBOLS = 5
        self.LEVERAGE = 50
        self.BASE_SPACING_PERCENT = 0.22
        self.MIN_PROFIT_TARGET = 0.0045
        self.PROFIT_SCALING_FACTOR = 1.3
        self.MIN_VOLUME_USD = 450000
        self.MAX_SPREAD_PERCENT = 0.0018
        self.VOL_LOW = 0.0025
        self.VOL_HIGH = 0.055
        self.SYMBOL_ROTATION_INTERVAL = 1800
        self.STOP_LOSS_THRESHOLD = -0.002
        self.REBATE_PER_FILL = 0.0002  # 0.02% Maker Rebate

params = GAParams()

# === EXCHANGE SETUP ===
def create_exchange():
    return ccxt.gateio({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap', 'defaultSettle': 'usdt'}
    })

# === UTILS ===
async def fetch_volatility(ex, symbol, timeframe='5m', lookback=24):
    try:
        ohlcv = await ex.fetch_ohlcv(symbol, timeframe, limit=lookback)
        closes = np.array([c[4] for c in ohlcv])
        returns = np.diff(np.log(closes))
        return np.std(returns) * np.sqrt(288)
    except:
        return None

def calculate_real_profit(amount, price_diff, fee_rate=0.0005):
    gross = amount * price_diff
    maker_rebate = amount * params.REBATE_PER_FILL
    fees = amount * (fee_rate * 2)
    slippage = amount * price_diff * 0.0001
    return gross + maker_rebate - fees - slippage

def dynamic_spacing(volatility):
    base = params.BASE_SPACING_PERCENT
    return base * (0.7 + (volatility - params.VOL_LOW) / (params.VOL_HIGH - params.VOL_LOW))

async def get_best_symbols(ex, max_symbols=5):
    try:
        await ex.load_markets()
        futures = [s for s in ex.symbols if 'USDT:USDT' in s]
        candidates = []
        for symbol in futures:
            ticker = await ex.fetch_ticker(symbol)
            vol = await fetch_volatility(ex, symbol)
            if not ticker['ask'] or not ticker['bid']:
                continue
            spread = (ticker['ask'] - ticker['bid']) / ticker['last']
            if spread < params.MAX_SPREAD_PERCENT and vol and params.VOL_LOW < vol < params.VOL_HIGH:
                candidates.append((symbol, vol))
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in candidates[:max_symbols]]
    except Exception as e:
        console.print(f"[red]Error fetching symbols: {e}")
        return []

# === GRID OPS ===
async def place_grid_orders(ex, symbol, grid_levels, pos_size):
    orders = []
    price_now = (await ex.fetch_ticker(symbol))['last']
    await ex.set_leverage(params.LEVERAGE, symbol)
    center_idx = len(grid_levels) // 2
    for idx, price in enumerate(sorted(grid_levels)):
        side = 'buy' if idx % 2 == 0 else 'sell'
        level_size = pos_size / len(grid_levels) / price_now
        try:
            order = await ex.create_order(symbol, 'limit', side, level_size * params.LEVERAGE, price)
            orders.append({
                'id': order['id'],
                'price': price,
                'side': side,
                'amount': level_size,
                'status': 'open',
                'symbol': symbol
            })
            await asyncio.sleep(0.1)
        except:
            pass
    return orders

async def breathing_grid_adjust(symbol, ex, orders, volatility):
    spacing_multiplier = 1.0
    if volatility < 0.003:
        spacing_multiplier = 0.8
    elif volatility > 0.05:
        spacing_multiplier = 1.2
    for o in orders:
        if o['status'] == 'open':
            original_price = o['price']
            delta = original_price * ((spacing_multiplier - 1) * 0.01)
            o['price'] += delta
            try:
                await ex.edit_order(o['id'], o['symbol'], price=o['price'])
            except:
                pass

async def monitor_grid(ex, symbol, orders, pos_size):
    global total_profit, total_trades, profit_rate
    volatility = await fetch_volatility(ex, symbol) or 0.01
    profit_target = params.MIN_PROFIT_TARGET
    grid_profit = 0
    trade_count = 0
    pulse_time = time.time()
    active_grids[symbol] = {"orders": orders, "profit": 0, "trades": 0}

    while symbol in active_grids:
        try:
            open_ids = [o['id'] for o in await ex.fetch_open_orders(symbol)]
            for order in list(orders):
                if order['status'] == 'open' and order['id'] not in open_ids:
                    order['status'] = 'filled'
                    side = 'sell' if order['side'] == 'buy' else 'buy'
                    new_price = order['price'] * (1 + profit_target) if side == 'sell' else order['price'] * (1 - profit_target)
                    real_profit = calculate_real_profit(order['amount'], order['price'] * profit_target)
                    grid_profit += real_profit
                    trade_count += 1
                    with thread_lock:
                        total_profit += real_profit
                        total_trades += 1
                        elapsed = time.time() - start_time
                        profit_rate = (total_profit * 100) / max(elapsed, 1)
                        active_grids[symbol]['profit'] = grid_profit
                        active_grids[symbol]['trades'] = trade_count
                    try:
                        new_order = await ex.create_order(symbol, 'limit', side, order['amount'] * params.LEVERAGE, new_price)
                        orders.append({
                            'id': new_order['id'],
                            'price': new_price,
                            'side': side,
                            'amount': order['amount'],
                            'status': 'open',
                            'symbol': symbol
                        })
                    except:
                        pass
                    rprint(f"[bold green]💸 Trade executed {symbol}: +${real_profit:.4f} profit[/bold green]")

            if time.time() - pulse_time > 30:
                pulse_time = time.time()
                vol_now = await fetch_volatility(ex, symbol) or volatility
                await breathing_grid_adjust(symbol, ex, orders, vol_now)

            if grid_profit > 0.20 * pos_size:
                rprint(f"[blue]💤 {symbol} hit 20% ROI. Sleeping 6h.[/blue]")
                for o in orders:
                    if o['status'] == 'open':
                        try:
                            await ex.cancel_order(o['id'], symbol)
                        except:
                            pass
                del active_grids[symbol]
                await asyncio.sleep(21600)
                return

        except Exception as e:
            console.print(f"[red]Grid monitor error: {e}[/red]")
            await asyncio.sleep(5)
        await asyncio.sleep(2)

# === MASTER SETUP ===
async def setup_symbol(ex, symbol, pos_size):
    price_now = (await ex.fetch_ticker(symbol))['last']
    volatility = await fetch_volatility(ex, symbol) or 0.01
    spacing = dynamic_spacing(volatility)
    grid_levels = [price_now * (1 + (i * spacing / 100)) for i in range(-4, 5)]
    orders = await place_grid_orders(ex, symbol, grid_levels, pos_size)
    asyncio.create_task(monitor_grid(ex, symbol, orders, pos_size))

async def main():
    global start_time
    ex = create_exchange()
    await ex.load_markets()
    balance = (await ex.fetch_balance())['total']['USDT']
    pos_size = (balance - SAFETY_BUFFER) / params.MAX_SYMBOLS
    start_time = time.time()
    symbols = await get_best_symbols(ex, params.MAX_SYMBOLS)
    for sym in symbols:
        await setup_symbol(ex, sym, pos_size)
    while True:
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
