import sys
try:
    import ccxt.async_support as ccxt
except ImportError:
    sys.exit("Error: ccxt module not found. Please install it with `pip install ccxt` and rerun.")
import asyncio
import time
import threading
import numpy as np
import os
from datetime import datetime
from deap import base, creator, tools, algorithms
from rich.console import Console
from rich.live import Live
from rich.table import Table
active_grids = {}
thread_lock = threading.Lock()
total_profit = 0.0
total_trades = 0
profit_rate = 0.0
ga_iterations = 0
INITIAL_DEPOSIT = None
start_time = None
class GAParams:
    def __init__(self):
        self.MAX_SYMBOLS = 5
        self.LEVERAGE = 65
        self.BASE_SPACING_PERCENT = 0.22
        self.MIN_PROFIT_TARGET = 0.0045
        self.PROFIT_SCALING_FACTOR = 1.3
        self.MIN_VOLUME_USD = 450000
        self.MAX_SPREAD_PERCENT = 0.0018
        self.RISK_FACTOR = 0.82
        self.REBALANCE_INTERVAL = 780
        self.SYMBOL_ROTATION_INTERVAL = 1620
        self.DYNAMIC_GRID_INTERVAL = 240
        self.VOL_LOW = 0.0025
        self.VOL_HIGH = 0.055
        self.STOP_LOSS_THRESHOLD = -0.0015
        self.ACCELERATION_FACTOR = 1.2
params = GAParams()
console = Console()
def create_exchange():
    return ccxt.gateio({
        'apiKey': os.getenv("GATEIO_API_KEY"),
        'secret': os.getenv("GATEIO_API_SECRET"),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap', 'defaultSettle': 'usdt'}
    })
def fetch_volatility(symbol, ex, timeframe='5m', lookback=24):
    try:
        ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=lookback)
        closes = np.array([c[4] for c in ohlcv])
        returns = np.diff(np.log(closes))
        return np.std(returns) * np.sqrt(288)
    except:
        return None
def dynamic_spacing(vol):
    base = params.BASE_SPACING_PERCENT
    if vol < params.VOL_LOW * 1.5:
        return base * 0.7
    if vol > params.VOL_HIGH * 0.8:
        return base * 1.4
    vol_range = params.VOL_HIGH * 0.8 - params.VOL_LOW * 1.5
    pos = (vol - params.VOL_LOW * 1.5) / vol_range
    return base * (0.7 + pos * 0.7)
def dynamic_profit_target(vol, market_type, success_rate=0.75):
    base = params.MIN_PROFIT_TARGET
    vol_factor = min(1.0, vol / params.VOL_HIGH) * params.PROFIT_SCALING_FACTOR
    market_multiplier = 1.2 if market_type == 'futures' else 1.0
    success_factor = 1 + (0.5 - min(success_rate, 1.0)) * 0.5
    return base * vol_factor * market_multiplier * success_factor
def calculate_real_profit(amount, price_diff, fee_rate=0.0005):
    gross = amount * price_diff
    fees = amount * (fee_rate * 2)
    slippage = amount * price_diff * 0.0002
    return gross - fees - slippage
async def get_best_symbols(ex, max_symbols=5):
    symbols, candidates = [], []
    markets = await ex.load_markets()
    futures = [s for s in markets if '/USDT:USDT' in s and markets[s]['active']]
    for i in range(0, len(futures), 20):
        batch = futures[i:i+20]
        tickers = await ex.fetch_tickers(batch)
        for sym in batch:
            t = tickers.get(sym)
            if not t or not t.get('last'): continue
            spread = (t['ask'] - t['bid']) / t['last']
            if spread > params.MAX_SPREAD_PERCENT: continue
            vol = fetch_volatility(sym, ex)
            if vol is None or not (params.VOL_LOW <= vol <= params.VOL_HIGH): continue
            if t.get('quoteVolume', 0) < params.MIN_VOLUME_USD: continue
            vol_score = (vol - params.VOL_LOW) / (params.VOL_HIGH - params.VOL_LOW)
            spread_score = 1 - (spread / params.MAX_SPREAD_PERCENT)
            volume_score = min(1, t['quoteVolume'] / (params.MIN_VOLUME_USD * 10))
            score = (vol_score * 0.5 + spread_score * 0.3 + volume_score * 0.2) * 1.2
            candidates.append((sym, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in candidates[:max_symbols]]
async def place_grid_orders(ex, symbol, levels, pos_size, market_type):
    orders = []
    price_now = (await ex.fetch_ticker(symbol))['last']
    if market_type == 'futures':
        try: await ex.set_leverage(params.LEVERAGE, symbol)
        except: pass
    sorted_lv = sorted(levels, key=lambda p: abs(p - price_now))
    for idx, price in enumerate(sorted_lv):
        side = 'buy' if idx % 2 == 0 else 'sell'
        prox = 1 - idx / len(sorted_lv) / 2
        amt = pos_size / len(sorted_lv) / price_now * prox
        try:
            if market_type == 'futures':
                order = await ex.create_order(symbol, 'limit', side, amt * params.LEVERAGE, price)
            else:
                order = await ex.create_limit_order(symbol, side, amt, price)
            orders.append({'id': order['id'], 'price': price, 'side': side, 'amount': amt, 'status': 'open', 'timestamp': time.time()})
            await asyncio.sleep(0.1)
        except: pass
    return orders
async def monitor_grid(ex, symbol, orders, pos_size, market_type):
    global total_profit, total_trades, profit_rate
    grid_profit = 0.0
    trades = 0
    success = 0
    volatility = fetch_volatility(symbol, ex) or 0.01
    profit_target = dynamic_profit_target(volatility, market_type)
    pulse = time.time()
    active_grids[symbol] = {'orders': orders, 'profit': 0, 'trades': 0, 'success_rate': 0.5}
    while symbol in active_grids:
        open_ids = [o['id'] for o in await ex.fetch_open_orders(symbol)]
        for o in list(orders):
            if o['status'] == 'open' and o['id'] not in open_ids:
                o['status'] = 'filled'
                profit = calculate_real_profit(o['amount'], o['price'] * profit_target)
                side = 'sell' if o['side'] == 'buy' else 'buy'
                new_price = o['price'] * (1 + profit_target) if side == 'sell' else o['price'] * (1 - profit_target)
                try:
                    if market_type == 'futures':
                        no = await ex.create_order(symbol, 'limit', side, o['amount'] * params.LEVERAGE, new_price)
                    else:
                        no = await ex.create_limit_order(symbol, side, o['amount'], new_price)
                    orders.append({'id': no['id'], 'price': new_price, 'side': side, 'amount': o['amount'], 'status': 'open', 'timestamp': time.time()})
                    grid_profit += profit
                    trades += 1
                    success += 1
                    with thread_lock:
                        total_profit += profit
                        total_trades += 1
                    profit_rate = (total_profit * 100) / max(time.time() - start_time, 1)
                    active_grids[symbol].update(profit=grid_profit, trades=trades, success_rate=success / trades)
                except: pass
        if time.time() - pulse > params.DYNAMIC_GRID_INTERVAL:
            pulse = time.time()
            vol_now = fetch_volatility(symbol, ex) or volatility
            adj = 0.98 if vol_now < params.VOL_LOW * 1.2 else 1.02 if vol_now > params.VOL_HIGH * 0.8 else 1
            profit_target = np.clip(profit_target * adj, params.MIN_PROFIT_TARGET, 0.01)
        if grid_profit > 0.2 * pos_size:
            console.print(f"{symbol} hit 20% ROI. Sleeping 6h.")
            for o in orders:
                if o['status'] == 'open':
                    try: await ex.cancel_order(o['id'], symbol)
                    except: pass
            del active_grids[symbol]
            await asyncio.sleep(21600)
            return
        await asyncio.sleep(2)
async def setup_symbol(ex, symbol, pos_size):
    mt = 'futures' if ':USDT' in symbol else 'spot'
    price = (await ex.fetch_ticker(symbol))['last']
    vol = fetch_volatility(symbol, ex) or 0.01
    spacing = dynamic_spacing(vol)
    levels = [price * (1 + i * spacing / 100) for i in range(-4, 5)]
    orders = await place_grid_orders(ex, symbol, levels, pos_size, mt)
    asyncio.create_task(monitor_grid(ex, symbol, orders, pos_size, mt))
async def rotate_symbols(ex):
    while True:
        await asyncio.sleep(params.SYMBOL_ROTATION_INTERVAL)
        console.print("Rotating Symbols...")
        syms = await get_best_symbols(ex, params.MAX_SYMBOLS)
        ps = (INITIAL_DEPOSIT - 30.0) / params.MAX_SYMBOLS
        for s in list(active_grids):
            if s not in syms:
                console.print(f"Killing underperformer {s}")
                try: await ex.cancel_all_orders(s)
                except: pass
                del active_grids[s]
        for s in syms:
            if s not in active_grids:
                console.print(f"Spawning new symbol {s}")
                await setup_symbol(ex, s, ps)
def setup_ga():
    tb = base.Toolbox()
    tb.register("population", lambda n: [])
    tb.register("select", tools.selBest)
    tb.register("varAnd", algorithms.varAnd)
    return tb
def evaluate_individual(ind, metrics):
    return (0.0,)
def apply_ga_params(ind):
    pass
async def dynamic_ga_evolver():
    global ga_iterations
    tb = setup_ga()
    pop = tb.population(10)
    while True:
        await asyncio.sleep(1800)
        ga_iterations += 1
        fits = [evaluate_individual(i, [profit_rate]) for i in pop]
        for i, f in zip(pop, fits): i.fitness.values = f
        off = tb.varAnd(pop, cxpb=0.5, mutpb=0.2)
        fits2 = [evaluate_individual(i, [profit_rate]) for i in off]
        for i, f in zip(off, fits2): i.fitness.values = f
        pop = tb.select(pop + off, k=len(pop))
        best = tb.select(pop, k=1)[0]
        apply_ga_params(best)
        await asyncio.sleep(0)
async def display_dashboard():
    while True:
        table = Table(title="Gigalithimonous Status")
        table.add_column("Sym")
        table.add_column("Trades")
        table.add_column("Profit")
        table.add_column("Success %")
        with thread_lock:
            for s, d in active_grids.items():
                table.add_row(s, str(d['trades']), f"${d['profit']:.4f}", f"{d['success_rate']*100:.1f}%")
            elapsed = (time.time() - start_time) / 60
            overview = f"⏱{elapsed:.1f}m | 💰${total_profit:.4f} | 🔥{profit_rate:.3f} CPS | ⚔{total_trades} Trades"
            style = "green" if profit_rate >= 0.09 else "yellow" if profit_rate >= 0.045 else "red"
        console.print(overview, style=style)
        console.print(table)
        await asyncio.sleep(5)
async def main():
    global INITIAL_DEPOSIT, start_time
    console.print("🚀 Gigalithimonous GA^RL^GA^3 INITIATED")
    ex = create_exchange()
    await ex.load_markets()
    INITIAL_DEPOSIT = (await ex.fetch_balance())['total']['USDT']
    start_time = time.time()
    syms = await get_best_symbols(ex, params.MAX_SYMBOLS)
    ps = (INITIAL_DEPOSIT - 30.0) / params.MAX_SYMBOLS
    for s in syms: await setup_symbol(ex, s, ps)
    await asyncio.gather(rotate_symbols(ex), dynamic_ga_evolver(), display_dashboard())
if __name__ == "__main__":
    asyncio.run(main())
import unittest
class TestUtils(unittest.TestCase):
    def test_calculate_real_profit(self):
        amt, diff, fee = 100, 0.01, 0.001
        expected = amt * diff - amt * (fee * 2) - amt * diff * 0.0002
        self.assertAlmostEqual(calculate_real_profit(amt, diff, fee), expected)
    def test_dynamic_spacing(self):
        params.BASE_SPACING_PERCENT = 1.0
        self.assertAlmostEqual(dynamic_spacing(params.VOL_LOW * 0.5), 0.7)
        self.assertAlmostEqual(dynamic_spacing(params.VOL_HIGH * 0.9), 1.4)
        mid = (params.VOL_LOW * 1.5 + params.VOL_HIGH * 0.8) / 2
        self.assertTrue(0.7 < dynamic_spacing(mid) < 1.4)
    def test_dynamic_profit_target(self):
        val = dynamic_profit_target(params.VOL_HIGH, 'spot', 1.0)
        self.assertTrue(val >= params.MIN_PROFIT_TARGET)
if __name__ == '__main__':
    unittest.main()

