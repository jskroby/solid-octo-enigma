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

def calculate_real_profit(amount, price_diff, fee_rate=0.0005):
    gross = amount * price_diff
    fees = amount * (fee_rate * 2)
    slippage = amount * price_diff * 0.0002
    return gross - fees - slippage

def dynamic_spacing(vol, base_spacing=0.22, vol_low=0.0025, vol_high=0.055):
    if vol < vol_low * 1.5:
        return base_spacing * 0.7
    if vol > vol_high * 0.8:
        return base_spacing * 1.4
    vol_range = vol_high * 0.8 - vol_low * 1.5
    pos = (vol - vol_low * 1.5) / vol_range
    return base_spacing * (0.7 + pos * 0.7)

def dynamic_profit_target(vol, market_type='spot', success_rate=0.75,
                         min_profit=0.0045, vol_high=0.055, scaling=1.3):
    base = min_profit
    vol_factor = min(1.0, vol / vol_high) * scaling
    market_multiplier = 1.2 if market_type == 'futures' else 1.0
    success_factor = 1 + (0.5 - min(success_rate, 1.0)) * 0.5
    calculated = base * vol_factor * market_multiplier * success_factor
    return max(base, calculated)

def create_exchange():
    try:
        import ccxt.async_support as ccxt_async
    except ImportError:
        raise ImportError("ccxt module not found. Install with pip install ccxt.")
    return ccxt_async.gateio({
        'apiKey': os.getenv("GATEIO_API_KEY"),
        'secret': os.getenv("GATEIO_API_SECRET"),
        'enableRateLimit': True,
        'options': {'defaultType': 'swap', 'defaultSettle': 'usdt'}
    })

active_grids = {}
thread_lock = threading.Lock()
total_profit = 0.0
total_trades = 0
profit_rate = 0.0
ga_iterations = 0
INITIAL_DEPOSIT = 0.0
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
        self.SYMBOL_ROTATION_INTERVAL = 1620
        self.DYNAMIC_GRID_INTERVAL = 240
        self.VOL_LOW = 0.0025
        self.VOL_HIGH = 0.055

params = GAParams()
console = Console()

async def get_best_symbols(ex, max_symbols=5):
    candidates = []
    markets = await ex.load_markets()
    futures = [s for s in markets if "/USDT:USDT" in s and markets[s]['active']]
    for i in range(0, len(futures), 20):
        batch = futures[i:i+20]
        tickers = await ex.fetch_tickers(batch)
        for s in batch:
            t = tickers.get(s)
            if not t or not t.get('last'):
                continue
            spread = (t['ask'] - t['bid']) / t['last']
            if spread > params.MAX_SPREAD_PERCENT:
                continue
            vol = None
            try:
                o = await ex.fetch_ohlcv(s, '5m', limit=24)
                closes = np.array([c[4] for c in o])
                returns = np.diff(np.log(closes))
                vol = np.std(returns) * np.sqrt(288)
            except:
                continue
            if vol < params.VOL_LOW or vol > params.VOL_HIGH:
                continue
            volume_usd = t.get('quoteVolume', 0)
            if volume_usd < params.MIN_VOLUME_USD:
                continue
            vol_score = (vol - params.VOL_LOW) / (params.VOL_HIGH - params.VOL_LOW)
            spread_score = 1 - (spread / params.MAX_SPREAD_PERCENT)
            volume_score = min(1, volume_usd / (params.MIN_VOLUME_USD * 10))
            score = (vol_score * 0.5 + spread_score * 0.3 + volume_score * 0.2) * 1.2
            candidates.append((s, score))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in candidates[:max_symbols]]

async def place_grid_orders(ex, symbol, levels, pos_size, market_type):
    orders = []
    p = (await ex.fetch_ticker(symbol))['last']
    if market_type == 'futures':
        try:
            await ex.set_leverage(params.LEVERAGE, symbol)
        except:
            pass
    lvl_sorted = sorted(enumerate(levels), key=lambda x: abs(x[1] - p))
    for idx, (li, price) in enumerate(lvl_sorted):
        side = 'buy' if li % 2 == 0 else 'sell'
        prox = 1 - (idx / len(levels)) * 0.5
        amt = pos_size / len(levels) / p * prox
        try:
            if market_type == 'futures':
                o = await ex.create_order(symbol, 'limit', side, amt * params.LEVERAGE, price)
            else:
                o = await ex.create_limit_order(symbol, side, amt, price)
            orders.append({'id': o['id'], 'price': price, 'side': side, 'amount': amt, 'status': 'open', 'ts': time.time()})
            await asyncio.sleep(0.1)
        except:
            pass
    return orders

async def monitor_grid(ex, symbol, orders, pos_size, market_type):
    global total_profit, total_trades, profit_rate
    grid_profit = 0.0
    tcount = 0
    scount = 0
    vol = params.VOL_LOW
    profit_target = params.MIN_PROFIT_TARGET
    active_grids[symbol] = {'orders': orders, 'profit': 0, 'trades': 0, 'success_rate': 0}
    pulse = time.time()
    while symbol in active_grids:
        try:
            open_ids = [o['id'] for o in await ex.fetch_open_orders(symbol)]
            for o in list(orders):
                if o['status'] == 'open' and o['id'] not in open_ids:
                    o['status'] = 'filled'
                    vol = calculate_real_profit(o['amount'], o['price'] * profit_target)
                    real_prof = vol
                    side = 'sell' if o['side'] == 'buy' else 'buy'
                    price_new = o['price'] * (1 + profit_target) if side == 'sell' else o['price'] * (1 - profit_target)
                    try:
                        if market_type == 'futures':
                            n = await ex.create_order(symbol, 'limit', side, o['amount'] * params.LEVERAGE, price_new)
                        else:
                            n = await ex.create_limit_order(symbol, side, o['amount'], price_new)
                        orders.append({'id': n['id'], 'price': price_new, 'side': side, 'amount': o['amount'], 'status': 'open', 'ts': time.time()})
                        grid_profit += real_prof
                        tcount += 1
                        scount += 1
                        with thread_lock:
                            total_profit += real_prof
                            total_trades += 1
                        elapsed = time.time() - start_time
                        profit_rate = (total_profit * 100) / max(elapsed, 1)
                        active_grids[symbol].update({'profit': grid_profit, 'trades': tcount, 'success_rate': scount / tcount})
                    except:
                        pass
            if time.time() - pulse > params.DYNAMIC_GRID_INTERVAL:
                pulse = time.time()
                try:
                    ohlcv = await ex.fetch_ohlcv(symbol, '5m', limit=24)
                    closes = np.array([c[4] for c in ohlcv])
                    returns = np.diff(np.log(closes))
                    current_vol = np.std(returns) * np.sqrt(288)
                except:
                    current_vol = vol
                adj = 0.98 if current_vol < params.VOL_LOW * 1.2 else 1.02 if current_vol > params.VOL_HIGH * 0.8 else 1
                profit_target = np.clip(profit_target * adj, params.MIN_PROFIT_TARGET, 0.01)
            if grid_profit > 0.2 * pos_size:
                for o in orders:
                    if o['status'] == 'open':
                        try: await ex.cancel_order(o['id'], symbol)
                        except: pass
                del active_grids[symbol]
                await asyncio.sleep(21600)
                return
        except:
            await asyncio.sleep(5)
        await asyncio.sleep(2)

async def setup_symbol(ex, symbol, pos_size):
    market = 'futures' if ':USDT' in symbol else 'spot'
    p = (await ex.fetch_ticker(symbol))['last']
    try:
        o = await ex.fetch_ohlcv(symbol, '5m', limit=24)
        c = np.array([x[4] for x in o])
        vol = np.std(np.diff(np.log(c))) * np.sqrt(288)
    except:
        vol = params.VOL_LOW
    spacing = dynamic_spacing(vol)
    levels = [p * (1 + (i * spacing / 100)) for i in range(-4, 5)]
    orders = await place_grid_orders(ex, symbol, levels, pos_size, market)
    asyncio.create_task(monitor_grid(ex, symbol, orders, pos_size, market))

async def rotate_symbols(ex):
    while True:
        await asyncio.sleep(params.SYMBOL_ROTATION_INTERVAL)
        symbols = await get_best_symbols(ex, params.MAX_SYMBOLS)
        pos_size = (INITIAL_DEPOSIT - 30.0) / params.MAX_SYMBOLS
        for s in list(active_grids):
            if s not in symbols:
                try: await ex.cancel_all_orders(s)
                except: pass
                del active_grids[s]
        for s in symbols:
            if s not in active_grids:
                asyncio.create_task(setup_symbol(ex, s, pos_size))

async def dynamic_ga_evolver():
    global ga_iterations
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = tools.Toolbox()
    toolbox.register("attr_float", np.random.rand)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=5)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (profit_rate,))
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    population = toolbox.population(n=10)
    while True:
        await asyncio.sleep(1800)
        ga_iterations += 1
        fits = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fits): ind.fitness.values = fit
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits_off = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fits_off): ind.fitness.values = fit
        population = toolbox.select(population + offspring, k=len(population))
        best = tools.selBest(population, 1)[0]
        await asyncio.sleep(0)

async def display_dashboard():
    while True:
        table = Table(title="Gigalithimonous Status")
        table.add_column("Symbol")
        table.add_column("Trades")
        table.add_column("Profit")
        table.add_column("Success %")
        with thread_lock:
            for sym, d in active_grids.items():
                table.add_row(sym, str(d['trades']), f"${d['profit']:.4f}", f"{d['success_rate']*100:.1f}%")
            elapsed = (time.time() - start_time) / 60
            overview = f"⏱{elapsed:.1f}m | 💰${total_profit:.4f} | 🔥{profit_rate:.3f} CPS | ⚔{total_trades} Trades"
            style = "green" if profit_rate >= 0.09 else "yellow" if profit_rate >= 0.045 else "red"
        console.print(overview, style=style)
        console.print(table)
        await asyncio.sleep(5)

async def main():
    global INITIAL_DEPOSIT, start_time
    start_time = time.time()
    ex = create_exchange()
    await ex.load_markets()
    bal = await ex.fetch_balance()
    INITIAL_DEPOSIT = bal['total'].get('USDT', 0)
    symbols = await get_best_symbols(ex, params.MAX_SYMBOLS)
    pos_size = (INITIAL_DEPOSIT - 30.0) / params.MAX_SYMBOLS
    for sym in symbols:
        asyncio.create_task(setup_symbol(ex, sym, pos_size))
    await asyncio.gather(rotate_symbols(ex), dynamic_ga_evolver(), display_dashboard())

if __name__ == '__main__':
    asyncio.run(main())

import unittest

class TestUtils(unittest.TestCase):
    def test_calculate_real_profit(self):
        self.assertAlmostEqual(calculate_real_profit(100, 0.01, 0.001), 100*0.01 - 100*(0.001*2) - 100*0.01*0.0002)
    def test_dynamic_spacing(self):
        self.assertAlmostEqual(dynamic_spacing(params.VOL_LOW*0.5, base_spacing=1.0, vol_low=params.VOL_LOW, vol_high=params.VOL_HIGH), 0.7)
        self.assertAlmostEqual(dynamic_spacing(params.VOL_HIGH*0.9, base_spacing=1.0, vol_low=params.VOL_LOW, vol_high=params.VOL_HIGH), 1.4)
        mid = (params.VOL_LOW*1.5 + params.VOL_HIGH*0.8)/2
        self.assertTrue(0.7 < dynamic_spacing(mid, base_spacing=1.0, vol_low=params.VOL_LOW, vol_high=params.VOL_HIGH) < 1.4)
    def test_dynamic_profit_target(self):
        self.assertGreaterEqual(dynamic_profit_target(params.VOL_HIGH, 'spot', 1.0, params.MIN_PROFIT_TARGET, params.VOL_HIGH, params.PROFIT_SCALING_FACTOR), params.MIN_PROFIT_TARGET)
        self.assertGreater(dynamic_profit_target(params.VOL_HIGH, 'futures', 1.0, params.MIN_PROFIT_TARGET, params.VOL_HIGH, params.PROFIT_SCALING_FACTOR), dynamic_profit_target(params.VOL_HIGH, 'spot', 1.0, params.MIN_PROFIT_TARGET, params.VOL_HIGH, params.PROFIT_SCALING_FACTOR))

if __name__ == '__main__':
    unittest.main()

