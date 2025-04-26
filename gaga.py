import asyncio
import ccxt.async_support as ccxt
import os, time, random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from deap import base, creator, tools
from rich.console import Console
from rich.table import Table
from rich.live import Live
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# ─── ENV & GLOBALS ──────────────────────────────────────────────────────────
console       = Console()
active_grids  = {}
total_profit  = 0.0
total_trades  = 0
profit_rate   = 0.0
ga_iterations = 0
INITIAL_BAL   = 0.0
START_TIME    = time.time()

# ─── STRATEGY PARAMS ─────────────────────────────────────────────────────────
BASE_POS_USD      = 2.0      
DCA_LEVELS_MAX    = 50       
DCA_SPAN_PCT      = 10.0     
LEVERAGE          = 50
TARGET_CPS        = 0.09     
SYMBOL_ALLOC      = 0.1      
GA_EVOLVE_SECS    = 900      
STOP_LOSS_TRIGGER = 0.08     # % move from grid center to trigger exit

# ─── UTILS ───────────────────────────────────────────────────────────────────
def create_exchange():
    return ccxt.gateio({
        'apiKey':    os.getenv("GATEIO_API_KEY"),
        'secret':    os.getenv("GATEIO_API_SECRET"),
        'enableRateLimit': True,
        'timeout': 20000,
        'options': {'defaultType': 'swap', 'defaultSettle': 'usdt'}
    })

def make_levels(mid_price, n, span_pct):
    step = span_pct / n
    start = mid_price * (1 - span_pct/100)
    return [start + i*(step/100)*mid_price for i in range(n)]

def update_cps():
    global profit_rate
    elapsed = time.time() - START_TIME
    profit_rate = (total_profit * 100) / max(elapsed, 1)

# ─── GA SETUP ─────────────────────────────────────────────────────────────────
def setup_ga():
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    tb = base.Toolbox()
    tb.register("attr_float", random.random)
    tb.register("individual", tools.initRepeat, creator.Individual, tb.attr_float, 3)
    tb.register("population", tools.initRepeat, list, tb.individual)
    tb.register("mate", tools.cxBlend, alpha=0.5)
    tb.register("mutate", tools.mutPolynomialBounded, eta=0.5, low=0.0, up=1.0, indpb=0.2)
    tb.register("select", tools.selTournament, tournsize=3)
    tb.register("evaluate", lambda ind: (sum(ind),))
    return tb

# ─── ORDER MANAGEMENT ────────────────────────────────────────────────────────
async def place_grid_orders(ex, symbol, levels):
    mkt = ex.markets[symbol]
    min_amt = mkt['limits']['amount']['min']
    ticker = await ex.fetch_ticker(symbol)
    p0 = ticker['last']

    await ex.set_leverage(LEVERAGE, symbol)
    await asyncio.sleep(0.1)

    size = max(BASE_POS_USD * LEVERAGE / p0 / len(levels), min_amt)

    orders = []
    for price in levels:
        try:
            o = await ex.create_limit_buy_order(symbol, size, price)
            orders.append({'id': o['id'], 'price': price, 'amount': size, 'status': 'open'})
            await asyncio.sleep(0.05)
        except Exception as e:
            console.print(f"[red]Order failed {symbol}@{price}: {e}")
    return orders

async def monitor_grid(ex, symbol, orders, alloc_usd):
    global total_profit, total_trades
    grid_profit = 0.0
    center_price = np.mean([o['price'] for o in orders])

    active_grids[symbol] = {'orders': orders, 'profit': 0.0, 'trades': 0, 'center': center_price}

    while True:
        try:
            ticker = await ex.fetch_ticker(symbol)
            price = ticker['last']
            fetched = await ex.fetch_open_orders(symbol)
            open_ids = {o['id'] for o in fetched} if fetched else set()

            for o in list(orders):
                if o['status'] == 'open' and o['id'] not in open_ids:
                    fill_price = o['price']
                    pnl = (price - fill_price) * o['amount'] * LEVERAGE
                    grid_profit += pnl
                    total_profit += pnl
                    total_trades += 1
                    o['status'] = 'filled'

            update_cps()
            active_grids[symbol]['profit'] = grid_profit
            active_grids[symbol]['trades'] += 1

            move_pct = abs(price - center_price) / center_price
            if move_pct > STOP_LOSS_TRIGGER:
                console.print(f"[red]⛔ {symbol} exit triggered by price move {move_pct*100:.2f}%")
                await close_positions(ex, symbol)
                return

            if grid_profit >= 0.25 * alloc_usd:
                console.print(f"[cyan]✅ {symbol} profit lock triggered")
                await close_positions(ex, symbol)
                return

            if not fetched:
                console.print(f"[yellow]⚠️  {symbol} lost all orders, rebuilding grid...")
                new_orders = await place_grid_orders(ex, symbol, make_levels(price, len(orders), DCA_SPAN_PCT))
                active_grids[symbol]['orders'] = new_orders

            await asyncio.sleep(3)
        except Exception as e:
            console.print(f"[red]Monitor error {symbol}: {e}")
            await asyncio.sleep(5)

async def close_positions(ex, symbol):
    try:
        pos = await ex.fetch_positions([symbol])
        for p in pos:
            amt = p['contracts']
            if amt > 0:
                await ex.create_market_sell_order(symbol, amt)
                console.print(f"[red]❗ Closed long {symbol} position {amt}")
    except Exception as e:
        console.print(f"[red]❌ Failed to close {symbol}: {e}")

# ─── GA EVOLVER ───────────────────────────────────────────────────────────────
async def dynamic_ga_evolver():
    global ga_iterations
    toolbox = setup_ga()
    pop = toolbox.population(n=10)

    while True:
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))
        
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < 0.5:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < 0.2:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        pop[:] = offspring
        ga_iterations += 1
        console.print(f"[magenta]🧬 GA Gen {ga_iterations} evolved")
        await asyncio.sleep(GA_EVOLVE_SECS)

# ─── DASHBOARD ────────────────────────────────────────────────────────────────
async def display_dashboard():
    with Live(refresh_per_second=1) as live:
        while True:
            table = Table(title="⚔ HyperGrid v3.1")
            table.add_column("Symbol")
            table.add_column("Profit")
            table.add_column("Trades")
            for sym, data in active_grids.items():
                table.add_row(sym, f"{data['profit']:.4f}", str(data['trades']))
            footer = f"⏱ {(time.time()-START_TIME)/60:.1f}m | 💰{total_profit:.4f} | 🔥{profit_rate:.4f}c/s | ⚔{total_trades}"
            live.update(table)
            console.print(footer, style="green" if profit_rate >= TARGET_CPS else "yellow")
            await asyncio.sleep(5)

# ─── MAIN ─────────────────────────────────────────────────────────────────────
async def main():
    global INITIAL_BAL, START_TIME
    console.print("[bold cyan]🚀 HyperGrid v3.1 — Fixed & Battle-Ready")
    ex = create_exchange()
    await ex.load_markets()
    bal = await ex.fetch_balance()
    INITIAL_BAL = bal['total'].get('USDT', 0)
    START_TIME = time.time()

    universe = [s for s, d in ex.markets.items() if '/USDT:USDT' in s and d.get('active')]
    console.print(f"[bold green]Found {len(universe)} tradable markets")

    for sym in universe:
        await setup_symbol(ex, sym, INITIAL_BAL)
        await asyncio.sleep(0.2)

    await asyncio.gather(
        dynamic_ga_evolver(),
        display_dashboard()
    )

async def setup_symbol(ex, symbol, balance):
    try:
        ticker = await ex.fetch_ticker(symbol)
        p0 = ticker['last']
        alloc_usd = balance * SYMBOL_ALLOC
        min_ct = ex.markets[symbol]['limits']['amount']['min']

        max_levels = int((alloc_usd * LEVERAGE) / (p0 * min_ct))
        n_lv = max(1, min(max_levels, DCA_LEVELS_MAX))

        levels = make_levels(p0, n_lv, DCA_SPAN_PCT)
        orders = await place_grid_orders(ex, symbol, levels)
        if orders:
            asyncio.create_task(monitor_grid(ex, symbol, orders, alloc_usd))
    except Exception as e:
        console.print(f"[red]setup_symbol error {symbol}: {e}")

if __name__ == "__main__":
    asyncio.run(main())
