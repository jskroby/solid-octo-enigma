import os
import asyncio
import json
import time
import random
import ccxt
import numpy as np
import websockets
from rich.console import Console
from rich.live import Live
from rich.table import Table

console = Console()

# ───── CONFIGURATION ─────────────────────────────────────────────────────
GATE_API_KEY    = os.getenv("GATE_API_KEY")
GATE_API_SECRET = os.getenv("GATE_API_SECRET")
exchange = ccxt.gateio({
    'apiKey':    GATE_API_KEY,
    'secret':    GATE_API_SECRET,
    'enableRateLimit': True,
    'options': {'defaultType':'swap'}
})

# Websocket endpoint
WS_URI = "wss://api.gateio.ws/ws/v4/"

# RL bandit parameters
EPSILON = 0.1

# GA hyperparameters
POP_SIZE      = 10
GENERATIONS   = 5
MUTATION_RATE = 0.2

# Trading params
NOTIONAL_PER     = float(os.getenv("NOTIONAL_PER", 2000000))  # USD notional per pair
REBATE_SPREAD    = float(os.getenv("REBATE_SPREAD", 0.0003))
MICRO_STEP       = float(os.getenv("MICRO_STEP", 0.001))
MICRO_PCT        = float(os.getenv("MICRO_PCT", 0.02))
REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", 60))

open_orders = {}
positions   = {}

# ───── BANDIT AGENT ─────────────────────────────────────────────────────
class BanditAgent:
    def __init__(self, arms):
        self.arms   = arms
        self.counts = {a: 0 for a in arms}
        self.values = {a: 0.0 for a in arms}
    def select(self):
        if random.random() < EPSILON:
            return random.choice(self.arms)
        return max(self.arms, key=lambda a: self.values[a])
    def update(self, arm, reward):
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n

# ───── GA FOR HYPERPARAM EVOLUTION ──────────────────────────────────────
def init_population():
    return [
        {
            "grid_width": random.uniform(0.002, 0.02),
            "tp_pct":     random.uniform(0.001, 0.01),
            "offset":     random.uniform(0.001, 0.01),
        }
        for _ in range(POP_SIZE)
    ]

def fitness(cfg, avg_spread):
    return avg_spread - cfg["grid_width"]

def evolve(pop, spreads):
    avg_spread = np.mean(list(spreads.values()))
    fits = [fitness(ind, avg_spread) for ind in pop]
    ranked = sorted(zip(pop, fits), key=lambda x: x[1], reverse=True)
    elites = [ind for ind,_ in ranked[:POP_SIZE//2]]
    new_pop = elites.copy()
    while len(new_pop) < POP_SIZE:
        p1, p2 = random.sample(elites, 2)
        child = {
            "grid_width": random.choice([p1["grid_width"], p2["grid_width"]]),
            "tp_pct":     random.choice([p1["tp_pct"], p2["tp_pct"]]),
            "offset":     random.choice([p1["offset"], p2["offset"]]),
        }
        if random.random() < MUTATION_RATE:
            key = random.choice(list(child.keys()))
            child[key] *= random.uniform(0.9, 1.1)
        new_pop.append(child)
    return new_pop

# ───── DASHBOARD ───────────────────────────────────────────────────────
def make_dashboard(agent, hyper_pop):
    table = Table(title="Monolithic RL+GA Grid Trader")
    table.add_column("Coin")
    table.add_column("Bandit Value", justify="right")
    table.add_column("GridWidth | TP% | Offset", justify="center")
    for arm in agent.arms:
        params = hyper_pop[0]
        table.add_row(
            arm,
            f"{agent.values[arm]:.6f}",
            f"{params['grid_width']:.4f} | {params['tp_pct']:.4f} | {params['offset']:.4f}"
        )
    return table

# ───── ORDER PLACEMENT ─────────────────────────────────────────────────
def place_hedged_grid(coin, bid, ask, cfg):
    # cancel old
    for oid in open_orders.get(coin, []):
        try: exchange.cancel_order(oid, coin)
        except: pass
    open_orders[coin] = []
    size = NOTIONAL_PER / ((bid+ask)/2)
    buy_price  = bid * (1 - cfg["offset"])
    sell_price = ask * (1 + cfg["offset"])
    buy  = exchange.create_limit_buy_order(coin, size, buy_price, params={'postOnly':True})
    sell = exchange.create_limit_sell_order(coin, size, sell_price, params={'postOnly':True})
    open_orders[coin] = [buy['id'], sell['id']]

# ───── PROCESS ORDER-BOOK & LIVE EXECUTION ────────────────────────────
async def process_order_books(agent, hyper_pop):
    async with websockets.connect(WS_URI) as ws:
        # subscribe to arms
        for coin in agent.arms:
            await ws.send(json.dumps({
                "time": int(time.time()),
                "channel": "spot.order_book",
                "event": "subscribe",
                "payload": [coin.replace("/", "_"), "0.1", "20"]
            }))
            await asyncio.sleep(0.05)
        while True:
            msg = await ws.recv()
            data = json.loads(msg)
            if data.get("event") == "update":
                res = data["result"]
                coin = res["currency_pair"]
                bids, asks = res.get("bids",[]), res.get("asks",[])
                if bids and asks:
                    bid = float(bids[0][0]); ask = float(asks[0][0])
                    spread = (ask - bid) / ask
                    agent.update(coin, spread)
                    # place hedged grid
                    cfg = hyper_pop[0]
                    place_hedged_grid(coin, bid, ask, cfg)

# ───── MAIN ────────────────────────────────────────────────────────────
async def main():
    markets = exchange.load_markets()
    arms = [s for s,m in markets.items() if m['active'] and s.endswith("/USDT")][:20]
    agent = BanditAgent(arms)
    hyper_pop = init_population()

    async def evolve_task():
        while True:
            # gather latest spreads as avg bandit values
            spreads = {arm: agent.values[arm] for arm in arms}
            new_pop = evolve(hyper_pop, spreads)
            hyper_pop.clear()
            hyper_pop.extend(new_pop)
            await asyncio.sleep(300)

    with Live(make_dashboard(agent, hyper_pop), refresh_per_second=2) as live:
        await asyncio.gather(
            process_order_books(agent, hyper_pop),
            evolve_task()
        )

if __name__ == "__main__":
    console.print("[bold green]Starting Monolithic RL+GA Grid Trader[/bold green]")
    asyncio.run(main())


