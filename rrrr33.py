#!/usr/bin/env python3
# === ULTRA HYDRA GOD BOT ===
# FINAL CARTMAN PATCH
# v999.0 FINAL FORM

# === IMPORTS ===
import ccxt.async_support as ccxt
import asyncio
import json
import time
import numpy as np
import pandas as pd
import random
import os
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.table import Table
import requests
from deap import base, creator, tools, algorithms
import warnings
warnings.filterwarnings('ignore')

# === CONFIGURATION ===
API_KEY = "4efbe203fbac0e4bcd0003a0910c801b"
API_SECRET = "8b0e9cf47fdd97f2a42bca571a74105c53a5f906cd4ada125938d05115d063a0"
DISCORD_WEBHOOK = ""
TELEGRAM_TOKEN = ""
TELEGRAM_CHAT_ID = ""

INITIAL_DEPOSIT = 200.0
SAFETY_BUFFER = 30.0
TARGET_CPS = 0.03

# === ULTRA OPTIMIZED PARAMETERS ===
BASE_LEVERAGE = 50
BASE_SPACING_PERCENT = 0.18
MIN_PROFIT_TARGET = 0.0042
MAX_SPREAD_PERCENT = 0.0015
MIN_VOLUME_USD = 600_000
VOL_LOW = 0.002
VOL_HIGH = 0.05
FRACTAL_LEVELS = 4
REBALANCE_INTERVAL = 900
SYMBOL_ROTATION_INTERVAL = 3600

# === STATE ===
console = Console()
profit_total = 0.0
start_time = time.time()
running = True
lock = asyncio.Lock()
active_orders = {}

# === EXCHANGE SETUP ===
async def create_exchange():
    ex = ccxt.gateio({
        "apiKey": API_KEY,
        "secret": API_SECRET,
        "enableRateLimit": True,
        "options": {"defaultType": "swap", "defaultSettle": "usdt"}
    })
    await ex.load_markets()
    return ex

# === UTILITIES ===
async def send_alert(msg, level="info"):
    now = datetime.now().strftime("%H:%M:%S")
    color = {"info": 3447003, "warn": 16776960, "error": 16711680}.get(level, 3447003)
    payload = {"embeds": [{"title": f"ULTRA HYDRA BOT [{level.upper()}]", "description": f"{now} - {msg}", "color": color}]}
    if DISCORD_WEBHOOK:
        try: requests.post(DISCORD_WEBHOOK, json=payload, timeout=5)
        except: pass

async def send_telegram(msg):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try: requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg}, timeout=5)
        except: pass

async def notify(msg, level="info"):
    console.print(f"[bold]{msg}[/bold]")
    await send_alert(msg, level)
    await send_telegram(msg)

def calc_cps():
    elapsed = time.time() - start_time
    return profit_total / elapsed if elapsed > 0 else 0

# === MARKET LOGIC ===
async def fetch_symbols(ex):
    markets = await ex.load_markets()
    symbols = [s for s in markets if s.endswith("/USDT:USDT")]
    return random.sample(symbols, min(6, len(symbols)))

async def analyze_volatility(ex, symbol):
    ohlcv = await ex.fetch_ohlcv(symbol, '5m', limit=50)
    closes = np.array([x[4] for x in ohlcv])
    returns = np.diff(np.log(closes))
    vol = np.std(returns) * np.sqrt(288)
    return vol

async def build_grid(price, levels, spacing, fractals):
    grid = []
    for f in range(fractals):
        for i in range(-levels, levels+1):
            grid.append(price * (1 + i * spacing * (1 + f*0.2)))
    grid = list(sorted(set(grid)))
    return grid

async def place_grid(ex, symbol, price, size, spacing, fractals):
    orders = []
    grid = await build_grid(price, 5, spacing, fractals)
    for level in grid:
        side = 'buy' if level < price else 'sell'
        try:
            order = await ex.create_order(symbol, 'limit', side, size, level, {"postOnly": True})
            orders.append(order)
            await asyncio.sleep(0.05)
        except Exception:
            continue
    return orders

async def monitor_grid(ex, symbol, orders):
    global profit_total
    try:
        while running:
            open_orders = await ex.fetch_open_orders(symbol)
            open_ids = {o['id'] for o in open_orders}
            for o in list(orders):
                if o['id'] not in open_ids:
                    try:
                        filled = o.get('filled', o['amount'])
                        if filled > 0:
                            last_price = (await ex.fetch_ticker(symbol))['last']
                            profit = (last_price - o['price']) * filled if o['side'] == 'sell' else (o['price'] - last_price) * filled
                            profit -= profit * 0.0008  # Fee/slip
                            async with lock:
                                profit_total += profit
                            orders.remove(o)
                            await notify(f"[{symbol}] Profit: {profit:.4f} USD", "info")
                    except:
                        pass
            await asyncio.sleep(2)
    except Exception as e:
        await notify(f"Monitor crashed {symbol}: {e}", "error")

# === MAIN LOGIC ===
async def worker(symbol):
    ex = await create_exchange()
    await notify(f"Started trading {symbol}")
    ticker = await ex.fetch_ticker(symbol)
    price = ticker['last']
    spacing = BASE_SPACING_PERCENT / 100
    vol = await analyze_volatility(ex, symbol)
    if vol and vol > VOL_LOW:
        spacing *= min(1.5, vol / VOL_LOW)
    size = (INITIAL_DEPOSIT - SAFETY_BUFFER) / price / 10
    orders = await place_grid(ex, symbol, price, size, spacing, FRACTAL_LEVELS)
    active_orders[symbol] = orders
    await monitor_grid(ex, symbol, orders)

async def main():
    ex = await create_exchange()
    symbols = await fetch_symbols(ex)
    await notify(f"Selected Symbols: {symbols}")
    tasks = [asyncio.create_task(worker(s)) for s in symbols]
    cps_task = asyncio.create_task(monitor_cps())
    await asyncio.gather(*tasks, cps_task)

async def monitor_cps():
    with Live(console=console, refresh_per_second=1) as live:
        while running:
            cps = calc_cps()
            t = Table()
            t.add_column("Total Profit", justify="right")
            t.add_column("Cents/Second", justify="right")
            t.add_row(f"${profit_total:.2f}", f"{cps:.4f}")
            live.update(t)
            await asyncio.sleep(1)

# === RUN ===
try:
    asyncio.run(main())
except KeyboardInterrupt:
    running = False
    console.print("[bold yellow]Shutting down ULTRA HYDRA GOD BOT...[/bold yellow]")
