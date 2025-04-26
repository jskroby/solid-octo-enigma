import ccxt
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from rich.console import Console
from rich.live import Live
from rich.table import Table
import sys
import random
from deap import base, creator, tools, algorithms
import requests
import json
import warnings
import asyncio
import queue
from scipy import stats

warnings.filterwarnings('ignore')

# === CONFIG ===
API_KEY =" 4efbe203fbac0e4bcd0003a0910c801b"
API_SECRET =" 8b0e9cf47fdd97f2a42bca571a74105c53a5f906cd4ada125938d05115d063a0"
INITIAL_DEPOSIT = 200.0
SAFETY_BUFFER = 30.0
TARGET_CPS = 0.03  # $0.03 per second = 3 cents

# === DYNAMIC PARAMETERS ===
class TradingParams:
    def __init__(self):
        self.MAX_SYMBOLS = 6
        self.BASE_LEVERAGE = 20
        self.BASE_SPACING_PERCENT = 0.22
        self.MIN_PROFIT_TARGET = 0.0045
        self.PROFIT_SCALING_FACTOR = 1.3
        self.MIN_VOLUME_USD = 450000
        self.MAX_SPREAD_PERCENT = 0.0018
        self.REBALANCE_INTERVAL = 780
        self.SYMBOL_ROTATION_INTERVAL = 1620
        self.GRID_REBUILD_INTERVAL = 240
        self.GA_OPTIMIZATION_INTERVAL = 1800
        self.VOL_LOW = 0.0025
        self.VOL_HIGH = 0.055
        self.STOP_LOSS_THRESHOLD = -0.0015
        self.MAX_DRAWDOWN_PERCENT = 3.0
        self.PROFIT_LOCK_THRESHOLD = 20.0
        self.FRACTAL_LEVELS = 3
        self.GRID_PULSATION_FACTOR = 0.05
        self.HEDGE_TRIGGER_PROFIT = 0.01
        self.HEDGE_SIZE_PERCENT = 0.2
        self.ATOMIC_COMPRESSION_THRESHOLD = 0.008
        self.EMERGENCY_COLLAPSE_SPREAD = 0.005

params = TradingParams()
console = Console()

# === STATE ===
start_time = time.time()
total_profit = 0.0
total_trades = 0
profit_history = []
stop_event = threading.Event()
symbol_data = {}
symbol_performance = {}
collapsed_symbols = set()
active_symbols = []
lock = threading.Lock()

def create_exchange():
    return ccxt.gateio({
        'apiKey': API_KEY,
        'secret': API_SECRET,
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
    ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=lookback)
    closes = np.array([c[4] for c in ohlcv])
    returns = np.diff(np.log(closes))
    return np.std(returns) * np.sqrt(288)

def analyze_symbol_trend(ex, symbol):
    # simplified trend: last vs first
    o = ex.fetch_ohlcv(symbol, '15m', limit=20)
    closes = np.array([x[4] for x in o])
    change = (closes[-1]/closes[0]-1)
    if change>0.002: return 'bullish', change
    if change<-0.002: return 'bearish', abs(change)
    return 'neutral', abs(change)

def get_best_symbols(ex):
    markets = ex.load_markets()
    futs = [s for s in markets if s.endswith('/USDT')]
    tickers = ex.fetch_tickers(futs)
    candidates = []
    for sym,t in tickers.items():
        if not t.get('last'): continue
        spread = (t['ask']-t['bid'])/t['last']
        if spread>params.MAX_SPREAD_PERCENT: continue
        vol = fetch_volatility(ex,sym)
        if vol is None or not(params.VOL_LOW<=vol<=params.VOL_HIGH): continue
        trend,stren = analyze_symbol_trend(ex,sym)
        score = (1-spread/params.MAX_SPREAD_PERCENT)*0.5 + stren*0.5
        candidates.append((sym,score))
    candidates.sort(key=lambda x: x[1],reverse=True)
    syms=[c[0] for c in candidates[:params.MAX_SYMBOLS]]
    return syms

# === GRID & ORDER FUNCTIONS ===
def dynamic_spacing(volatility):
    base=params.BASE_SPACING_PERCENT
    return base * (1+volatility/params.VOL_HIGH)

def place_grid(ex, symbol):
    price = ex.fetch_ticker(symbol)['last']
    vol = fetch_volatility(ex,symbol)
    spacing=dynamic_spacing(vol)/100
    levels = [price*(1+(i-(params.FRACTAL_LEVELS))/10*spacing) for i in range(1,params.FRACTAL_LEVELS*2)]
    size = (INITIAL_DEPOSIT - SAFETY_BUFFER)/price/len(levels)
    orders=[]
    for p in levels:
        side='buy' if p<price else 'sell'
        try:
            o=ex.create_limit_order(symbol,side,size,p)
            orders.append(o)
        except:
            pass
    return orders

def monitor_symbol(ex,symbol):
    global total_profit,total_trades
    orders=place_grid(ex,symbol)
    while not stop_event.is_set():
        open_orders=ex.fetch_open_orders(symbol)
        filled=[o for o in orders if o['id'] not in [oo['id'] for oo in open_orders]]
        for f in filled:
            profit=(f['price'] - f['average'])*f['filled'] if f['side']=='sell' else (f['average']-f['price'])*f['filled']
            with lock:
                total_profit+=profit
                total_trades+=1
            send_notification(f"{symbol} filled {f['side']} profit {profit:.4f}")
        time.sleep(1)

# === PROFIT MONITOR ===
def profit_monitor():
    with Live(console=console,refresh_per_second=1) as live:
        while not stop_event.is_set():
            elapsed=time.time()-start_time
            cps = total_profit/elapsed/0.01
            table=Table()
            table.add_column("Total Profit")
            table.add_column("CPS (cents/sec)")
            table.add_row(f"${total_profit:.2f}",f"{cps:.2f}")
            live.update(table)
            time.sleep(1)

# === INTERACTIVE DEEP SEEK ===
def deep_seek():
    while not stop_event.is_set():
        time.sleep(60*30)
        console.print("\n🐲 TRUE HYDRA SEEK: Market volatility high. Widen spacing? (yes/no)")
        resp=sys.stdin.readline().strip().lower()
        if resp.startswith('y'):
            params.BASE_SPACING_PERCENT*=1.2
            console.print("Spacing widened by 20%.")

# === NOTIFICATION ===
def send_notification(msg): console.print(f"[green]{msg}[/green]")

# === MAIN ===
def main():
    ex=create_exchange()
    syms=get_best_symbols(ex)
    threads=[]
    # start symbol threads
    for s in syms:
        t=threading.Thread(target=monitor_symbol,args=(ex,s),daemon=True)
        threads.append(t); t.start()
    # start profit monitor
    t=threading.Thread(target=profit_monitor,daemon=True); threads.append(t); t.start()
    # start deep seek
    t=threading.Thread(target=deep_seek,daemon=True); threads.append(t); t.start()
    # keep alive
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        stop_event.set()
        console.print("Shutting down...")

if __name__ == '__main__':
    main()

