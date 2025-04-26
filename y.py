# hydra_autotrade_engine.py
import ccxt, os, time, random, threading, numpy as np

API_KEY = os.getenv('GATEIO_API_KEY')
API_SECRET = os.getenv('GATEIO_API_SECRET')

MAX_ACTIVE_GRIDS = 32
LEVERAGE = 20
GRID_SPACING = 0.15 / 100  # 0.15% spacing
TARGET_GRID_PROFIT = 0.0003  # 0.3% profit target
IMMORTALITY_THRESHOLD = 0.80  # Min score to auto-trade
TRADING_PAIR_LIST = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "APT/USDT:USDT"]

active_symbols = {}
thread_lock = threading.Lock()

def create_exchange():
    exchange = ccxt.gateio({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {'defaultType': 'swap', 'defaultSettle': 'usdt'}
    })
    exchange.urls['api']['swap'] = 'https://api.gateio.ws/api/v4'
    return exchange

def calculate_grid_levels(center_price, num_levels=9):
    levels = [center_price * (1 + (i - num_levels//2) * GRID_SPACING) for i in range(num_levels)]
    return levels

def place_grid_orders(ex, symbol, position_size_usd):
    try:
        ex.set_leverage(LEVERAGE, symbol)
        last_price = ex.fetch_ticker(symbol)['last']
        grid_levels = calculate_grid_levels(last_price)
        size_per_level = position_size_usd / len(grid_levels)

        orders = []
        for idx, price in enumerate(grid_levels):
            side = 'buy' if idx % 2 == 0 else 'sell'
            amount = (size_per_level / price) * LEVERAGE
            o = ex.create_limit_order(symbol, side, amount, price, {
                'timeInForce': 'GTC',
                'reduceOnly': False,
                'marginMode': 'cross'
            })
            orders.append(o)
            print(f"[💥] {side.upper()} Order {amount:.5f} @ {price:.5f}")
        return orders
    except Exception as e:
        print(f"[RED] Grid Error: {e}")
        return []

def monitor_grid(ex, symbol, orders, position_size_usd):
    global active_symbols
    profit = 0
    while True:
        try:
            for order in list(orders):
                o = ex.fetch_order(order['id'], symbol)
                if o['status'] == 'closed':
                    pnl = (o['price'] - order['price']) * o['filled']
                    if order['side'] == 'sell':
                        pnl = -pnl
                    profit += pnl
                    orders.remove(order)

            if profit >= position_size_usd * TARGET_GRID_PROFIT:
                print(f"[🤑] LOCKDOWN: {symbol} +${profit:.2f}")
                for o in orders:
                    ex.cancel_order(o['id'], symbol)
                with thread_lock:
                    active_symbols.pop(symbol, None)
                return

            time.sleep(5)
        except Exception as e:
            print(f"[RED] Monitor Fail: {e}")
            time.sleep(5)

def find_best_immortal_symbol(ex):
    best_score = 0
    best_symbol = None
    for sym in TRADING_PAIR_LIST:
        try:
            ticker = ex.fetch_ticker(sym)
            spread = (ticker['ask'] - ticker['bid']) / ticker['last']
            vol = ticker.get('quoteVolume', 0)
            if vol < 1_000_000:
                continue
            score = (1 - spread * 500) + (vol/10_000_000)
            if score > best_score:
                best_score = score
                best_symbol = sym
        except Exception:
            continue
    return best_symbol, best_score

def hydra_main_loop():
    ex = create_exchange()

    while True:
        free_balance = ex.fetch_balance({'type': 'swap'})['USDT']['free']
        pos_size = (free_balance / MAX_ACTIVE_GRIDS) * 0.9

        needed = MAX_ACTIVE_GRIDS - len(active_symbols)
        if needed > 0:
            for _ in range(needed):
                sym, score = find_best_immortal_symbol(ex)
                if sym and score >= IMMORTALITY_THRESHOLD and sym not in active_symbols:
                    print(f"[🔍] Immortal Coin Found: {sym} | Score: {score:.2f}")
                    orders = place_grid_orders(ex, sym, pos_size)
                    threading.Thread(target=monitor_grid, args=(ex, sym, orders, pos_size), daemon=True).start()
                    with thread_lock:
                        active_symbols[sym] = True

        time.sleep(30)

if __name__ == "__main__":
    print("🚀 SK++ Overlord Auto-Trade Engine: Activated")
    hydra_main_loop()
