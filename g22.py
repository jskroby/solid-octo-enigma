import time, threading, random, os, requests
import ccxt
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from collections import defaultdict

console = Console()
thread_lock = threading.Lock()

# ==================== GLOBAL STATE ====================
active_grids = defaultdict(dict)
total_profit = 0.0
symbol_metrics = defaultdict(lambda: {'uptime':0, 'profit':0, 'cps':0})
circuit_breaker = {'fails':0, 'last_trip':0}
ip_pool = ['192.168.1.%d'%i for i in range(100,200)]
current_ip_idx = 0
start_time = time.time()

# ==================== CORE PARAMS ====================
TARGET_ROI = 0.20  # 20% daily
GRID_LAYERS = 3     # Micro, Medium, Macro
NUKE_SPREAD = 0.0002
STEALTH_COOLDOWN = 21600  # 6 hours

def discord_alert(msg):
    """Send message to war room"""
    try:
        requests.post(os.getenv('DISCORD_WEBHOOK'), 
                     json={'content': f"`HYDRA:` {msg}"})
    except Exception as e:
        console.print(f"[red]Discord failed: {e}")

def rotate_ip():
    """Anti-shadowban IP rotation"""
    global current_ip_idx
    current_ip_idx = (current_ip_idx + 1) % len(ip_pool)
    os.environ['HTTP_PROXY'] = f"http://{ip_pool[current_ip_idx]}:8080"
    console.print(f"[yellow]Rotated to IP: {ip_pool[current_ip_idx]}")

def fractal_spacing_adjuster(symbol):
    """Volatility pulse decay algorithm"""
    base = symbol_metrics[symbol].get('base_spacing', 0.001)
    volatility = symbol_data[symbol]['volatility']
    time_decay = 0.99 ** (symbol_metrics[symbol]['uptime']/3600)
    new_spacing = base * (volatility/0.005) * time_decay
    return np.clip(new_spacing, 0.0003, 0.015)

def place_multi_layer_orders(ex, symbol, center_price, pos_size):
    """Triple grid density deployment"""
    orders = []
    for layer in range(GRID_LAYERS):
        spacing = fractal_spacing_adjuster(symbol) * (0.5 ** layer)
        levels = calculate_grid_levels(center_price, 
                                      NUM_GRID_CIRCLES, 
                                      spacing)
        layer_size = pos_size / GRID_LAYERS
        orders += place_grid_orders(ex, symbol, levels, 
                                   layer_size, LEVERAGE, 
                                   center_price)
    return orders

def atomic_yield_compression(ex, symbol, last_price):
    """Double-fill burst zones"""
    for _ in range(2):  # Immediate flip orders
        buy_price = last_price * 0.997
        sell_price = last_price * 1.003
        for price in [buy_price, sell_price]:
            side = 'buy' if price < last_price else 'sell'
            try:
                o = ex.create_limit_order(symbol, side, 
                                         pos_size/last_price, 
                                         price)
                with thread_lock:
                    active_grids[symbol]['orders'].append({
                        'id': o['id'],
                        'side': side,
                        'price': price,
                        'amount': pos_size/last_price,
                        'status': 'open'
                    })
            except Exception as e:
                console.print(f"[red]Atomic fail: {e}")

def monitor_grid(ex, symbol, grid_orders, pos_size):
    global total_profit, symbol_metrics
    
    with thread_lock:
        active_grids[symbol] = {
            'orders': grid_orders,
            'profit': 0.0,
            'start': time.time()
        }
    
    last_rotate = time.time()
    
    while symbol in active_grids:
        try:
            # Rotate IP every 5-15 mins randomly
            if random.random() < 0.01:
                rotate_ip()
            
            # Update metrics
            symbol_metrics[symbol]['uptime'] = time.time() - active_grids[symbol]['start']
            symbol_metrics[symbol]['cps'] = active_grids[symbol]['profit'] / max(1, symbol_metrics[symbol]['uptime'])
            
            # ===== PROFIT LOCKDOWN =====
            if active_grids[symbol]['profit'] >= pos_size * TARGET_ROI:
                discord_alert(f"🚀 {symbol} LOCKDOWN @ +{active_grids[symbol]['profit']:.2f}")
                for o in grid_orders:
                    ex.cancel_order(o['id'], symbol)
                with thread_lock:
                    active_grids.pop(symbol)
                console.print(f"[blue]💤 Cooling {symbol} for {STEALTH_COOLDOWN/3600}h")
                time.sleep(STEALTH_COOLDOWN)
                new_sym = get_best_symbols(ex, 1)[0]
                setup_grid_for_symbol(ex, new_sym, pos_size)
                return
            
            # ===== AUTO-NUKER =====
            if symbol_data[symbol]['spread'] < NUKE_SPREAD:
                discord_alert(f"💣 {symbol} NUKED (spread collapse)")
                for o in grid_orders:
                    ex.cancel_order(o['id'], symbol)
                with thread_lock:
                    active_grids.pop(symbol)
                return
            
            # ===== FRACTAL TIME-SCALING =====
            current_spacing = fractal_spacing_adjuster(symbol)
            if time.time() - last_rotate > 60:
                last_price = ex.fetch_ticker(symbol)['last']
                new_levels = calculate_grid_levels(last_price,
                                                  NUM_GRID_CIRCLES,
                                                  current_spacing)
                # Rebuild grid with new spacing
                for o in grid_orders:
                    ex.cancel_order(o['id'], symbol)
                new_orders = place_multi_layer_orders(ex, symbol,
                                                     last_price,
                                                     pos_size)
                with thread_lock:
                    active_grids[symbol]['orders'] = new_orders
                last_rotate = time.time()
            
            # ===== VOLATILITY BURSTS =====
            if symbol_data[symbol]['volatility'] > 0.015:
                discord_alert(f"⚡ {symbol} VOLATILITY BURST")
                last_price = ex.fetch_ticker(symbol)['last']
                atomic_yield_compression(ex, symbol, last_price)
                # Tighten grids temporarily
                symbol_metrics[symbol]['base_spacing'] *= 0.7
            
            # ===== MICRO-RECOVERY =====
            for order in list(grid_orders):
                if order['status'] == 'open' and order['id'] not in open_orders:
                    # Handle partial fills
                    filled_amt = order['filled'] = ex.get_order(order['id'])['filled']
                    if filled_amt > 0:
                        # Immediate replacement order
                        new_o = ex.create_limit_order(
                            symbol, order['side'], 
                            order['amount'] - filled_amt,
                            order['price']
                        )
                        grid_orders.append({
                            'id': new_o['id'],
                            **new_o
                        })
            
            time.sleep(5 + random.random()*3)  # Jittered sleep
            
        except Exception as e:
            console.print(f"[red]Grid failure {symbol}: {e}")
            circuit_breaker['fails'] +=1
            if circuit_breaker['fails'] > 3:
                discord_alert("🛑 CIRCUIT BREAKER TRIPPED!")
                os._exit(1)

# ==================== ENHANCED SETUP ====================            
def setup_grid_for_symbol(ex, symbol, pos_size):
    try:
        rotate_ip()
        price = ex.fetch_ticker(symbol)['last']
        symbol_data[symbol] = get_market_metrics(ex, symbol)
        adjust_leverage(ex, symbol)
        
        # Initialize fractal spacing
        symbol_metrics[symbol]['base_spacing'] = calculate_dynamic_spacing(symbol)
        
        orders = place_multi_layer_orders(ex, symbol, price, pos_size)
        t = threading.Thread(target=monitor_grid, 
                            args=(ex, symbol, orders, pos_size),
                            daemon=True)
        t.start()
        discord_alert(f"🎯 {symbol} GRID DEPLOYED")
    except Exception as e:
        discord_alert(f"❌ {symbol} SETUP FAILED: {str(e)}")
        console.print(f"[red]Setup error: {e}")

# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    ex = create_exchange()
    symbols = get_best_symbols(ex, 3)
    pos_size = ex.fetch_balance()['USDT']['free'] * 0.2  # 20% per grid
    
    for sym in symbols:
        setup_grid_for_symbol(ex, sym, pos_size)
    
    # Start dashboard thread
    threading.Thread(target=display_dashboard, daemon=True).start()
    
    # Infinite execution
    while True:
        time.sleep(3600)
        # Auto-scale symbols if margin allows
        if ex.fetch_balance()['USDT']['free'] > pos_size * 1.5:
            new_sym = get_best_symbols(ex, 1)[0]
            setup_grid_for_symbol(ex, new_sym, pos_size)
