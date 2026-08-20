"""
Forward-Test Bot -- GitHub Actions version (v3: adds Filter C + fully retroactive)
====================================================================================
Strategy: BOS (market structure break) + FVG (fair value gap) retracement entry,
filtered by (a) Ichimoku trend confluence and (b) FVG size > 0.3x ATR(14)
("Filter C" -- only trade "significant" imbalances, not noise-sized gaps).
Backtested on 7 months of 15m BTCUSDT: n=916, win_rate=61.0%, PF=3.13.

v3 change: Filter C was missing from earlier bot versions (v1/v2 only had
FVG+Ichimoku, matching the 52.5%-win-rate baseline, NOT the 61%-win-rate
final strategy). This version adds it back in.

Also fixes a class of bugs from v1: previously, position management,
pending-setup checks, and new-signal detection each only looked at the
SINGLE most recently closed candle. If GitHub Actions ran late (which it
does, sometimes by hours), everything in between was silently skipped.
This version processes every new closed candle since the last run, ONE AT
A TIME, IN CHRONOLOGICAL ORDER -- exactly mirroring the validated backtest.

Requires: pip install ccxt pandas numpy
"""

import ccxt
import pandas as pd
import numpy as np
import json
import csv
import os
from datetime import datetime, timezone

# ================= CONFIG =================
SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
RISK_PCT = 0.01
RR = 2.0
MAX_WAIT_BARS = 20
FVG_ATR_MULT = 0.3   # Filter C: FVG size must exceed this multiple of ATR(14)

STATE_FILE = "state.json"
LOG_FILE = "forward_test_log.csv"

exchange = ccxt.cryptocom({'enableRateLimit': True})


# ================= State load/save =================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return dict(balance=100.0, open_positions=[], pending_setups=[], last_processed_time=None)

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp','event','direction','price','stop','target','r_result','balance'])

_existing_log_keys = None  # cache of (timestamp, event, direction, price) tuples already in the log

def _load_existing_log_keys():
    global _existing_log_keys
    keys = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for row in reader:
                if len(row) >= 4:
                    keys.add((row[0], row[1], row[2], row[3]))
    _existing_log_keys = keys

def log_row(candle_time, event, direction='', price='', stop='', target='', r_result='', balance=''):
    global _existing_log_keys
    if _existing_log_keys is None:
        _load_existing_log_keys()
    key = (str(candle_time), event, direction, str(price))
    if key in _existing_log_keys:
        # Idempotency guard: this exact event was already logged (e.g. from a
        # prior run that raced with this one). Skip writing a duplicate.
        print(f"  (skipped duplicate log row: {key})")
        return
    _existing_log_keys.add(key)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        # Use the CANDLE's timestamp, not "now" -- so catch-up runs record events
        # at the time they actually happened in the market, not when the bot noticed.
        w.writerow([candle_time, event, direction, price, stop, target, r_result, balance])


# ================= Data + indicators (computed once per run, over the whole fetched window) =================
def fetch_candles(limit=500):
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['ts','open','high','low','close','volume'])
    df['dt'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
    return df

def compute_indicators(df):
    high, low, close = df['high'], df['low'], df['close']
    def donchian_mid(s_high, s_low, period):
        return (s_high.rolling(period).max() + s_low.rolling(period).min())/2
    tenkan = donchian_mid(high, low, 9)
    kijun  = donchian_mid(high, low, 26)
    df['tenkan'] = tenkan
    df['kijun'] = kijun
    df['senkou_a'] = ((tenkan+kijun)/2).shift(26)
    df['senkou_b'] = donchian_mid(high, low, 52).shift(26)
    K = 2
    df['swing_high'] = df['high'] == df['high'].rolling(2*K+1, center=True).max()
    df['swing_low']  = df['low']  == df['low'].rolling(2*K+1, center=True).min()
    # ATR(14) -- used by Filter C (FVG-size filter)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    df['atr'] = tr.ewm(alpha=1/14, adjust=False).mean()
    return df

def ichimoku_bullish(row):
    if pd.isna(row['senkou_a']) or pd.isna(row['senkou_b']):
        return False
    return row['close'] > max(row['senkou_a'], row['senkou_b']) and row['tenkan'] > row['kijun']

def ichimoku_bearish(row):
    if pd.isna(row['senkou_a']) or pd.isna(row['senkou_b']):
        return False
    return row['close'] < min(row['senkou_a'], row['senkou_b']) and row['tenkan'] < row['kijun']

def find_last_fvg(df, start, end, direction):
    H, L = df['high'].values, df['low'].values
    for i in range(end-1, max(1,start)-1, -1):
        if direction=='bullish' and H[i-2] < L[i]:
            return (i, L[i], H[i-2])
        if direction=='bearish' and L[i-2] > H[i]:
            return (i, L[i-2], H[i])
    return None

def prev_swing_idx(df, i, kind):
    col = 'swing_high' if kind=='high' else 'swing_low'
    j = i-1
    while j >= 0:
        if df[col].iloc[j]:
            return j
        j -= 1
    return None


# ================= Per-candle processing (the core fix) =================
def process_candle(df, i, state):
    """Run the FULL strategy logic for a single closed candle at index i:
    1) manage open positions against this candle's high/low
    2) check pending setups for retrace-triggered entries
    3) detect a fresh BOS+FVG setup forming at this candle
    All three happen for every candle in order -- none can be skipped."""
    H, L, C = df['high'].values, df['low'].values, df['close'].values
    row = df.iloc[i]
    ct = df['dt'].iloc[i].isoformat()

    # --- 1) position management ---
    still_open = []
    for d in state['open_positions']:
        hit_stop = (L[i] <= d['stop']) if d['direction']=='long' else (H[i] >= d['stop'])
        hit_target = (H[i] >= d['target']) if d['direction']=='long' else (L[i] <= d['target'])
        if hit_stop:
            r = -1.0
            state['balance'] *= (1 + RISK_PCT*r)
            log_row(ct, 'EXIT_STOP', d['direction'], d['entry'], d['stop'], d['target'], r, round(state['balance'],2))
            print(f"[{ct}] STOP HIT ({d['direction']}). r={r}. Balance=${state['balance']:.2f}")
        elif hit_target:
            r = RR
            state['balance'] *= (1 + RISK_PCT*r)
            log_row(ct, 'EXIT_TARGET', d['direction'], d['entry'], d['stop'], d['target'], r, round(state['balance'],2))
            print(f"[{ct}] TARGET HIT ({d['direction']}). r={r}. Balance=${state['balance']:.2f}")
        else:
            still_open.append(d)
    state['open_positions'] = still_open

    # --- 2) pending setups: check for retrace + Ichimoku confirmation ---
    still_pending = []
    for setup in state['pending_setups']:
        bos_time = pd.Timestamp(setup['bos_time'])
        bars_waited = round((df['dt'].iloc[i] - bos_time).total_seconds() / 60 / 15)
        if bars_waited > MAX_WAIT_BARS:
            print(f"[{ct}] Pending {setup['direction']} setup expired.")
            continue
        if bars_waited <= 0:
            still_pending.append(setup)
            continue

        triggered = False
        if setup['direction'] == 'long':
            if L[i] <= setup['gtop']:
                if ichimoku_bullish(row):
                    entry = (setup['gtop']+setup['gbottom'])/2
                    stop = setup['gbottom']*0.9995
                    risk = entry-stop
                    if risk > 0:
                        target = entry+risk*RR
                        pos = dict(direction='long', entry=entry, stop=stop, target=target)
                        state['open_positions'].append(pos)
                        log_row(ct, 'ENTRY', 'long', entry, stop, target, '', round(state['balance'],2))
                        print(f"[{ct}] NEW LONG entry={entry:.1f} stop={stop:.1f} target={target:.1f}")
                triggered = True
        else:
            if H[i] >= setup['gbottom']:
                if ichimoku_bearish(row):
                    entry = (setup['gtop']+setup['gbottom'])/2
                    stop = setup['gtop']*1.0005
                    risk = stop-entry
                    if risk > 0:
                        target = entry-risk*RR
                        pos = dict(direction='short', entry=entry, stop=stop, target=target)
                        state['open_positions'].append(pos)
                        log_row(ct, 'ENTRY', 'short', entry, stop, target, '', round(state['balance'],2))
                        print(f"[{ct}] NEW SHORT entry={entry:.1f} stop={stop:.1f} target={target:.1f}")
                triggered = True

        if not triggered:
            still_pending.append(setup)
    state['pending_setups'] = still_pending

    # --- 3) detect a fresh BOS + FVG at this candle ---
    if i < 60:
        return
    ATR = df['atr'].values

    def already_queued(direction, gtop, gbottom, tol=1e-6):
        """Dedup guard: same gap (within tolerance) already pending in this direction."""
        for s in state['pending_setups']:
            if s['direction'] == direction and abs(s['gtop']-gtop) < tol and abs(s['gbottom']-gbottom) < tol:
                return True
        return False

    psh = prev_swing_idx(df, i, 'high')
    # Only fire on the actual breakout candle (previous close had NOT yet broken
    # the swing) -- otherwise, since psh doesn't move until a new opposite swing
    # forms, C[i] > H[psh] stays true for many subsequent candles and the same
    # FVG gets re-queued over and over, producing duplicate trades.
    fresh_break_up = psh is not None and C[i] > H[psh] and not (i > 0 and C[i-1] > H[psh])
    if fresh_break_up:
        fvg = find_last_fvg(df, max(psh, i-10), i+1, 'bullish')
        if fvg:
            _, gtop, gbottom = fvg
            # Filter C: only take FVGs that represent a "significant" imbalance
            # relative to normal market volatility (ATR), not noise-sized gaps.
            if not np.isnan(ATR[i]) and (gtop - gbottom) > FVG_ATR_MULT * ATR[i]:
                if not already_queued('long', gtop, gbottom):
                    state['pending_setups'].append(dict(direction='long', gtop=gtop, gbottom=gbottom,
                                                         bos_time=df['dt'].iloc[i].isoformat()))
                    print(f"[{ct}] New LONG BOS+FVG detected (passed Filter C).")
                else:
                    print(f"[{ct}] LONG BOS+FVG matches an already-queued setup -- skipped (dedup).")

    psl = prev_swing_idx(df, i, 'low')
    fresh_break_down = psl is not None and C[i] < L[psl] and not (i > 0 and C[i-1] < L[psl])
    if fresh_break_down:
        fvg = find_last_fvg(df, max(psl, i-10), i+1, 'bearish')
        if fvg:
            _, gtop, gbottom = fvg
            if not np.isnan(ATR[i]) and (gtop - gbottom) > FVG_ATR_MULT * ATR[i]:
                if not already_queued('short', gtop, gbottom):
                    state['pending_setups'].append(dict(direction='short', gtop=gtop, gbottom=gbottom,
                                                         bos_time=df['dt'].iloc[i].isoformat()))
                    print(f"[{ct}] New SHORT BOS+FVG detected (passed Filter C).")
                else:
                    print(f"[{ct}] SHORT BOS+FVG matches an already-queued setup -- skipped (dedup).")


# ================= Main (single run, catches up on ALL missed candles) =================
def main():
    init_log()
    state = load_state()
    print(f"=== Run at {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Loaded state: balance=${state['balance']:.2f}, "
          f"{len(state['open_positions'])} open, {len(state['pending_setups'])} pending, "
          f"last_processed_time={state.get('last_processed_time')}")

    df = fetch_candles(limit=500)  # 500 candles = ~5.2 days of 15m history, plenty of buffer
    df = compute_indicators(df)
    N = len(df)

    last_processed_time = state.get('last_processed_time')
    if last_processed_time is None:
        start_idx = 60  # first run: need enough history for swings/Ichimoku, don't replay everything
    else:
        last_ts = pd.Timestamp(last_processed_time)
        mask = df['dt'] > last_ts
        start_idx = mask.idxmax() if mask.any() else N - 1

    end_idx = N - 1  # never process the still-forming last candle

    n_new = max(0, end_idx - start_idx)
    print(f"Processing {n_new} new closed candle(s) (index {start_idx} to {end_idx-1})...")

    for i in range(start_idx, end_idx):
        process_candle(df, i, state)

    if end_idx > start_idx:
        state['last_processed_time'] = df['dt'].iloc[end_idx - 1].isoformat()

    save_state(state)
    print(f"Run complete. Balance=${state['balance']:.2f}, "
          f"{len(state['open_positions'])} open, {len(state['pending_setups'])} pending.")

if __name__ == "__main__":
    main()
