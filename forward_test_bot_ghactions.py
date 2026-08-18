"""
Forward-Test Bot -- GitHub Actions version (single-run, not a loop)
=====================================================================
Designed to be triggered every 15 minutes by a GitHub Actions cron schedule.
Each run:
  1. Loads saved state (open positions, pending setups, balance) from state.json
  2. Fetches latest candles (public data, no API key needed)
  3. Walks forward through EVERY newly closed candle since the last run
     (not just the most recent one) so no signal is skipped if a run is
     delayed or missed.
  4. Manages open positions, checks pending setups, looks for new signals
  5. Appends any events to forward_test_log.csv
  6. Saves updated state back to state.json
  7. Exits (the GitHub Actions workflow commits the updated files back to the repo)

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
STARTING_BALANCE = 100.0
MAX_WAIT_BARS = 20

STATE_FILE = "state.json"
LOG_FILE = "forward_test_log.csv"

exchange = ccxt.cryptocom({'enableRateLimit': True})  # public data only, no keys needed


# ================= State load/save =================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return dict(balance=STARTING_BALANCE, open_positions=[], pending_setups=[], last_candle_time=None)

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp','event','direction','price','stop','target','r_result','balance'])

def log_row(event, direction='', price='', stop='', target='', r_result='', balance=''):
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([datetime.now(timezone.utc).isoformat(), event, direction, price, stop, target, r_result, balance])


# ================= Data + indicators =================
def fetch_candles(limit=200):
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


# ================= Strategy logic =================
# NOTE: these now take an explicit candle index `i` instead of always
# assuming "the latest closed candle". This lets main() walk forward
# through every candle that closed since the last run, in order,
# instead of only ever looking at the single newest one.
def detect_new_bos_setup(df, i):
    if i < 60:
        return None
    H, L, C = df['high'].values, df['low'].values, df['close'].values
    psh = prev_swing_idx(df, i, 'high')
    if psh is not None and C[i] > H[psh]:
        fvg = find_last_fvg(df, max(psh, i-10), i+1, 'bullish')
        if fvg:
            _, gtop, gbottom = fvg
            return dict(direction='long', gtop=gtop, gbottom=gbottom, bos_time=df['dt'].iloc[i].isoformat())
    psl = prev_swing_idx(df, i, 'low')
    if psl is not None and C[i] < L[psl]:
        fvg = find_last_fvg(df, max(psl, i-10), i+1, 'bearish')
        if fvg:
            _, gtop, gbottom = fvg
            return dict(direction='short', gtop=gtop, gbottom=gbottom, bos_time=df['dt'].iloc[i].isoformat())
    return None

def check_pending_retrace(df, setup, i):
    row = df.iloc[i]
    H, L = df['high'].values, df['low'].values
    bos_time = pd.Timestamp(setup['bos_time'])
    elapsed_minutes = (df['dt'].iloc[i] - bos_time).total_seconds() / 60
    bars_waited = round(elapsed_minutes / 15)
    if bars_waited > MAX_WAIT_BARS:
        return 'expired'
    if bars_waited <= 0:
        return None
    if setup['direction'] == 'long':
        if L[i] <= setup['gtop']:
            if ichimoku_bullish(row):
                entry = (setup['gtop']+setup['gbottom'])/2
                stop = setup['gbottom']*0.9995
                risk = entry-stop
                if risk > 0:
                    return dict(direction='long', entry=entry, stop=stop, target=entry+risk*RR)
            return 'expired'
    else:
        if H[i] >= setup['gbottom']:
            if ichimoku_bearish(row):
                entry = (setup['gtop']+setup['gbottom'])/2
                stop = setup['gtop']*1.0005
                risk = stop-entry
                if risk > 0:
                    return dict(direction='short', entry=entry, stop=stop, target=entry-risk*RR)
            return 'expired'
    return None


# ================= Position management =================
def manage_open_positions(state, high, low):
    still_open = []
    for d in state['open_positions']:
        hit_stop = (low <= d['stop']) if d['direction']=='long' else (high >= d['stop'])
        hit_target = (high >= d['target']) if d['direction']=='long' else (low <= d['target'])
        if hit_stop:
            r = -1.0
            state['balance'] *= (1 + RISK_PCT*r)
            log_row('EXIT_STOP', d['direction'], d['entry'], d['stop'], d['target'], r, round(state['balance'],2))
            print(f"STOP HIT ({d['direction']}). r={r}. Balance=${state['balance']:.2f}")
        elif hit_target:
            r = RR
            state['balance'] *= (1 + RISK_PCT*r)
            log_row('EXIT_TARGET', d['direction'], d['entry'], d['stop'], d['target'], r, round(state['balance'],2))
            print(f"TARGET HIT ({d['direction']}). r={r}. Balance=${state['balance']:.2f}")
        else:
            still_open.append(d)
    state['open_positions'] = still_open

def open_new_position(state, signal):
    state['open_positions'].append(signal)
    log_row('ENTRY', signal['direction'], signal['entry'], signal['stop'], signal['target'], '', round(state['balance'],2))
    print(f"NEW {signal['direction'].upper()} entry={signal['entry']:.1f} stop={signal['stop']:.1f} "
          f"target={signal['target']:.1f}  ({len(state['open_positions'])} position(s) now open)")


# ================= Helpers for walking through missed candles =================
def find_index_of_time(df, iso_time):
    """Return the row index in df whose 'dt' matches iso_time, or None if not found
    (e.g. it's older than the 200-candle window we fetched)."""
    if iso_time is None:
        return None
    target = pd.Timestamp(iso_time)
    matches = df.index[df['dt'] == target]
    if len(matches) == 0:
        return None
    return matches[0]


# ================= Main (single run) =================
def main():
    init_log()
    state = load_state()
    print(f"=== Run at {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Loaded state: balance=${state['balance']:.2f}, "
          f"{len(state['open_positions'])} open position(s), {len(state['pending_setups'])} pending setup(s)")

    df = fetch_candles(limit=200)
    df = compute_indicators(df)

    latest_closed_idx = len(df) - 2  # last fully-closed candle in this fetch

    # Always check open positions against the *currently forming* candle's
    # high/low, so stop/target hits are caught in real time regardless of
    # whether a new candle has closed yet.
    last_row = df.iloc[-1]
    manage_open_positions(state, last_row['high'], last_row['low'])

    last_idx = find_index_of_time(df, state.get('last_candle_time'))

    if last_idx is None:
        # First run ever, or the last processed candle fell outside our
        # 200-candle window (a very long gap) -- fall back to just the
        # newest closed candle so we don't reprocess ancient history.
        start_idx = latest_closed_idx
    else:
        start_idx = last_idx + 1

    if start_idx > latest_closed_idx:
        print("No new closed candle since last run -- nothing to do.")
    else:
        skipped = latest_closed_idx - start_idx
        if skipped > 0:
            print(f"Catching up on {skipped + 1} closed candle(s) missed since last run.")

        # Walk forward through every newly closed candle IN ORDER, so no
        # BOS/FVG setup or retrace trigger in between is ever skipped.
        for i in range(start_idx, latest_closed_idx + 1):
            row = df.iloc[i]

            # Also check open positions against this historical candle's
            # high/low (more accurate than only checking the live candle
            # when catching up on several missed candles at once).
            manage_open_positions(state, row['high'], row['low'])

            still_pending = []
            for setup in state['pending_setups']:
                result = check_pending_retrace(df, setup, i)
                if result == 'expired':
                    print(f"Pending {setup['direction']} setup expired.")
                elif isinstance(result, dict):
                    open_new_position(state, result)
                else:
                    still_pending.append(setup)
            state['pending_setups'] = still_pending

            new_setup = detect_new_bos_setup(df, i)
            if new_setup:
                state['pending_setups'].append(new_setup)
                print(f"New {new_setup['direction']} BOS+FVG detected "
                      f"({len(state['pending_setups'])} setup(s) now pending).")

        state['last_candle_time'] = df['dt'].iloc[latest_closed_idx].isoformat()

    save_state(state)
    print(f"Run complete. Balance=${state['balance']:.2f}, "
          f"{len(state['open_positions'])} open, {len(state['pending_setups'])} pending.")

if __name__ == "__main__":
    main()
