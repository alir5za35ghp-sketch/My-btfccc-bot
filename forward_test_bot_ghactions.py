"""
Forward-Test Bot -- COMPLETE FINAL VERSION (rebuilt after environment reset)
================================================================================
All fixes included:
  1. Retroactive catch-up: processes EVERY closed candle since the last run.
  2. Filter C: FVG size must exceed 0.3x ATR(14).
  3. Proper dedup: only skips exact-same-FVG duplicates, not all post-BOS candles.
  4. ATR-based stop: stop = FVG edge +/- 0.5x ATR(14), not a tiny fixed buffer.
  5. Minimum stop distance filter: rejects setups with stop < 0.2% of price.
  6. Log dedup guard against race conditions.
Requires: pip install ccxt pandas numpy
"""

import ccxt
import pandas as pd
import numpy as np
import json
import csv
import os
from datetime import datetime, timezone

SYMBOL = "BTC/USDT"
TIMEFRAME = "15m"
RISK_PCT = 0.01
RR = 2.0
MAX_WAIT_BARS = 20
FVG_ATR_MULT = 0.3
STOP_ATR_MULT = 0.5
MIN_STOP_PCT = 0.2

STATE_FILE = "state.json"
LOG_FILE = "forward_test_log.csv"

exchange = ccxt.cryptocom({'enableRateLimit': True})


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return dict(balance=100.0, open_positions=[], pending_setups=[], last_processed_time=None, recent_fvg_keys=[])

def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

def init_log():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp','event','direction','price','stop','target','r_result','balance'])

_existing_log_keys = None

def _load_existing_log_keys():
    global _existing_log_keys
    keys = set()
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', newline='') as f:
            reader = csv.reader(f)
            next(reader, None)
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
        print(f"  (skipped duplicate log row: {key})")
        return
    _existing_log_keys.add(key)
    with open(LOG_FILE, 'a', newline='') as f:
        w = csv.writer(f)
        w.writerow([candle_time, event, direction, price, stop, target, r_result, balance])


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


def process_candle(df, i, state):
    H, L, C = df['high'].values, df['low'].values, df['close'].values
    ATR = df['atr'].values
    row = df.iloc[i]
    ct = df['dt'].iloc[i].isoformat()

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
                    stop = setup['gbottom'] - STOP_ATR_MULT * setup['atr_at_bos']
                    risk = entry-stop
                    if risk > 0 and (risk/entry*100) >= MIN_STOP_PCT:
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
                    stop = setup['gtop'] + STOP_ATR_MULT * setup['atr_at_bos']
                    risk = stop-entry
                    if risk > 0 and (risk/entry*100) >= MIN_STOP_PCT:
                        target = entry-risk*RR
                        pos = dict(direction='short', entry=entry, stop=stop, target=target)
                        state['open_positions'].append(pos)
                        log_row(ct, 'ENTRY', 'short', entry, stop, target, '', round(state['balance'],2))
                        print(f"[{ct}] NEW SHORT entry={entry:.1f} stop={stop:.1f} target={target:.1f}")
                triggered = True

        if not triggered:
            still_pending.append(setup)
    state['pending_setups'] = still_pending

    if i < 60:
        return

    def _fvg_already_seen(direction, gtop, gbottom):
        key = (direction, round(gtop, 2), round(gbottom, 2))
        for s in state['pending_setups']:
            if (s['direction'], round(s['gtop'], 2), round(s['gbottom'], 2)) == key:
                return True
        for k in state.get('recent_fvg_keys', []):
            if tuple(k) == key:
                return True
        return False

    def _remember_fvg(direction, gtop, gbottom):
        key = [direction, round(gtop, 2), round(gbottom, 2)]
        recent = state.setdefault('recent_fvg_keys', [])
        recent.append(key)
        if len(recent) > 200:
            del recent[:len(recent)-200]

    psh = prev_swing_idx(df, i, 'high')
    if psh is not None and C[i] > H[psh]:
        fvg = find_last_fvg(df, max(psh, i-10), i+1, 'bullish')
        if fvg:
            _, gtop, gbottom = fvg
            if not np.isnan(ATR[i]) and (gtop - gbottom) > FVG_ATR_MULT * ATR[i]:
                if not _fvg_already_seen('long', gtop, gbottom):
                    state['pending_setups'].append(dict(direction='long', gtop=gtop, gbottom=gbottom,
                                                         atr_at_bos=float(ATR[i]),
                                                         bos_time=df['dt'].iloc[i].isoformat()))
                    _remember_fvg('long', gtop, gbottom)
                    print(f"[{ct}] New LONG BOS+FVG detected (unique, passed Filter C).")
    psl = prev_swing_idx(df, i, 'low')
    if psl is not None and C[i] < L[psl]:
        fvg = find_last_fvg(df, max(psl, i-10), i+1, 'bearish')
        if fvg:
            _, gtop, gbottom = fvg
            if not np.isnan(ATR[i]) and (gtop - gbottom) > FVG_ATR_MULT * ATR[i]:
                if not _fvg_already_seen('short', gtop, gbottom):
                    state['pending_setups'].append(dict(direction='short', gtop=gtop, gbottom=gbottom,
                                                         atr_at_bos=float(ATR[i]),
                                                         bos_time=df['dt'].iloc[i].isoformat()))
                    _remember_fvg('short', gtop, gbottom)
                    print(f"[{ct}] New SHORT BOS+FVG detected (unique, passed Filter C).")


def main():
    init_log()
    state = load_state()
    print(f"=== Run at {datetime.now(timezone.utc).isoformat()} ===")
    print(f"Loaded state: balance=${state['balance']:.2f}, "
          f"{len(state['open_positions'])} open, {len(state['pending_setups'])} pending, "
          f"last_processed_time={state.get('last_processed_time')}")

    df = fetch_candles(limit=500)
    df = compute_indicators(df)
    N = len(df)

    last_processed_time = state.get('last_processed_time')
    if last_processed_time is None:
        start_idx = 60
    else:
        last_ts = pd.Timestamp(last_processed_time)
        mask = df['dt'] > last_ts
        start_idx = mask.idxmax() if mask.any() else N - 1

    end_idx = N - 1
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
