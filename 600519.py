#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import threading
import pandas as pd
import numpy as np

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData

# ------------------------------------------------------------
# Utility functions for technical indicators
# ------------------------------------------------------------

def compute_ema(series, period):
    """
    Compute Exponential Moving Average for a pandas Series.
    """
    return series.ewm(span=period, adjust=False).mean()

def compute_rsi(series, period=14):
    """
    Compute RSI (Relative Strength Index).
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).abs()
    loss = (delta.where(delta < 0, 0)).abs()
    
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_bollinger_bands(series, period=20, num_std=2):
    """
    Compute Bollinger Bands: returns (middle_band, upper_band, lower_band).
    """
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper_band = sma + num_std * std
    lower_band = sma - num_std * std
    return sma, upper_band, lower_band

def compute_macd(series, fast=12, slow=26, signal=9):
    """
    Compute MACD line, Signal line, and MACD Histogram.
    """
    ema_fast = compute_ema(series, fast)
    ema_slow = compute_ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = compute_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def compute_atr(df, period=14):
    """
    Compute ATR (Average True Range) for the bars in DataFrame df.
    df needs columns: ['High','Low','Close'].
    """
    # True range
    df['H-L'] = df['High'] - df['Low']
    df['H-Cp'] = (df['High'] - df['Close'].shift(1)).abs()
    df['L-Cp'] = (df['Low'] - df['Close'].shift(1)).abs()
    tr = df[['H-L', 'H-Cp', 'L-Cp']].max(axis=1)

    atr = tr.rolling(window=period).mean()
    return atr

def compute_parabolic_sar(df, step=0.02, max_step=0.2):
    """
    Simplified Parabolic SAR implementation.
    Returns a Series with SAR values. 
    For robust usage, prefer a dedicated library.
    """
    # This is a rudimentary approach:
    # We'll store SAR in a list we update as we iterate.
    # 1) Start direction: up or down
    sar = [np.nan] * len(df)
    # Some default to seed the first position
    # In practice, you'd do more robust seeding.
    if len(df) < 2:
        return pd.Series(sar, index=df.index, name="PSAR")

    # Start with direction as up if the second close is above the first
    # Otherwise, start down.
    uptrend = df['Close'].iloc[1] > df['Close'].iloc[0]
    ep = df['High'].iloc[0] if uptrend else df['Low'].iloc[0]
    sar[0] = df['Low'].iloc[0] if uptrend else df['High'].iloc[0]
    af = step

    for i in range(1, len(df)):
        prev_sar = sar[i-1]
        prev_close = df['Close'].iloc[i-1]
        current_high = df['High'].iloc[i]
        current_low = df['Low'].iloc[i]

        # Update SAR
        new_sar = prev_sar + af*(ep - prev_sar)

        # Check for reversal
        if uptrend:
            if current_low < new_sar:
                # Reversal to downtrend
                uptrend = False
                new_sar = max(df['High'].iloc[i-1], df['High'].iloc[i])
                ep = df['Low'].iloc[i]
                af = step
            else:
                # Continue uptrend
                if current_high > ep:
                    ep = current_high
                    af = min(af + step, max_step)

                # Prevent SAR above previous or current low
                new_sar = min(new_sar, df['Low'].iloc[i-1], current_low)
        else:
            if current_high > new_sar:
                # Reversal to uptrend
                uptrend = True
                new_sar = min(df['Low'].iloc[i-1], df['Low'].iloc[i])
                ep = df['High'].iloc[i]
                af = step
            else:
                # Continue downtrend
                if current_low < ep:
                    ep = current_low
                    af = min(af + step, max_step)
                # Prevent SAR below previous or current high
                new_sar = max(new_sar, df['High'].iloc[i-1], current_high)

        sar[i] = new_sar

    return pd.Series(sar, index=df.index, name="PSAR")

def compute_vwap(df):
    """
    Compute Volume Weighted Average Price.
    For intraday usage, you'd typically reset calculations at each session open.
    For demonstration, we do a cumulative calculation here.
    """
    # Typical formula: VWAP = cumsum(Price * Volume) / cumsum(Volume)
    q = df['Close'] * df['Volume']
    vwap = q.cumsum() / df['Volume'].cumsum()
    return vwap

# ------------------------------------------------------------
# IB API Classes
# ------------------------------------------------------------

class IBWrapper(EWrapper):
    """
    EWrapper provides the methods that receive data from TWS or IB Gateway.
    We'll override only the ones we need for real-time bars.
    """
    def __init__(self):
        super().__init__()
        self.data = []  # Will store incoming real-time bars

    def realtimeBar(self, reqId, time_, open_, high, low, close, volume, wap, count):
        """
        IB calls this for each real-time bar.
        By default, real-time bars in IB are 5-second bars if requested.
        """
        bar = {
            'Time': pd.to_datetime(time_, unit='s'),
            'Open': open_,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        }
        self.data.append(bar)
        self.on_bar_update()

    def on_bar_update(self):
        """
        Triggered each time we receive a new bar.
        Here we can compute indicators & print to the console.
        """
        df = pd.DataFrame(self.data)
        
        # Only compute indicators if we have enough bars
        if len(df) < 30:
            # Not enough data to calculate certain indicators, or wait
            last_bar = df.iloc[-1]
            print(f"[Waiting for more bars] Last close: {last_bar['Close']:.2f}, Volume: {last_bar['Volume']}")
            return
        
        # Compute the Indicators
        df['EMA9'] = compute_ema(df['Close'], 9)
        df['EMA21'] = compute_ema(df['Close'], 21)
        df['RSI14'] = compute_rsi(df['Close'], period=14)
        
        # Bollinger
        middle, upper, lower = compute_bollinger_bands(df['Close'], period=20, num_std=2)
        df['BB_MID'] = middle
        df['BB_UP'] = upper
        df['BB_LOW'] = lower

        # MACD
        macd_line, signal_line, hist = compute_macd(df['Close'], fast=12, slow=26, signal=9)
        df['MACD'] = macd_line
        df['MACD_SIGNAL'] = signal_line
        df['MACD_HIST'] = hist

        # Volume Weighted Average Price
        df['VWAP'] = compute_vwap(df)

        # ATR
        df['ATR14'] = compute_atr(df, period=14)

        # Parabolic SAR
        df['PSAR'] = compute_parabolic_sar(df)

        # For S/R lines, you'd typically do your own price action logic,
        # but we’ll skip a robust pivot detection example for brevity.
        
        # Focus on last bar’s values
        last_row = df.iloc[-1]
        
        # Clear screen or just print. For advanced terminal UI, use curses or 'rich'.
        print("\n" + "-"*70)
        print(f"TIME:     {last_row['Time']}")
        print(f"OPEN:     {last_row['Open']:.2f}")
        print(f"HIGH:     {last_row['High']:.2f}")
        print(f"LOW:      {last_row['Low']:.2f}")
        print(f"CLOSE:    {last_row['Close']:.2f}")
        print(f"VOLUME:   {last_row['Volume']}")
        
        # Print indicators
        print(f"EMA9:     {last_row['EMA9']:.2f}")
        print(f"EMA21:    {last_row['EMA21']:.2f}")
        print(f"RSI14:    {last_row['RSI14']:.2f}")
        print(f"BB_UP:    {last_row['BB_UP']:.2f}")
        print(f"BB_MID:   {last_row['BB_MID']:.2f}")
        print(f"BB_LOW:   {last_row['BB_LOW']:.2f}")
        print(f"MACD:     {last_row['MACD']:.2f}")
        print(f"MACD_SIG: {last_row['MACD_SIGNAL']:.2f}")
        print(f"HIST:     {last_row['MACD_HIST']:.2f}")
        print(f"VWAP:     {last_row['VWAP']:.2f}")
        print(f"ATR14:    {last_row['ATR14']:.2f if not np.isnan(last_row['ATR14']) else 'N/A'}")
        print(f"PSAR:     {last_row['PSAR']:.2f if not np.isnan(last_row['PSAR']) else 'N/A'}")

        # Simple interpretation examples
        # 1) EMA cross
        if last_row['EMA9'] > last_row['EMA21']:
            print("EMA9 > EMA21 -> Potential Bullish Momentum")
        else:
            print("EMA9 < EMA21 -> Potential Bearish Momentum")

        # 2) RSI
        if last_row['RSI14'] > 70:
            print("RSI > 70 -> Overbought region")
        elif last_row['RSI14'] < 30:
            print("RSI < 30 -> Oversold region")

        # 3) Bollinger break
        if last_row['Close'] > last_row['BB_UP']:
            print("Price above upper Bollinger Band -> Strong upward momentum")
        elif last_row['Close'] < last_row['BB_LOW']:
            print("Price below lower Bollinger Band -> Strong downward momentum")

        # 4) MACD
        if last_row['MACD'] > last_row['MACD_SIGNAL']:
            print("MACD > Signal -> Bullish crossover")
        else:
            print("MACD < Signal -> Bearish crossover")
        
        # 5) Volume analysis – in a real system, compare last volume to average
        # This is just a placeholder check
        avg_vol = df['Volume'].tail(10).mean()
        if last_row['Volume'] > avg_vol:
            print("Volume picking up -> Confirms stronger momentum")
        else:
            print("Volume below average -> Possibly weaker momentum")

        # 6) VWAP
        if last_row['Close'] > last_row['VWAP']:
            print("Price above VWAP -> Intraday bullish bias")
        else:
            print("Price below VWAP -> Intraday bearish bias")

        # 7) ATR -> Typically used for volatility-based stops
        # 8) Parabolic SAR -> Check if the dot is below or above the price
        if not np.isnan(last_row['PSAR']):
            if last_row['PSAR'] < last_row['Close']:
                print("Parabolic SAR below price -> Uptrend signal")
            else:
                print("Parabolic SAR above price -> Downtrend signal")

        # 9) Price Action with Support & Resistance: omitted for brevity
        print("-"*70)


class IBApp(IBWrapper, EClient):
    """
    Combine EWrapper + EClient into a single App class.
    """
    def __init__(self, host="127.0.0.1", port=7496, client_id=6):
        IBWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)
        self.host = host
        self.port = port
        self.client_id = client_id

    def start_app(self):
        self.connect(self.host, self.port, self.client_id)
        # Launch the EClient.run() loop in a separate thread
        thread = threading.Thread(target=self.run)
        thread.start()

        # Wait a few seconds to connect
        time.sleep(2)

        # Define the contract for 600519 (Kweichow Moutai on SSE)
        contract = Contract()
        contract.symbol = "600519"
        contract.secType = "STK"
        contract.exchange = "SEHK"  # or "SGEX" or "SSE"? 
        # For Chinese A-shares on SSE (Shanghai Stock Exchange) in IB, 
        # you may need 'SSE' or 'SEHK' depending on IB's listing.
        # Double-check the correct exchange + currency
        contract.currency = "CNY"

        # Request real-time bars (5-second bars)
        # DataType: Live or Frozen. 
        # useRTH=0 means we want data even outside regular trading hours (if available).
        self.reqRealTimeBars(
            reqId=1,
            contract=contract,
            barSize=5,  # ignored by TWS, which uses 5sec
            whatToShow="TRADES",
            useRTH=0,
            realTimeBarsOptions=[]
        )

    def stop_app(self):
        self.disconnect()


def main():
    app = IBApp(host="127.0.0.1", port=7496, client_id=5)
    app.start_app()

    print("Connected to IB. Listening for real-time bars on 600519 ...")

    try:
        # Keep the script running, or press Ctrl-C to stop
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Interrupted. Closing connection...")
    finally:
        app.stop_app()
        print("App stopped.")


if __name__ == "__main__":
    main()
