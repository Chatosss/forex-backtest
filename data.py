"""Carregamento de dados OHLC de forex, com cache local em CSV.

Fonte primaria: MetaTrader 5 (dados da corretora, spread real por barra).
Fonte alternativa: yfinance (apenas diario, sem spread).

O MT5 e usado SOMENTE para leitura de cotacoes. Nenhuma ordem e enviada.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# Tamanho do pip por familia de simbolo.
PIP_SIZE = {
    "JPY": 0.01,     # pares com iene cotam com 3 casas
    "XAU": 0.10,     # ouro: 1 pip = 0.10 USD
    "default": 0.0001,
}


def pip_size(symbol: str) -> float:
    if symbol.endswith("JPY"):
        return PIP_SIZE["JPY"]
    if symbol.startswith("XAU"):
        return PIP_SIZE["XAU"]
    return PIP_SIZE["default"]


def _mt5_timeframe(tf: str):
    import MetaTrader5 as mt5
    return {
        "M15": mt5.TIMEFRAME_M15,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }[tf]


def load_mt5(symbol: str, timeframe: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Baixa barras do terminal MT5. Retorna DataFrame indexado por tempo."""
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 nao inicializou: {mt5.last_error()}")
    try:
        # garante que o simbolo esta selecionado no Market Watch
        if not mt5.symbol_select(symbol, True):
            raise RuntimeError(f"simbolo indisponivel: {symbol}")

        t0 = dt.datetime.fromisoformat(start)
        t1 = dt.datetime.fromisoformat(end) if end else dt.datetime.now() + dt.timedelta(days=1)
        rates = mt5.copy_rates_range(symbol, _mt5_timeframe(timeframe), t0, t1)
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"sem barras para {symbol} {timeframe}: {mt5.last_error()}")

        info = mt5.symbol_info(symbol)
        point = info.point if info else pip_size(symbol) / 10
    finally:
        mt5.shutdown()

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.set_index("time")
    df = df.rename(columns={"tick_volume": "volume"})
    # spread vem em points; converte para preco
    df["spread_price"] = df["spread"].astype(float) * point
    return df[["open", "high", "low", "close", "volume", "spread_price"]]


def load_yahoo(symbol: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Fallback diario via Yahoo Finance. Sem dado de spread."""
    import yfinance as yf

    ticker = f"{symbol}=X"
    df = yf.download(ticker, start=start, end=end, interval="1d",
                     progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f"Yahoo nao retornou dados para {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index.name = "time"
    df["spread_price"] = pip_size(symbol) * 1.5  # estimativa
    return df


def get(symbol: str, timeframe: str = "H1", start: str = "2021-01-01",
        end: str | None = None, source: str = "mt5", refresh: bool = False) -> pd.DataFrame:
    """Carrega barras usando cache em disco."""
    cache = DATA_DIR / f"{symbol}_{timeframe}_{source}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
    else:
        df = load_mt5(symbol, timeframe, start, end) if source == "mt5" \
            else load_yahoo(symbol, start, end)
        df.to_csv(cache)

    df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]
    # sanidade: remove barras degeneradas
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0)]
    return df.dropna(subset=["open", "high", "low", "close"])


if __name__ == "__main__":
    for sym in ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "XAUUSD"]:
        for tf in ["H1", "D1"]:
            try:
                d = get(sym, tf, refresh=True)
                sp = d["spread_price"] / pip_size(sym)
                print(f"{sym:7} {tf:3} {len(d):>6} barras  "
                      f"{d.index[0]:%Y-%m-%d} -> {d.index[-1]:%Y-%m-%d}  "
                      f"spread medio {sp.mean():.2f} pips")
            except Exception as e:
                print(f"{sym:7} {tf:3} ERRO: {e}")
