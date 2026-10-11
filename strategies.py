"""Estrategias e indicadores.

Todo indicador aqui e causal: usa apenas barras ja fechadas. O motor ainda
desloca a execucao em uma barra, entao nao existe vies de futuro.

Stop e alvo saem de multiplos do ATR, o que faz o risco se adaptar a
volatilidade de cada par em vez de fixar pips na mao.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import Signals


# --------------------------------------------------------------------------
# indicadores
# --------------------------------------------------------------------------
def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    # media de Wilder
    roll_up = up.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    roll_down = down.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(50)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev = c.shift()
    tr = pd.concat([h - l, (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def sma(s: pd.Series, period: int) -> pd.Series:
    return s.rolling(period).mean()


def cross_above(a: pd.Series, b) -> np.ndarray:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return ((a > b) & (a.shift() <= b.shift())).fillna(False).to_numpy()


def cross_below(a: pd.Series, b) -> np.ndarray:
    b = b if isinstance(b, pd.Series) else pd.Series(b, index=a.index)
    return ((a < b) & (a.shift() >= b.shift())).fillna(False).to_numpy()


def _risk(sig: Signals, df: pd.DataFrame, atr_period: int,
          sl_mult: float, tp_mult: float) -> pd.Series:
    a = atr(df, atr_period)
    sig.sl_dist = (a * sl_mult).to_numpy()
    sig.tp_dist = (a * tp_mult).to_numpy()
    return a


# --------------------------------------------------------------------------
# estrategias
# --------------------------------------------------------------------------
def rsi_basico(df, period=14, oversold=30, overbought=70,
               sl_mult=2.0, tp_mult=3.0, atr_period=14) -> Signals:
    """RSI classico de reversao a media, nos dois sentidos.

    Compra quando o RSI sai da zona de sobrevenda (cruza 30 para cima).
    Vende quando sai da zona de sobrecompra (cruza 70 para baixo).
    Exige o cruzamento e nao apenas o nivel: entrar so porque o RSI esta
    abaixo de 30 faz o modelo comprar durante toda a queda.
    """
    s = Signals(len(df))
    r = rsi(df["close"], period)
    s.long_entry = cross_above(r, oversold)
    s.short_entry = cross_below(r, overbought)
    # saida tambem pelo lado oposto do oscilador
    s.long_exit = cross_above(r, overbought)
    s.short_exit = cross_below(r, oversold)
    _risk(s, df, atr_period, sl_mult, tp_mult)
    s.plots = {"RSI": r.to_numpy()}
    return s


def rsi_com_tendencia(df, period=14, oversold=35, overbought=65,
                      trend=200, sl_mult=2.0, tp_mult=3.0, atr_period=14) -> Signals:
    """RSI operado apenas a favor da tendencia da EMA de 200.

    Mesma logica do RSI basico, mas so compra acima da media longa e so
    vende abaixo dela. Os limiares sao mais frouxos (35/65) porque o filtro
    ja corta boa parte dos sinais.
    """
    s = Signals(len(df))
    r = rsi(df["close"], period)
    e = ema(df["close"], trend)
    up = (df["close"] > e).to_numpy()
    s.long_entry = cross_above(r, oversold) & up
    s.short_entry = cross_below(r, overbought) & ~up
    s.long_exit = cross_above(r, overbought)
    s.short_exit = cross_below(r, oversold)
    _risk(s, df, atr_period, sl_mult, tp_mult)
    s.plots = {"RSI": r.to_numpy(), f"EMA{trend}": e.to_numpy()}
    return s


def rsi_extremo(df, period=2, oversold=10, overbought=90, trend=200,
                sl_mult=2.0, tp_mult=2.0, atr_period=14) -> Signals:
    """RSI de 2 periodos no estilo Connors, com filtro de tendencia.

    RSI curto dispara com muito mais frequencia e pega exaustao de curto
    prazo em vez de ciclo. Entra no nivel, sem esperar o cruzamento.
    """
    s = Signals(len(df))
    r = rsi(df["close"], period)
    e = ema(df["close"], trend)
    up = (df["close"] > e).to_numpy()
    s.long_entry = (r < oversold).to_numpy() & up
    s.short_entry = (r > overbought).to_numpy() & ~up
    s.long_exit = (r > 60).to_numpy()
    s.short_exit = (r < 40).to_numpy()
    _risk(s, df, atr_period, sl_mult, tp_mult)
    s.plots = {"RSI2": r.to_numpy(), f"EMA{trend}": e.to_numpy()}
    return s


def cruzamento_medias(df, fast=20, slow=50, sl_mult=2.0, tp_mult=4.0,
                      atr_period=14) -> Signals:
    """Cruzamento de EMAs: seguidor de tendencia puro."""
    s = Signals(len(df))
    f, sl_ = ema(df["close"], fast), ema(df["close"], slow)
    s.long_entry = cross_above(f, sl_)
    s.short_entry = cross_below(f, sl_)
    s.long_exit = cross_below(f, sl_)
    s.short_exit = cross_above(f, sl_)
    _risk(s, df, atr_period, sl_mult, tp_mult)
    s.plots = {f"EMA{fast}": f.to_numpy(), f"EMA{slow}": sl_.to_numpy()}
    return s


def bandas_bollinger(df, period=20, dev=2.0, sl_mult=2.0, tp_mult=2.5,
                     atr_period=14) -> Signals:
    """Reversao a media nas bandas: compra na banda inferior, vende na superior."""
    s = Signals(len(df))
    mid = sma(df["close"], period)
    sd = df["close"].rolling(period).std()
    up, lo = mid + dev * sd, mid - dev * sd
    s.long_entry = cross_above(df["close"], lo)
    s.short_entry = cross_below(df["close"], up)
    s.long_exit = cross_above(df["close"], mid)
    s.short_exit = cross_below(df["close"], mid)
    _risk(s, df, atr_period, sl_mult, tp_mult)
    s.plots = {"BB_sup": up.to_numpy(), "BB_med": mid.to_numpy(), "BB_inf": lo.to_numpy()}
    return s


def macd_cruzamento(df, fast=12, slow=26, signal=9, sl_mult=2.0, tp_mult=3.0,
                    atr_period=14) -> Signals:
    """MACD cruzando a linha de sinal."""
    s = Signals(len(df))
    line = ema(df["close"], fast) - ema(df["close"], slow)
    sig_line = line.ewm(span=signal, adjust=False).mean()
    s.long_entry = cross_above(line, sig_line)
    s.short_entry = cross_below(line, sig_line)
    s.long_exit = cross_below(line, sig_line)
    s.short_exit = cross_above(line, sig_line)
    _risk(s, df, atr_period, sl_mult, tp_mult)
    s.plots = {"MACD": line.to_numpy(), "Sinal": sig_line.to_numpy()}
    return s


def rompimento_donchian(df, period=20, exit_period=10, sl_mult=2.0, tp_mult=6.0,
                        atr_period=14) -> Signals:
    """Rompimento de canal no estilo Tartaruga: compra na maxima de N barras."""
    s = Signals(len(df))
    hh = df["high"].rolling(period).max().shift()
    ll = df["low"].rolling(period).min().shift()
    xh = df["high"].rolling(exit_period).max().shift()
    xl = df["low"].rolling(exit_period).min().shift()
    s.long_entry = (df["close"] > hh).to_numpy()
    s.short_entry = (df["close"] < ll).to_numpy()
    s.long_exit = (df["close"] < xl).to_numpy()
    s.short_exit = (df["close"] > xh).to_numpy()
    _risk(s, df, atr_period, sl_mult, tp_mult)
    s.plots = {"Canal_sup": hh.to_numpy(), "Canal_inf": ll.to_numpy()}
    return s


def comprar_e_segurar(df, **_) -> Signals:
    """Referencia: compra na primeira barra e segura ate o fim."""
    s = Signals(len(df))
    s.long_entry[20] = True
    s.full_size = True
    s.sl_dist = np.full(len(df), df["close"].iloc[0] * 0.99)   # stop inalcancavel
    s.tp_dist = np.full(len(df), np.inf)
    return s


CATALOGO = {
    "RSI 14 reversão":          (rsi_basico, {}),
    "RSI 14 + tendência":       (rsi_com_tendencia, {}),
    "RSI 2 extremo":            (rsi_extremo, {}),
    "Cruzamento EMA 20/50":     (cruzamento_medias, {}),
    "Bandas de Bollinger":      (bandas_bollinger, {}),
    "MACD 12/26/9":             (macd_cruzamento, {}),
    "Rompimento Donchian 20":   (rompimento_donchian, {}),
    "Comprar e segurar":        (comprar_e_segurar, {}),
}
