"""Motor de backtest para forex.

Premissas de execucao (todas conservadoras e explicitas):

  * Sinal calculado no fechamento da barra i e executado na ABERTURA da barra
    i+1. Sem olhar o futuro.
  * Barras do MT5 sao bid. Compra sai no ask (bid + spread), venda sai no bid.
    O spread inteiro e pago uma vez por operacao completa.
  * Se stop e alvo caberiam na mesma barra, assume-se que o STOP bateu
    primeiro. E o pior caso.
  * Tamanho da posicao por risco: arrisca risk_pct do capital atual na
    distancia ate o stop.
  * Lucro e convertido para USD quando a moeda de cotacao nao e USD
    (USDJPY, USDCHF).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Piso de spread realista, em pips. O spread gravado no historico do MT5 e
# otimista demais (registra o minimo da barra), entao usamos o maior entre o
# valor gravado e este piso.
# Modelo de conta raw/ECN, coerente com a conta do usuario: spread apertado
# MAIS comissao. Cobrar spread de conta standard junto da comissao seria
# contar o mesmo custo duas vezes.
SPREAD_FLOOR_PIPS = {
    "EURUSD": 0.2, "GBPUSD": 0.3, "USDJPY": 0.2,
    "AUDUSD": 0.3, "USDCHF": 0.4, "XAUUSD": 3.0,   # 3 x 0.10 = US$ 0.30
}
# Comissao ida-e-volta em USD por lote padrao (1 lote = 100k unidades).
COMMISSION_PER_LOT = 7.0


@dataclass
class Config:
    initial_cash: float = 10_000.0
    risk_pct: float = 0.01          # 1% do capital por operacao
    spread_floor: bool = True
    commission: bool = True
    max_bars_in_trade: int = 0      # 0 = sem limite de tempo
    max_leverage: float = 30.0


@dataclass
class Trade:
    direction: int                  # +1 comprado, -1 vendido
    entry_idx: int
    exit_idx: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float                     # unidades da moeda base
    pnl: float                      # USD
    pnl_pct: float                  # sobre o capital no momento da entrada
    reason: str                     # stop | alvo | sinal | tempo | fim
    bars_held: int


@dataclass
class Result:
    name: str
    symbol: str
    timeframe: str
    equity: pd.Series
    trades: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    plots: dict = field(default_factory=dict)


class Signals:
    """O que uma estrategia devolve, tudo alinhado ao indice das barras."""

    def __init__(self, n: int):
        self.long_entry = np.zeros(n, dtype=bool)
        self.short_entry = np.zeros(n, dtype=bool)
        self.long_exit = np.zeros(n, dtype=bool)
        self.short_exit = np.zeros(n, dtype=bool)
        self.sl_dist = np.full(n, np.nan)   # distancia ate o stop, em preco
        self.tp_dist = np.full(n, np.nan)   # distancia ate o alvo, em preco
        self.plots = {}                     # indicadores para o grafico
        self.full_size = False              # usa o capital todo (benchmark)


def _quote_to_usd(symbol: str, price: np.ndarray) -> np.ndarray:
    """Fator que converte lucro da moeda de cotacao para USD."""
    if symbol.startswith("USD"):        # USDJPY, USDCHF -> lucro em JPY/CHF
        return 1.0 / price
    return np.ones_like(price)          # EURUSD, XAUUSD -> lucro ja em USD


def run(df: pd.DataFrame, sig: Signals, name: str, symbol: str,
        timeframe: str, cfg: Config | None = None) -> Result:
    cfg = cfg or Config()
    from data import pip_size

    pip = pip_size(symbol)
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    times = df.index

    spread = df["spread_price"].to_numpy(float)
    if cfg.spread_floor:
        floor = SPREAD_FLOOR_PIPS.get(symbol, 1.0) * pip
        spread = np.maximum(spread, floor)

    usd = _quote_to_usd(symbol, c)

    cash = cfg.initial_cash
    equity = np.full(n, cfg.initial_cash)
    trades = []

    pos = 0          # 0 fora, +1 comprado, -1 vendido
    entry_px = sl = tp = size = 0.0
    entry_i = 0
    entry_cash = cfg.initial_cash

    for i in range(1, n):
        # ---- valor da posicao aberta, marcado a mercado ----
        equity[i] = cash + (pos * (c[i] - entry_px) * size * usd[i] if pos else 0.0)

        # ---- 1. a posicao aberta bate stop/alvo dentro DESTA barra? ----
        if pos != 0:
            exit_px = None
            reason = ""
            if pos == 1:
                if l[i] <= sl:                      # stop primeiro (pior caso)
                    exit_px, reason = sl, "stop"
                elif h[i] >= tp:
                    exit_px, reason = tp, "alvo"
            else:
                # venda fecha comprando no ask = bid + spread
                if h[i] + spread[i] >= sl:
                    exit_px, reason = sl, "stop"
                elif l[i] + spread[i] <= tp:
                    exit_px, reason = tp, "alvo"

            # saida por sinal ou por tempo -> executa na abertura desta barra
            if exit_px is None:
                timeout = bool(cfg.max_bars_in_trade) and (i - entry_i) >= cfg.max_bars_in_trade
                by_signal = (pos == 1 and sig.long_exit[i - 1]) or \
                            (pos == -1 and sig.short_exit[i - 1])
                if by_signal or timeout:
                    exit_px = o[i] if pos == 1 else o[i] + spread[i]
                    reason = "sinal" if by_signal else "tempo"

            if exit_px is not None:
                gross = pos * (exit_px - entry_px) * size * usd[i]
                comm = (size / 100_000) * COMMISSION_PER_LOT if cfg.commission else 0.0
                pnl = gross - comm
                cash += pnl
                equity[i] = cash
                trades.append(Trade(
                    direction=pos, entry_idx=entry_i, exit_idx=i,
                    entry_time=times[entry_i], exit_time=times[i],
                    entry_price=entry_px, exit_price=exit_px, size=size,
                    pnl=pnl, pnl_pct=pnl / entry_cash * 100, reason=reason,
                    bars_held=i - entry_i,
                ))
                pos = 0

        # ---- 2. abrir posicao nova a partir do sinal da barra anterior ----
        if pos == 0 and cash > 0:
            go_long = sig.long_entry[i - 1]
            go_short = sig.short_entry[i - 1]
            if go_long or go_short:
                d = 1 if go_long else -1
                sl_d = sig.sl_dist[i - 1]
                tp_d = sig.tp_dist[i - 1]
                if np.isfinite(sl_d) and sl_d > 0:
                    px = o[i] + spread[i] if d == 1 else o[i]
                    risk_usd = cash * cfg.risk_pct
                    per_unit = sl_d * usd[i]        # risco por unidade, em USD
                    sz = risk_usd / per_unit if per_unit > 0 else 0.0
                    if sig.full_size:           # benchmark: nocional = capital
                        sz = cash / (c[i] if not symbol.startswith("USD") else 1.0)
                    # teto de alavancagem sobre o nocional em USD
                    notional_usd_per_unit = c[i] * usd[i] if symbol.startswith("USD") else c[i]
                    if symbol.startswith("USD"):
                        notional_usd_per_unit = 1.0   # 1 unidade de USD = 1 USD
                    cap = cash * cfg.max_leverage / notional_usd_per_unit
                    sz = min(sz, cap)
                    if sz > 0:
                        pos, entry_px, size, entry_i = d, px, sz, i
                        entry_cash = cash
                        sl = px - d * sl_d
                        tp = px + d * tp_d if np.isfinite(tp_d) else px + d * sl_d * 1000

    # ---- fecha o que sobrou aberto na ultima barra ----
    if pos != 0:
        i = n - 1
        exit_px = c[i] if pos == 1 else c[i] + spread[i]
        gross = pos * (exit_px - entry_px) * size * usd[i]
        comm = (size / 100_000) * COMMISSION_PER_LOT if cfg.commission else 0.0
        pnl = gross - comm
        cash += pnl
        equity[i] = cash
        trades.append(Trade(
            direction=pos, entry_idx=entry_i, exit_idx=i,
            entry_time=times[entry_i], exit_time=times[i],
            entry_price=entry_px, exit_price=exit_px, size=size,
            pnl=pnl, pnl_pct=pnl / entry_cash * 100, reason="fim",
            bars_held=i - entry_i,
        ))

    eq = pd.Series(equity, index=times, name="equity")
    res = Result(name=name, symbol=symbol, timeframe=timeframe,
                 equity=eq, trades=trades, plots=sig.plots)
    res.stats = compute_stats(res, df)
    return res


def compute_stats(res: Result, df: pd.DataFrame) -> dict:
    eq = res.equity
    trades = res.trades
    n_days = max((eq.index[-1] - eq.index[0]).days, 1)
    years = n_days / 365.25

    total_ret = (eq.iloc[-1] / eq.iloc[0] - 1) * 100
    cagr = ((eq.iloc[-1] / eq.iloc[0]) ** (1 / years) - 1) * 100 if eq.iloc[-1] > 0 else -100.0

    # retornos diarios da curva de capital -> metricas comparaveis entre TFs
    daily = eq.resample("1D").last().ffill()
    dr = daily.pct_change().dropna()
    ann = np.sqrt(252)
    sharpe = float(dr.mean() / dr.std() * ann) if len(dr) > 1 and dr.std() > 0 else 0.0
    neg = dr[dr < 0]
    sortino = float(dr.mean() / neg.std() * ann) if len(neg) > 1 and neg.std() > 0 else 0.0

    running_max = eq.cummax()
    dd = (eq / running_max - 1) * 100
    max_dd = float(dd.min())

    under = (eq < running_max).to_numpy()
    longest = cur = 0
    for u in under:
        cur = cur + 1 if u else 0
        longest = max(longest, cur)
    bars_per_day = len(eq) / n_days
    dd_days = longest / bars_per_day if bars_per_day > 0 else 0.0

    calmar = cagr / abs(max_dd) if max_dd < 0 else 0.0

    pnls = np.array([t.pnl for t in trades]) if trades else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) else 0.0

    exposure = sum(t.bars_held for t in trades) / len(eq) * 100 if trades else 0.0
    sqn = float(np.sqrt(len(pnls)) * pnls.mean() / pnls.std()) if len(pnls) > 1 and pnls.std() > 0 else 0.0
    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    if gross_loss > 0:
        pf = round(gross_win / gross_loss, 2)
    else:
        pf = 99.0 if gross_win > 0 else 0.0

    return {
        "retorno_pct": round(float(total_ret), 2),
        "cagr_pct": round(float(cagr), 2),
        "buy_hold_pct": round(float(bh), 2),
        "capital_final": round(float(eq.iloc[-1]), 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(float(calmar), 2),
        "max_dd_pct": round(max_dd, 2),
        "max_dd_dias": round(float(dd_days), 1),
        "n_trades": len(trades),
        "taxa_acerto_pct": round(float(len(wins) / len(pnls) * 100), 2) if len(pnls) else 0.0,
        "fator_lucro": pf,
        "expectativa_usd": round(float(pnls.mean()), 2) if len(pnls) else 0.0,
        "ganho_medio": round(float(wins.mean()), 2) if len(wins) else 0.0,
        "perda_media": round(float(losses.mean()), 2) if len(losses) else 0.0,
        "maior_ganho": round(float(pnls.max()), 2) if len(pnls) else 0.0,
        "maior_perda": round(float(pnls.min()), 2) if len(pnls) else 0.0,
        "exposicao_pct": round(float(exposure), 2),
        "barras_por_trade": round(float(np.mean([t.bars_held for t in trades])), 1) if trades else 0.0,
        "sqn": round(sqn, 2),
        "anos": round(years, 2),
    }
