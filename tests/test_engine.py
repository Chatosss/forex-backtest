"""Testes do motor com dados sinteticos, onde o resultado certo e conhecido."""
import numpy as np
import pandas as pd
import pytest

import engine
from engine import Config, Signals


def barras(opens, highs, lows, closes, spread=0.0):
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="D")
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows, "close": closes,
        "volume": 1, "spread_price": spread,
    }, index=idx)


def cfg_sem_custo(**kw):
    return Config(spread_floor=False, commission=False, **kw)


def test_sinal_executa_na_abertura_da_barra_seguinte():
    df = barras([1.0, 1.10, 1.20, 1.30], [1.0, 1.15, 1.25, 1.35],
                [1.0, 1.05, 1.15, 1.25], [1.0, 1.12, 1.22, 1.32])
    sig = Signals(len(df))
    sig.long_entry[0] = True
    sig.sl_dist[:] = 0.5
    sig.tp_dist[:] = np.inf
    r = engine.run(df, sig, "t", "EURUSD", "D1", cfg_sem_custo())
    t = r.trades[0]
    assert t.entry_idx == 1
    assert t.entry_price == pytest.approx(1.10)   # abertura da barra 1, nao fechamento da 0


def test_stop_tem_prioridade_quando_stop_e_alvo_cabem_na_mesma_barra():
    # barra 2 vai de 0.90 a 1.20: toca stop (0.95) e alvo (1.10) ao mesmo tempo
    df = barras([1.0, 1.0, 1.0], [1.0, 1.0, 1.20], [1.0, 1.0, 0.90], [1.0, 1.0, 1.0])
    sig = Signals(len(df))
    sig.long_entry[0] = True
    sig.sl_dist[:] = 0.05
    sig.tp_dist[:] = 0.10
    r = engine.run(df, sig, "t", "EURUSD", "D1", cfg_sem_custo())
    assert r.trades[0].reason == "stop"
    assert r.trades[0].pnl < 0


def test_stop_custa_exatamente_o_risco_configurado():
    df = barras([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.90], [1.0, 1.0, 0.95])
    sig = Signals(len(df))
    sig.long_entry[0] = True
    sig.sl_dist[:] = 0.05
    sig.tp_dist[:] = 1.0
    r = engine.run(df, sig, "t", "EURUSD", "D1", cfg_sem_custo(risk_pct=0.01, max_leverage=1000))
    assert r.trades[0].pnl == pytest.approx(-100.0)   # 1% de 10.000


def test_comissao_e_cobrada_por_lote():
    df = barras([1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.90], [1.0, 1.0, 0.95])
    sig = Signals(len(df))
    sig.long_entry[0] = True
    sig.sl_dist[:] = 0.05
    sig.tp_dist[:] = 1.0
    base = Config(spread_floor=False, commission=False, max_leverage=1000)
    com = Config(spread_floor=False, commission=True, max_leverage=1000)
    r0 = engine.run(df, sig, "t", "EURUSD", "D1", base).trades[0]
    r1 = engine.run(df, sig, "t", "EURUSD", "D1", com).trades[0]
    lotes = r0.size / 100_000
    assert r0.pnl - r1.pnl == pytest.approx(lotes * engine.COMMISSION_PER_LOT)


def test_venda_entra_no_bid_e_sai_no_ask():
    spread = 0.001
    df = barras([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], spread=spread)
    sig = Signals(len(df))
    sig.short_entry[0] = True
    sig.sl_dist[:] = 0.5
    sig.tp_dist[:] = 0.5
    r = engine.run(df, sig, "t", "EURUSD", "D1", Config(spread_floor=False, commission=False))
    t = r.trades[0]
    assert t.entry_price == pytest.approx(1.0)            # vende no bid
    assert t.exit_price == pytest.approx(1.0 + spread)    # recompra no ask
    assert t.pnl < 0                                       # pagou o spread


def test_curva_parada_sem_operacoes():
    n = 30
    df = barras([1.0] * n, [1.0] * n, [1.0] * n, [1.0] * n)
    r = engine.run(df, Signals(n), "t", "EURUSD", "D1", cfg_sem_custo())
    assert r.stats["n_trades"] == 0
    assert r.stats["retorno_pct"] == 0
    assert r.stats["max_dd_pct"] == 0
