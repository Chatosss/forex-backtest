"""Roda a matriz de backtests e exporta o JSON que alimenta o dashboard."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import data
import engine
import strategies as st

OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
DASHBOARD = Path(__file__).parent / "dashboard" / "dados.js"

PARES = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCHF", "XAUUSD"]
TEMPOS = ["D1", "H1"]
INICIO = "2021-01-01"


def compacta_candles(df, max_barras=None):
    """OHLC em arrays paralelos: bem menor que lista de objetos."""
    if max_barras and len(df) > max_barras:
        df = df.iloc[-max_barras:]
    dig = 3 if df["close"].iloc[-1] > 50 else 5
    return {
        "t": [int(ts.timestamp()) for ts in df.index],
        "o": [round(v, dig) for v in df["open"]],
        "h": [round(v, dig) for v in df["high"]],
        "l": [round(v, dig) for v in df["low"]],
        "c": [round(v, dig) for v in df["close"]],
    }


def main(pares=PARES, tempos=TEMPOS, fonte="mt5"):
    resultados = []
    candles = {}
    linhas_tempo = {}       # eixo de tempo das curvas, compartilhado por par/tf
    fim = ""                # ultima data presente nos dados carregados

    for tf in tempos:
        for sym in pares:
            try:
                df = data.get(sym, tf, start=INICIO, source=fonte)
            except Exception as e:  # sem cache e sem MT5/Yahoo disponivel
                print(f"{tf} {sym:7} pulado: {e}")
                continue
            chave = f"{sym}_{tf}"
            fim = max(fim, str(df.index[-1].date()))
            # no H1 o grafico mostra a janela mais recente; o backtest usa tudo
            candles[chave] = compacta_candles(df, max_barras=4000 if tf == "H1" else None)
            base_ts = candles[chave]["t"][0]

            for nome, (fn, kw) in st.CATALOGO.items():
                sig = fn(df, **kw)
                r = engine.run(df, sig, nome, sym, tf)
                dig = 3 if df["close"].iloc[-1] > 50 else 5

                # so exporta marcadores que caem dentro da janela do grafico
                ops = []
                for t in r.trades:
                    te, tx = int(t.entry_time.timestamp()), int(t.exit_time.timestamp())
                    if tx < base_ts:
                        continue
                    ops.append([
                        te, tx, t.direction,
                        round(t.entry_price, dig), round(t.exit_price, dig),
                        round(t.pnl, 2), t.reason,
                    ])

                # curva de capital reamostrada; o eixo de tempo e guardado
                # uma vez so por par/tf em vez de repetir em cada estrategia
                eq = r.equity.resample("1D").last().ffill().dropna()
                if chave not in linhas_tempo:
                    linhas_tempo[chave] = [int(ts.timestamp()) for ts in eq.index]
                curva = [round(float(v), 1) for v in eq]

                resultados.append({
                    "estrategia": nome,
                    "par": sym,
                    "tempo": tf,
                    "id": f"{nome}|{sym}|{tf}",
                    "stats": r.stats,
                    "ops": ops,
                    "n_ops_total": len(r.trades),
                    "curva": curva,
                })
                s = r.stats
                print(f"{tf} {sym:7} {nome:26} ret={s['retorno_pct']:>9.2f}%  "
                      f"dd={s['max_dd_pct']:>7.2f}%  sh={s['sharpe']:>5.2f}  "
                      f"ops={s['n_trades']:>5}  PF={s['fator_lucro']}")

    payload = {
        "meta": {
            "inicio": INICIO,
            "fim": fim,
            "capital_inicial": engine.Config().initial_cash,
            "risco_por_op_pct": engine.Config().risk_pct * 100,
            "comissao_por_lote": engine.COMMISSION_PER_LOT,
            "spread_piso": engine.SPREAD_FLOOR_PIPS,
            "fonte": "MetaTrader 5 - somente leitura" if fonte == "mt5" else "Yahoo Finance (diario)",
            "pares": list(pares),
            "tempos": list(tempos),
        },
        "candles": candles,
        "linhas_tempo": linhas_tempo,
        "resultados": resultados,
    }

    p = OUT / "resultados.json"
    p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"\n{len(resultados)} backtests -> {p}  ({p.stat().st_size/1e6:.2f} MB)")

    # o painel abre direto do disco, entao os dados vao num .js em vez de fetch()
    DASHBOARD.write_text("window.LAB=" + p.read_text(encoding="utf-8") + ";", encoding="utf-8")
    print(f"painel atualizado -> {DASHBOARD}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Roda a matriz de backtests forex.")
    ap.add_argument("--pares", nargs="+", default=PARES)
    ap.add_argument("--tempos", nargs="+", default=TEMPOS, choices=["M15", "H1", "H4", "D1"])
    ap.add_argument("--fonte", default="mt5", choices=["mt5", "yahoo"])
    a = ap.parse_args()
    main(a.pares, a.tempos, a.fonte)
