"""Os indicadores nao podem olhar para o futuro."""
import numpy as np

import data
import strategies as st


def test_indicadores_sao_causais():
    df = data.get("EURUSD", "D1")
    corte = len(df) // 2
    for nome, (fn, kw) in st.CATALOGO.items():
        inteiro = fn(df, **kw)
        metade = fn(df.iloc[:corte], **kw)
        # o sinal ate o corte tem que ser igual com ou sem os dados depois dele
        for campo in ("long_entry", "short_entry", "long_exit", "short_exit"):
            np.testing.assert_array_equal(
                getattr(inteiro, campo)[:corte - 1], getattr(metade, campo)[:corte - 1],
                err_msg=f"{nome}: {campo} muda quando dados futuros entram",
            )


def test_pip_size():
    assert data.pip_size("EURUSD") == 0.0001
    assert data.pip_size("USDJPY") == 0.01
    assert data.pip_size("XAUUSD") == 0.10
