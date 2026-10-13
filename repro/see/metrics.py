"""Métricas do artigo: MAE, MdAE, RMSE e PRED(25) (Seção III-B.3).

Também calculamos SA (Standardized Accuracy), que aparece no código do
repositório original e é a métrica usada por Tawosi et al. e Choetkiertikul et al.
"""
import numpy as np


def pred25(y_true, y_pred) -> float:
    """Proporção de predições com erro relativo <= 25%.

    Mesmo comportamento do código original: |y - ŷ| / y. Quando y = 0 (existem
    148 stories com 0 SP no NEO) a divisão dá inf ou NaN e a predição conta como
    erro. Isso é registrado no README como decisão herdada do código original.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(y_true - y_pred) / y_true
    return float(np.mean(rel <= 0.25))


def all_metrics(y_true, y_pred, y_train_mean=None) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = np.abs(y_true - y_pred)
    out = {
        "n": int(len(y_true)),
        "MAE": float(err.mean()),
        "MdAE": float(np.median(err)),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "PRED25": pred25(y_true, y_pred),
    }
    if y_train_mean is not None:
        mae_baseline = float(np.mean(np.abs(y_true - y_train_mean)))
        out["SA"] = float(1 - out["MAE"] / mae_baseline) if mae_baseline > 0 else float("nan")
    return out
