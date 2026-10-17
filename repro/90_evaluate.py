"""Avaliação — calcula as métricas a partir das predições salvas por qualquer família.

    python repro/90_evaluate.py

Lê results/predictions/{dataset}/{protocol}/*.csv e grava
results/tables/metrics.csv, com uma linha por dataset × protocolo × modelo:

  within   métricas no subconjunto `within_eval` (protocolo principal do artigo),
           e também no test set completo (colunas *_fulltest) para conferir o
           que o artigo afirma na Seção VI-A (diferença < 2%)
  cross    métricas no subconjunto `cross_eval`
  lopo     métricas por projeto deixado de fora, depois a média entre os folds
           (como o artigo descreve)
  outros   (fewshot, perturb_*) todas as linhas do arquivo

Predições inválidas (LLM que não devolveu número) entram como NaN. Elas são
contadas em `n_invalid`, e as métricas são calculadas trocando-as pela
mediana do treino (`fallback`), para não premiar o modelo por descartar casos
difíceis.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged  # noqa: E402
from see.metrics import all_metrics  # noqa: E402

FAMILY_ORDER = {"Baseline": -1, "ML": 0, "Transformer": 1, "LLM": 2}


def score(pred: pd.DataFrame, train_median: float, train_mean: float) -> dict:
    y_pred = pred["y_pred"].astype(float)
    n_invalid = int(y_pred.isna().sum())
    m = all_metrics(pred["y_true"], y_pred.fillna(train_median), y_train_mean=train_mean)
    m["n_invalid"] = n_invalid
    return m


def evaluate_file(path: Path, dataset: str, protocol: str, df: pd.DataFrame) -> list:
    pred = pd.read_csv(path, dtype={"issue_id": str, "project": str})
    model = path.stem
    family = pred["family"].iloc[0] if "family" in pred.columns else "?"
    train = df[df.within == "train"]
    med, mean = float(train.sp.median()), float(train.sp.mean())
    base = {"dataset": dataset, "protocol": protocol, "family": family, "model": model}

    if protocol == "within":
        ids_eval = set(df.loc[df.within_eval == 1, "issue_id"])
        sub = pred[pred.issue_id.isin(ids_eval)]
        if len(sub) < len(ids_eval):
            print(f"  aviso: {path.name} cobre {len(sub)}/{len(ids_eval)} issues do subconjunto")
        row = {**base, **score(sub, med, mean)}
        if len(pred) > len(sub):  # o modelo previu o test inteiro
            full = score(pred, med, mean)
            row.update({f"{k}_fulltest": v for k, v in full.items() if k in ("n", "MAE", "MdAE", "RMSE", "PRED25")})
        return [row]

    if protocol == "lopo":
        per_fold = []
        for proj, g in pred.groupby("project"):
            tr = df[df.project != proj]
            per_fold.append(score(g, float(tr.sp.median()), float(tr.sp.mean())))
        agg = pd.DataFrame(per_fold)
        row = {**base, **agg.drop(columns=["n", "n_invalid"]).mean().to_dict(),
               "n": int(agg.n.sum()), "n_invalid": int(agg.n_invalid.sum()), "n_folds": len(agg)}
        return [row]

    if protocol == "cross":
        tr = df[df.cross == "train"]
        med, mean = float(tr.sp.median()), float(tr.sp.mean())
    return [{**base, **score(pred, med, mean)}]


def main():
    rows = []
    for dataset in config.ALL_DATASETS:
        base = config.PREDICTIONS / dataset
        if not base.exists():
            continue
        df = load_merged(dataset)
        for proto_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(proto_dir.glob("*.csv")):
                rows.extend(evaluate_file(f, dataset, proto_dir.name, df))
    if not rows:
        sys.exit("Nenhuma predição encontrada em results/predictions/.")
    out = pd.DataFrame(rows)
    out["_fam"] = out.family.map(FAMILY_ORDER).fillna(9)
    out = out.sort_values(["dataset", "protocol", "_fam", "model"]).drop(columns="_fam")
    config.TABLES.mkdir(parents=True, exist_ok=True)
    out.to_csv(config.TABLES / "metrics.csv", index=False)

    cols = ["family", "model", "n", "n_invalid", "MAE", "MdAE", "RMSE", "PRED25"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 160):
        for (ds, proto), g in out.groupby(["dataset", "protocol"], sort=False):
            print(f"\n=== {ds} / {proto} ===")
            print(g[cols].to_string(index=False))
    print(f"\nTabela completa: {config.TABLES / 'metrics.csv'}")


if __name__ == "__main__":
    main()
