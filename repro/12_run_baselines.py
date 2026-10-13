"""Baselines ingênuos — o "teste de sanidade" que o artigo NÃO faz.

    python repro/12_run_baselines.py

Tawosi, Moussa & Sarro (2022) e Tawosi, Al-Subaihin & Sarro (2022, LHC-SE)
mostraram que nenhuma técnica sofisticada supera com segurança um estimador
ingênuo: prever sempre a mediana do treino. Qualquer modelo que não bata esses
baselines não aprendeu nada útil com o texto.

Modelos (família "Baseline"):
  Mean            média de SP do treino
  Median          mediana de SP do treino
  ProjectMedian   mediana de SP do treino DO MESMO PROJETO (só within; nos protocolos
                  cross e LOPO o projeto de teste é novo, então não existe)
  ProjectMean     idem, com a média

ProjectMedian é o baseline mais justo no within: cada time usa sua própria
escala de story points, e ele captura exatamente isso — sem ler uma palavra.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged, save_predictions  # noqa: E402


def emit(df, pr_mask, preds, dataset, protocol, name, fold_col=None):
    out = df.loc[pr_mask, ["issue_id", "project"]].copy()
    out["y_true"] = df.loc[pr_mask, "sp"].to_numpy()
    out["y_pred"] = preds
    out["family"] = "Baseline"
    out["fold"] = fold_col if fold_col is not None else "all"
    return out


def main():
    for dataset in config.DATASETS:
        df = load_merged(dataset)

        # ---- within ----
        tr, pr = df.within == "train", df.within == "test"
        train = df[tr]
        for name, value in [("Mean", train.sp.mean()), ("Median", train.sp.median())]:
            save_predictions(emit(df, pr, value, dataset, "within", name), dataset, "within", name)
        for name, agg in [("ProjectMedian", "median"), ("ProjectMean", "mean")]:
            per_proj = train.groupby("project").sp.agg(agg)
            save_predictions(emit(df, pr, df.loc[pr, "project"].map(per_proj).to_numpy(), dataset, "within", name),
                             dataset, "within", name)

        # ---- cross ----
        train = df[df.cross == "train"]
        pr = df.cross_eval == 1
        for name, value in [("Mean", train.sp.mean()), ("Median", train.sp.median())]:
            save_predictions(emit(df, pr, value, dataset, "cross", name), dataset, "cross", name)

        # ---- LOPO ----
        for name, agg in [("Mean", "mean"), ("Median", "median")]:
            parts = []
            for p in sorted(df.project.unique()):
                pr = (df.project == p) & (df.lopo_eval == 1)
                value = df.loc[df.project != p, "sp"].agg(agg)
                parts.append(emit(df, pr, value, dataset, "lopo", name, fold_col=p))
            save_predictions(pd.concat(parts), dataset, "lopo", name)
        print(f"{dataset}: baselines salvos (within, cross, lopo)")


if __name__ == "__main__":
    main()
