"""Variante FIEL AO CÓDIGO — reproduz literalmente ml_for_story_point.py (só Agile).

    python repro/11_run_ml_original_code.py                       # 5 modelos
    python repro/11_run_ml_original_code.py --models KNN LinearRegression

Por que existe: rodando o código original, KNN e Regressão Linear reproduzem
EXATAMENTE as linhas da Tabela 14 do artigo (MAE 5,268 e 5,934). Ou seja, os
resultados de ML publicados vêm deste pipeline, e não do protocolo descrito no texto:

  - matriz TF-IDF do repositório (93.192 termos, ajustada no dataset INTEIRO);
  - SVD de 3.000 componentes, também ajustada com o teste;
  - split aleatório global 70/30 (random_state=42): o teste tem 6.994 issues,
    e não ~1.000 (50% de um teste de 10%) como diz a Seção IV-A.

Memória: em float64 a SVD passa de 6 GB de RAM. Usamos float32 por padrão (~3–4 GB),
o que muda as métricas só na 3ª casa decimal. `--float64` reproduz bit a bit.
SVR é lento nesta matriz densa (mais de 1 h); rode-o por último ou no Colab.

Saída: results/predictions/agile/original_code/{modelo}.csv
"""
import argparse
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_processed, log_cost, save_predictions  # noqa: E402
from see.metrics import all_metrics  # noqa: E402

# reaproveita os hiperparâmetros de 10_run_ml.py (os mesmos do código original)
_spec = importlib.util.spec_from_file_location("ml", Path(__file__).with_name("10_run_ml.py"))
ml = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ml)

TFIDF = config.ROOT.parent / "features TF-IDF" / "tf_idf_matrix.npz"
PAPER_TABLE14 = {"SVR": 4.257, "RandomForest": 5.112, "XGBoost": 4.971, "KNN": 5.268, "LinearRegression": 5.934}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["KNN", "LinearRegression", "RandomForest", "XGBoost", "SVR"],
                    choices=ml.MODELS)
    ap.add_argument("--float64", action="store_true", help="precisão original (precisa de >6 GB de RAM)")
    args = ap.parse_args()

    df = load_processed("agile")  # mesma ordem de linhas do data_csv/data original
    y = df["sp"].to_numpy()
    X = sparse.load_npz(TFIDF)
    assert X.shape[0] == len(df), "a matriz TF-IDF não corresponde ao dataset Agile"
    if not args.float64:
        X = X.astype(np.float32)
    print(f"Matriz TF-IDF original: {X.shape}. Rodando SVD(3000)...", flush=True)
    t0 = time.perf_counter()
    Xr = TruncatedSVD(n_components=3000, random_state=42).fit_transform(X)
    print(f"  SVD em {time.perf_counter() - t0:.0f}s", flush=True)
    idx = np.arange(len(df))
    i_tr, i_te = train_test_split(idx, test_size=0.3, random_state=42)

    for name in args.models:
        model = ml.make_model(name)
        # o código original usa random_state=99 no RF e no XGBoost
        if name in ("RandomForest", "XGBoost"):
            model.set_params(random_state=99)
        if name == "KNN":
            model.set_params(metric="euclidean")
        t0 = time.perf_counter()
        model.fit(Xr[i_tr], y[i_tr])
        t1 = time.perf_counter()
        y_hat = model.predict(Xr[i_te])
        t2 = time.perf_counter()
        out = df.iloc[i_te][["issue_id", "project"]].copy()
        out["y_true"], out["y_pred"], out["family"] = y[i_te], y_hat, "ML"
        out["fold"], out["y_train_mean"] = "all", y[i_tr].mean()
        save_predictions(out, "agile", "original_code", name)
        log_cost("agile", "original_code", "ML", name, n_train=len(i_tr), n_pred=len(i_te),
                 train_seconds=round(t1 - t0, 3), infer_seconds=round(t2 - t1, 4),
                 notes="pipeline literal do repositório original")
        m = all_metrics(y[i_te], y_hat)
        print(f"  {name:<17} MAE={m['MAE']:.3f} MdAE={m['MdAE']:.3f} RMSE={m['RMSE']:.3f} "
              f"PRED25={m['PRED25']:.3f}   | artigo (Tabela 14): MAE={PAPER_TABLE14[name]}", flush=True)


if __name__ == "__main__":
    main()
