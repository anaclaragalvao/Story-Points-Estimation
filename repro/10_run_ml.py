"""Etapa ML — TF-IDF + 5 regressores clássicos (Tabela 2 do artigo). Roda na CPU.

    python repro/10_run_ml.py                          # within, nos dois datasets
    python repro/10_run_ml.py --protocol cross lopo    # generalização
    python repro/10_run_ml.py --dataset agile --models SVR KNN
    python repro/10_run_ml.py --svd 3000               # variante do CÓDIGO original
    python repro/10_run_ml.py --dataset neo --features text+repo   # EXTENSÃO: + características dos repositórios

Protocolos:
  within  treina no train do within-project e prevê o test inteiro
          (a avaliação depois filtra o subconjunto `within_eval`)
  cross   treina nos projetos de treino e prevê o subconjunto `cross_eval`
  lopo    para cada projeto, treina em todos os outros e prevê suas amostras `lopo_eval`
  perturb treina como no within e prevê os textos perturbados de data/perturbed/
          (gere-os antes com 40_perturb_texts.py); salva em perturb_s{seed}/

Diferenças em relação ao repositório original (documentadas no README):
  - o TF-IDF é ajustado só no conjunto de treino (o original ajusta no dataset
    inteiro, o que vaza informação do teste);
  - por padrão NÃO usamos SVD, porque o artigo não menciona; `--svd 3000`
    reproduz o que o código faz.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged, log_cost, save_predictions  # noqa: E402


def make_model(name: str):
    """Hiperparâmetros da Tabela 2 (coluna story-point)."""
    if name == "LinearRegression":
        return LinearRegression()
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=500, max_features="log2",
                                     random_state=config.SEED, n_jobs=-1)
    if name == "SVR":
        return SVR(kernel="rbf", C=10, epsilon=0.1)
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=5, weights="distance", n_jobs=-1)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        # colsample_bytree=0.7 vem do código original (a Tabela 2 não cita)
        return XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=8, subsample=0.9,
                            colsample_bytree=0.7, objective="reg:squarederror",
                            tree_method="hist", random_state=config.SEED, n_jobs=-1)
    raise ValueError(name)


MODELS = ["LinearRegression", "RandomForest", "SVR", "KNN", "XGBoost"]


def featurize(train_text, pred_text, svd: int):
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2))
    x_tr = vec.fit_transform(train_text)
    x_pr = vec.transform(pred_text)
    if svd:
        reducer = TruncatedSVD(n_components=min(svd, x_tr.shape[1] - 1), random_state=config.SEED)
        x_tr = reducer.fit_transform(x_tr)
        x_pr = reducer.transform(x_pr)
    return x_tr, x_pr


REPO_FEATURES = ["log_stars", "log_forks", "log_open_issues", "n_topics", "project_age_days",
                 "org_gitlab", "org_minds", "org_fpc"]
FEATURE_SUFFIX = {"text": "", "repo": "_repoonly", "text+repo": "+repo"}


def repo_matrix(df: pd.DataFrame) -> np.ndarray:
    """EXTENSÃO: características dos repositórios (só NEO; veja 04_repo_features.py)."""
    path = config.DATA_PROCESSED / "neo_repo_features.csv"
    if not path.exists():
        sys.exit("Rode antes: python repro/04_repo_features.py")
    feats = pd.read_csv(path, dtype={"issue_id": str}).set_index("issue_id")[REPO_FEATURES]
    return feats.loc[df.issue_id].to_numpy(dtype=float)


def build_x(df, tr, pr, svd: int, features: str):
    """Monta as matrizes de treino e predição para o conjunto de features escolhido."""
    from scipy import sparse
    from sklearn.preprocessing import StandardScaler
    if features == "text":
        return featurize(df.loc[tr, "text_full"], df.loc[pr, "text_full"], svd)
    scaler = StandardScaler()
    r_tr = scaler.fit_transform(repo_matrix(df[tr]))
    r_pr = scaler.transform(repo_matrix(df[pr]))
    if features == "repo":
        return r_tr, r_pr
    x_tr, x_pr = featurize(df.loc[tr, "text_full"], df.loc[pr, "text_full"], svd)
    if sparse.issparse(x_tr):
        return sparse.hstack([x_tr, sparse.csr_matrix(r_tr)]).tocsr(), sparse.hstack([x_pr, sparse.csr_matrix(r_pr)]).tocsr()
    return np.hstack([x_tr, r_tr]), np.hstack([x_pr, r_pr])


def fit_predict(name, x_tr, y_tr, x_pr):
    model = make_model(name)
    t0 = time.perf_counter()
    model.fit(x_tr, y_tr)
    t1 = time.perf_counter()
    y_hat = model.predict(x_pr)
    t2 = time.perf_counter()
    return np.asarray(y_hat, dtype=float), t1 - t0, t2 - t1


def folds(df: pd.DataFrame, protocol: str, lopo_projects=None):
    """Gera (nome_do_fold, máscara_treino, máscara_predição)."""
    if protocol == "within":
        yield "all", df.within == "train", df.within == "test"
    elif protocol == "cross":
        yield "all", df.cross == "train", df.cross_eval == 1
    elif protocol == "lopo":
        for p in lopo_projects or sorted(df.project.unique()):
            yield p, df.project != p, (df.project == p) & (df.lopo_eval == 1)
    else:
        raise ValueError(protocol)


def perturbed_sets(dataset: str):
    files = sorted((config.ROOT / "data" / "perturbed").glob(f"{dataset}_s*.csv"))
    if not files:
        sys.exit("Não há textos perturbados. Rode antes: python repro/40_perturb_texts.py")
    for f in files:
        seed = f.stem.split("_s")[-1]
        yield f"perturb_s{seed}", pd.read_csv(f, dtype={"issue_id": str}, keep_default_na=False)


def run_perturb(dataset: str, models: list, svd: int):
    df = load_merged(dataset)
    tr = df.within == "train"
    sets = list(perturbed_sets(dataset))
    y_tr = df.loc[tr, "sp"].to_numpy()
    suffix = f"_svd{svd}" if svd else ""
    for name in models:
        # treina uma vez e prevê todas as versões perturbadas
        vec_texts = [s.text_full for _, s in sets]
        x_tr, x_all = featurize(df.loc[tr, "text_full"], pd.concat(vec_texts), svd)
        y_hat_all, t_fit, t_pred = fit_predict(name, x_tr, y_tr, x_all)
        start = 0
        for proto, s in sets:
            part = s[["issue_id"]].merge(df[["issue_id", "project", "sp"]], on="issue_id")
            part = part.rename(columns={"sp": "y_true"})
            part["y_pred"] = y_hat_all[start:start + len(s)]
            part["family"], part["fold"], part["y_train_mean"] = "ML", "all", y_tr.mean()
            start += len(s)
            path = save_predictions(part, dataset, proto, name + suffix)
            print(f"  [{dataset}/{proto}] {name:<17} MAE={np.mean(np.abs(part.y_true - part.y_pred)):.3f}"
                  f"  salvo: {path.relative_to(config.ROOT)}", flush=True)
        log_cost(dataset, "perturb", "ML", name + suffix, n_train=int(tr.sum()), n_pred=len(y_hat_all),
                 train_seconds=round(t_fit, 3), infer_seconds=round(t_pred, 4))


def run(dataset: str, protocol: str, models: list, svd: int, lopo_projects=None, features: str = "text"):
    if features != "text" and config.base_dataset(dataset) != "neo":
        print(f"  pulando {dataset}: características de repositório só existem no NEO")
        return
    if protocol == "perturb":
        if features != "text":
            sys.exit("O protocolo perturb só usa texto.")
        return run_perturb(dataset, models, svd)
    df = load_merged(dataset)
    suffix = (f"_svd{svd}" if svd else "") + FEATURE_SUFFIX[features]
    collected = {m: [] for m in models}
    for fold, tr, pr in folds(df, protocol, lopo_projects):
        x_tr, x_pr = build_x(df, tr, pr, svd, features)
        y_tr = df.loc[tr, "sp"].to_numpy()
        for name in models:
            y_hat, t_fit, t_pred = fit_predict(name, x_tr, y_tr, x_pr)
            part = df.loc[pr, ["issue_id", "project"]].copy()
            part["y_true"] = df.loc[pr, "sp"].to_numpy()
            part["y_pred"] = y_hat
            part["fold"] = fold
            part["family"] = "ML"
            part["y_train_mean"] = y_tr.mean()
            collected[name].append(part)
            log_cost(dataset, protocol, "ML", name + suffix, n_train=int(tr.sum()), n_pred=int(pr.sum()),
                     train_seconds=round(t_fit, 3), infer_seconds=round(t_pred, 4),
                     notes=f"fold={fold}")
            print(f"  [{dataset}/{protocol}/{fold}] {name:<17} treino {t_fit:7.1f}s  "
                  f"MAE={np.mean(np.abs(part.y_true - y_hat)):.3f}", flush=True)
            if protocol != "lopo":  # fold único: salva já, sem esperar os outros modelos
                path = save_predictions(part, dataset, protocol, name + suffix)
                print(f"  salvo: {path.relative_to(config.ROOT)}", flush=True)
    if protocol == "lopo":
        for name, parts in collected.items():
            path = save_predictions(pd.concat(parts), dataset, protocol, name + suffix)
            print(f"  salvo: {path.relative_to(config.ROOT)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", nargs="+", default=config.DATASETS, choices=config.ALL_DATASETS)
    ap.add_argument("--protocol", nargs="+", default=["within"], choices=["within", "cross", "lopo", "perturb"])
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--svd", type=int, default=0, help="componentes de SVD (0 = sem SVD, como no artigo)")
    ap.add_argument("--lopo-projects", nargs="*", help="limitar o LOPO a alguns projetos (cada um re-treina tudo)")
    ap.add_argument("--features", default="text", choices=list(FEATURE_SUFFIX),
                    help="EXTENSÃO (só NEO): text = como no artigo; repo = só características do repositório; "
                         "text+repo = as duas")
    args = ap.parse_args()
    for dataset in args.dataset:
        for protocol in args.protocol:
            print(f"== {dataset} / {protocol} ==")
            run(dataset, protocol, args.models, args.svd, args.lopo_projects, args.features)


if __name__ == "__main__":
    main()
