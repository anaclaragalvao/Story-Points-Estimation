"""EXTENSÃO (não faz parte do artigo) — o contexto do repositório ajuda a ESTIMAR story points?

Usa os metadados que o RAISE extrai do GitHub (08_project_cards.py --coletar/--exportar) de
duas formas e compara tudo com a mediana do projeto, no subconjunto within_eval das issues de
projetos com repositório (raise_repos.json):

  A. ML com características do repositório: estrelas, forks, watchers, contribuidores,
     releases, issues abertas, tópicos, linguagens e idade do repositório. Três versões
     treinadas no mesmo treino (within): só texto, só repositório, texto + repositório.
  B. LLM com o cartão do projeto: o prompt ZS1 com e sem o cartão (30_run_llm.py --cartao).
     Esta parte só aparece se as duas predições existirem.
  C. Exploratório, por projeto: correlação de Spearman entre as características do
     repositório e a mediana de story points do projeto (n = número de projetos).

    python repro/31_estimativa_contexto.py
    python repro/31_estimativa_contexto.py --models KNN XGBoost RandomForest --llm llama3

Antes: 08_project_cards.py (--coletar e --exportar bastam para A e C; os cartões são para B),
12_run_baselines.py e, para B, os dois comandos:
    python repro/30_run_llm.py --provider ollama --model llama3
    python repro/30_run_llm.py --provider ollama --model llama3 --cartao

Saídas:
  results/predictions/{dataset}/within_raise/{modelo}{,_raiseonly,+raise}.csv
  results/tables/raise_repo_features.csv      características por projeto
  results/tables/contexto_estimativa.csv      comparações pareadas (A e B)
  results/tables/contexto_correlacao.csv      correlações por projeto (C)

Como ler (cuidados que valem para o TCC):
  - As características do repositório são iguais para todas as issues de um projeto. No
    within-project elas funcionam como um identificador do projeto, o que tende a empatar
    com a mediana do projeto (foi o que a Extensão 1 mostrou no NEO).
  - Os metadados são um retrato da data da coleta (2026), anos depois das issues.
  - Na parte C há só 8 projetos no Agile e 9 no NEO: é descritivo, sem poder estatístico.
  - Predições inválidas do LLM (sem número) viram a mediana do treino, como no 90_evaluate.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged, save_predictions  # noqa: E402

EXPORT = config.DATA_RAW / "raise" / "githubmetadata.json"
FEATURES = ["log_stars", "log_forks", "log_watchers", "log_contributors", "log_releases", "log_open_issues",
            "n_topics", "n_languages", "pct_main_language", "repo_age_years"]
PROTO = "within_raise"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"),
                                                  config.ROOT / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- características
def repo_features() -> pd.DataFrame:
    """Uma linha por (dataset, projeto) com as características numéricas do repositório."""
    if not EXPORT.exists():
        sys.exit(f"{EXPORT} não existe. Rode antes: python repro/08_project_cards.py --coletar e --exportar")
    meta = {str(r.get("repository", "")).lower(): r for r in json.loads(EXPORT.read_text(encoding="utf-8"))}
    mapa = json.loads(config.RAISE_REPOS.read_text(encoding="utf-8"))
    rows = []
    for ds in config.DATASETS:
        for proj, repos in mapa.get(ds, {}).items():
            m = meta.get(repos[0].lower())
            if m is None:
                print(f"  {ds}/{proj}: {repos[0]} não está na exportação do RAISE")
                continue
            langs = m.get("languages") or {}
            if isinstance(langs, str):
                langs = json.loads(langs or "{}")
            langs = langs.get("languages", []) if isinstance(langs, dict) else langs
            topics = m.get("topics") or []
            if isinstance(topics, str):
                topics = json.loads(topics or "[]")
            created = pd.to_datetime(m.get("github_created_at"), utc=True, errors="coerce")
            mined = pd.to_datetime(m.get("time_mined") or m.get("last_sync"), utc=True, errors="coerce")

            def num(k):
                v = m.get(k)
                return np.nan if v is None else float(v)
            rows.append({
                "dataset": ds, "project": str(proj), "repo": m.get("repository"),
                "stars": num("stars_count"), "forks": num("forks_count"), "watchers": num("watchers_count"),
                "contributors": num("contributors_count"), "releases": num("releases_count"),
                "open_issues": num("open_issues_count"), "n_topics": float(len(topics)),
                "n_languages": float(len(langs)),
                "pct_main_language": max([float(d.get("percentage", 0)) for d in langs], default=np.nan),
                "repo_age_years": (mined - created).days / 365.25 if pd.notna(created) and pd.notna(mined) else np.nan,
            })
    f = pd.DataFrame(rows)
    for c in ["stars", "forks", "watchers", "contributors", "releases", "open_issues"]:
        f[f"log_{c}"] = np.log1p(f[c])
    falt = {c: int(f[c].isna().sum()) for c in FEATURES if f[c].isna().any()}
    if falt:
        print(f"  valores ausentes (preenchidos com a mediana dos projetos do dataset): {falt}")
        for c in falt:
            f[c] = f.groupby("dataset")[c].transform(lambda s: s.fillna(s.median())).fillna(0)
    return f


# --------------------------------------------------------------------------- comparações
def paired(a: pd.Series, b: pd.Series, label_a: str, label_b: str, y: pd.Series, seed=0) -> dict:
    """Erro absoluto pareado: Δ = |erro b| − |erro a| (negativo = b erra menos)."""
    ea, eb = (y - a).abs(), (y - b).abs()
    d = (eb - ea).dropna()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d.to_numpy(), (5000, len(d)), replace=True).mean(1) if len(d) >= 5 else np.array([np.nan])
    p = stats.wilcoxon(d).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
    return {"comparação": f"{label_b} × {label_a}", f"MAE A": ea.mean(), "MAE B": eb.mean(),
            "Δ MAE (B − A)": d.mean(), "IC95% inf": np.percentile(boots, 2.5), "IC95% sup": np.percentile(boots, 97.5),
            "p (Wilcoxon)": p, "n": len(d)}


def run_ml(ds: str, feats: pd.DataFrame, models: list) -> dict:
    """Treina as três versões no within restrito aos projetos com repositório. Devolve {nome: y_pred no eval}."""
    from scipy import sparse
    from sklearn.preprocessing import StandardScaler
    ml = load_script("10_run_ml.py")
    df = load_merged(ds)
    fp = feats[feats.dataset == ds].set_index("project")
    df = df[df.project.isin(fp.index)].reset_index(drop=True)
    tr, pr = (df.within == "train").to_numpy(), (df.within == "test").to_numpy()
    y_tr = df.loc[tr, "sp"].to_numpy()
    R = fp.loc[df.project, FEATURES].to_numpy(dtype=float)
    sc = StandardScaler().fit(R[tr])
    r_tr, r_pr = sc.transform(R[tr]), sc.transform(R[pr])
    t_tr, t_pr = ml.featurize(df.loc[tr, "text_full"], df.loc[pr, "text_full"], 0)
    sets = {"": (t_tr, t_pr), "_raiseonly": (r_tr, r_pr),
            "+raise": (sparse.hstack([t_tr, sparse.csr_matrix(r_tr)]).tocsr(),
                       sparse.hstack([t_pr, sparse.csr_matrix(r_pr)]).tocsr())}
    out = {}
    for name in models:
        for suffix, (x_tr, x_pr) in sets.items():
            model = ml.make_model(name)
            model.fit(x_tr, y_tr)
            part = df.loc[pr, ["issue_id", "project"]].copy()
            part["y_true"] = df.loc[pr, "sp"].to_numpy()
            part["y_pred"] = np.asarray(model.predict(x_pr), dtype=float)
            part["family"], part["fold"], part["y_train_mean"] = "ML", "all", y_tr.mean()
            save_predictions(part, ds, PROTO, name + suffix)
            out[name + suffix] = part.set_index("issue_id").y_pred
            ev = part[df.loc[pr, "within_eval"].to_numpy() == 1]
            print(f"  [{ds}] {name + suffix:<22} MAE (within_eval) = {np.mean(np.abs(ev.y_true - ev.y_pred)):.3f}",
                  flush=True)
    # mediana do projeto no mesmo treino restrito
    med = df[tr].groupby("project").sp.median()
    out["ProjectMedian"] = pd.Series(df.loc[pr, "project"].map(med).to_numpy(), index=df.loc[pr, "issue_id"])
    return out, df[pr & (df.within_eval == 1).to_numpy()].set_index("issue_id")


def llm_preds(ds: str, llm: str):
    base = config.PREDICTIONS / ds / "within"
    files = {"LLM ZS1": base / f"{llm}__ZS1.csv", "LLM ZS1 + cartão": base / f"{llm}__ZS1__cartao.csv"}
    if not all(f.exists() for f in files.values()):
        faltam = [str(f.relative_to(config.ROOT)) for f in files.values() if not f.exists()]
        print(f"  [{ds}] parte B pulada: faltam {faltam}")
        return {}
    return {k: pd.read_csv(f, dtype={"issue_id": str}).set_index("issue_id").y_pred.astype(float)
            for k, f in files.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["KNN", "XGBoost"],
                    help="modelos de ML (SVR e RandomForest são mais lentos)")
    ap.add_argument("--llm", default="llama3", help="nome do modelo usado no 30_run_llm.py (ex.: llama3)")
    args = ap.parse_args()

    feats = repo_features()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    feats.to_csv(config.TABLES / "raise_repo_features.csv", index=False)
    print("\nCaracterísticas por projeto (metadados do RAISE):")
    print(feats[["dataset", "project", "repo", "stars", "forks", "contributors", "releases", "n_topics",
                 "n_languages", "repo_age_years"]].round(1).to_string(index=False))

    rows, corr = [], []
    for ds in config.DATASETS:
        if not (feats.dataset == ds).any():
            continue
        print(f"\n== {ds}: ML no within restrito aos projetos com repositório ==")
        preds, ev = run_ml(ds, feats, args.models)
        y = ev.sp
        fallback = load_merged(ds).query("within == 'train'").sp.median()
        P = {k: v.reindex(ev.index) for k, v in preds.items()}
        for name in args.models:
            for a, b in [(name, name + "+raise"), (name, name + "_raiseonly"), ("ProjectMedian", name),
                         ("ProjectMedian", name + "+raise"), ("ProjectMedian", name + "_raiseonly")]:
                rows.append({"dataset": ds, "parte": "A. ML + repositório", **paired(P[a], P[b], a, b, y)})
        L = llm_preds(ds, args.llm)
        if L:
            idx = ev.index.intersection(L["LLM ZS1"].index).intersection(L["LLM ZS1 + cartão"].index)
            yl = y.loc[idx]
            Lf = {k: v.loc[idx].fillna(fallback) for k, v in L.items()}
            pm = P["ProjectMedian"].loc[idx]
            for a, b, pa, pb in [("LLM ZS1", "LLM ZS1 + cartão", Lf["LLM ZS1"], Lf["LLM ZS1 + cartão"]),
                                 ("ProjectMedian", "LLM ZS1", pm, Lf["LLM ZS1"]),
                                 ("ProjectMedian", "LLM ZS1 + cartão", pm, Lf["LLM ZS1 + cartão"])]:
                rows.append({"dataset": ds, "parte": "B. LLM + cartão", **paired(pa, pb, a, b, yl)})

        # C. por projeto: características × mediana de SP do projeto (todo o dataset)
        d = load_merged(ds)
        sp = d[d.project.isin(feats[feats.dataset == ds].project)].groupby("project").sp.median().rename("sp_mediana")
        fp = feats[feats.dataset == ds].set_index("project").join(sp)
        for c in FEATURES:
            if fp[c].nunique() > 1:
                rho, p = stats.spearmanr(fp[c], fp.sp_mediana)
                corr.append({"dataset": ds, "característica": c, "rho (Spearman)": rho, "p": p, "n projetos": len(fp)})

    res = pd.DataFrame(rows).set_index(["dataset", "parte", "comparação"])
    res.to_csv(config.TABLES / "contexto_estimativa.csv")
    pd.set_option("display.width", 200)
    print("\n== Comparações pareadas no within_eval (Δ negativo = B erra menos que A) ==")
    print(res.round(3).to_string())
    if corr:
        c = pd.DataFrame(corr)
        c.to_csv(config.TABLES / "contexto_correlacao.csv", index=False)
        print("\n== C. Exploratório: característica do repositório × mediana de SP do projeto ==")
        print(c.round(3).to_string(index=False))
        print("Só 8–9 projetos por dataset: trate como descritivo, não como evidência.")
    print(f"\nTabelas em {config.TABLES.relative_to(config.ROOT)}/: raise_repo_features.csv, "
          "contexto_estimativa.csv, contexto_correlacao.csv")


if __name__ == "__main__":
    main()
