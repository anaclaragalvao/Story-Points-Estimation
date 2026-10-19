"""EXTENSÃO (não faz parte do artigo) — o contexto do repositório ajuda a ESTIMAR story points?

Usa os metadados que o RAISE extrai do GitHub (08_project_cards.py --coletar/--exportar),
só nas issues de projetos com repositório (raise_repos.json):

  A. ML com características do repositório: estrelas, forks, watchers, contribuidores,
     releases, issues abertas, tópicos, número de linguagens, % da linguagem principal,
     idade do repositório e a LINGUAGEM PRINCIPAL (uma coluna 0/1 por linguagem). Três
     versões treinadas nos mesmos dados: só texto, só repositório, texto + repositório.
     Em dois protocolos:
       within  projeto conhecido: treina no train do próprio projeto e avalia no within_eval.
               Referência: mediana do projeto (ProjectMedian).
       lopo    projeto NOVO: para cada projeto, treina nos outros e avalia nas amostras
               lopo_eval dele. É o teste justo para "saber a linguagem/o tipo do projeto
               ajuda?", porque o modelo não conhece a escala do time. Referência: mediana
               do treino (Median), já que a mediana do projeto não existe para um projeto novo.
  B. LLM com o cartão do projeto: o prompt ZS1 com e sem o cartão (30_run_llm.py --cartao),
     no within. Só aparece se as duas predições existirem.
  C. Exploratório, por projeto: correlação de Spearman entre as características do
     repositório e a mediana de story points do projeto (n = número de projetos).

    python repro/31_estimativa_contexto.py --datasets agile_limpo neo_limpo
    python repro/31_estimativa_contexto.py --protocolos within          # só o within (mais rápido)
    python repro/31_estimativa_contexto.py --models KNN XGBoost RandomForest --llm llama3

Antes: 08_project_cards.py (--coletar e --exportar bastam para A e C; os cartões são para B)
e, para B, com o MESMO --dataset:
    python repro/30_run_llm.py --provider ollama --model llama3 --dataset agile_limpo neo_limpo
    python repro/30_run_llm.py --provider ollama --model llama3 --dataset agile_limpo neo_limpo --cartao

Saídas:
  results/predictions/{dataset}/{within_raise,lopo_raise}/{modelo}{,_raiseonly,+raise}.csv
  results/tables/raise_repo_features.csv      características por projeto
  results/tables/contexto_estimativa.csv      comparações pareadas (A e B)
  results/tables/contexto_correlacao.csv      correlações por projeto (C)

Como ler (cuidados que valem para o TCC):
  - As características do repositório são iguais para todas as issues de um projeto. No
    within elas funcionam como um identificador do projeto e tendem a empatar com a
    mediana do projeto (foi o que a Extensão 1 mostrou no NEO). No lopo há só 7–8
    projetos de treino por fold: pouca variedade para aprender o efeito de uma linguagem.
  - Os metadados são um retrato da data da coleta (2026), anos depois das issues.
  - Na parte C há só 8 projetos no Agile e 9 no NEO: é descritivo, sem poder estatístico.
  - Predições inválidas do LLM (sem número) viram a mediana do treino, como no 90_evaluate.
"""
import argparse
import importlib.util
import json
import sys
import time
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


def load_script(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"),
                                                  config.ROOT / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- características
def repo_features() -> pd.DataFrame:
    """Uma linha por (dataset base, projeto) com as características do repositório."""
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
            langs = sorted(langs, key=lambda d: -float(d.get("percentage", 0)))
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
                "main_language": langs[0]["language"] if langs else "(desconhecida)",
                "stars": num("stars_count"), "forks": num("forks_count"), "watchers": num("watchers_count"),
                "contributors": num("contributors_count"), "releases": num("releases_count"),
                "open_issues": num("open_issues_count"), "n_topics": float(len(topics)),
                "n_languages": float(len(langs)),
                "pct_main_language": float(langs[0].get("percentage", 0)) if langs else np.nan,
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


def repo_matrix(fp: pd.DataFrame, projects: pd.Series) -> tuple:
    """Características numéricas + linguagem principal em colunas 0/1, uma linha por issue."""
    langs = sorted(fp.main_language.unique())
    tab = fp[FEATURES].copy()
    for lg in langs:
        tab[f"lang_{lg}"] = (fp.main_language == lg).astype(float)
    return tab.loc[projects].to_numpy(dtype=float), list(tab.columns)


# --------------------------------------------------------------------------- comparações
def paired(a: pd.Series, b: pd.Series, label_a: str, label_b: str, y: pd.Series, seed=0) -> dict:
    """Erro absoluto pareado: Δ = |erro b| − |erro a| (negativo = b erra menos)."""
    ea, eb = (y - a).abs(), (y - b).abs()
    d = (eb - ea).dropna()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d.to_numpy(), (5000, len(d)), replace=True).mean(1) if len(d) >= 5 else np.array([np.nan])
    p = stats.wilcoxon(d).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
    return {"comparação": f"{label_b} × {label_a}", "MAE A": ea.mean(), "MAE B": eb.mean(),
            "Δ MAE (B − A)": d.mean(), "IC95% inf": np.percentile(boots, 2.5), "IC95% sup": np.percentile(boots, 97.5),
            "p (Wilcoxon)": p, "n": len(d)}


def fit_three(ml, models, df, tr, pr, fp, ds, proto, fold, store):
    """Treina só texto / só repositório / texto + repositório num fold e guarda as predições."""
    from scipy import sparse
    from sklearn.preprocessing import StandardScaler
    y_tr = df.loc[tr, "sp"].to_numpy()
    R, _ = repo_matrix(fp, df.project)
    sc = StandardScaler().fit(R[tr])
    r_tr, r_pr = np.nan_to_num(sc.transform(R[tr])), np.nan_to_num(sc.transform(R[pr]))
    t_tr, t_pr = ml.featurize(df.loc[tr, "text_full"], df.loc[pr, "text_full"], 0)
    sets = {"": (t_tr, t_pr), "_raiseonly": (r_tr, r_pr),
            "+raise": (sparse.hstack([t_tr, sparse.csr_matrix(r_tr)]).tocsr(),
                       sparse.hstack([t_pr, sparse.csr_matrix(r_pr)]).tocsr())}
    for name in models:
        for suffix, (x_tr, x_pr) in sets.items():
            model = ml.make_model(name)
            model.fit(x_tr, y_tr)
            part = df.loc[pr, ["issue_id", "project"]].copy()
            part["y_true"] = df.loc[pr, "sp"].to_numpy()
            part["y_pred"] = np.asarray(model.predict(x_pr), dtype=float)
            part["family"], part["fold"], part["y_train_mean"] = "ML", fold, y_tr.mean()
            store.setdefault(name + suffix, []).append(part)
    # referências que não leem o texto
    base = df.loc[pr, ["issue_id", "project", "sp"]].rename(columns={"sp": "y_true"})
    if proto == "within":
        med = df[tr].groupby("project").sp.median()
        store.setdefault("ProjectMedian", []).append(base.assign(y_pred=base.project.map(med).to_numpy(), fold=fold))
    store.setdefault("Median", []).append(base.assign(y_pred=float(np.median(y_tr)), fold=fold))


def run_ml(ds: str, feats: pd.DataFrame, models: list, proto: str):
    """Devolve ({modelo: y_pred por issue}, issues avaliadas com y_true)."""
    ml = load_script("10_run_ml.py")
    df = load_merged(ds)
    fp = feats[feats.dataset == config.base_dataset(ds)].set_index("project")
    df = df[df.project.isin(fp.index)].reset_index(drop=True)
    store = {}
    t0 = time.time()
    if proto == "within":
        fit_three(ml, models, df, (df.within == "train").to_numpy(), (df.within == "test").to_numpy(),
                  fp, ds, proto, "all", store)
        ev_mask = (df.within_eval == 1)
    else:
        for p in sorted(df.project.unique()):
            pr = ((df.project == p) & (df.lopo_eval == 1)).to_numpy()
            if pr.sum() == 0:
                continue
            fit_three(ml, models, df, (df.project != p).to_numpy(), pr, fp, ds, proto, p, store)
            print(f"  [{ds}/lopo] fold {p} pronto ({time.time() - t0:.0f} s)", flush=True)
        ev_mask = (df.lopo_eval == 1)
    out = {}
    for name, parts in store.items():
        pred = pd.concat(parts)
        if name not in ("ProjectMedian", "Median"):
            save_predictions(pred, ds, f"{proto}_raise", name)
        out[name] = pred.set_index("issue_id").y_pred
    ev = df[ev_mask.to_numpy()].set_index("issue_id")
    for name in out:
        e = out[name].reindex(ev.index)
        print(f"  [{ds}/{proto}] {name:<22} MAE = {np.mean(np.abs(ev.sp - e)):.3f}  (n = {e.notna().sum()})")
    return out, ev


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
    ap.add_argument("--datasets", nargs="+", default=config.DATASETS, choices=config.ALL_DATASETS,
                    help="agile/neo (do artigo) ou agile_limpo/neo_limpo (versões limpas)")
    ap.add_argument("--protocolos", nargs="+", default=["within", "lopo"], choices=["within", "lopo"],
                    help="within = projeto conhecido; lopo = projeto novo (re-treina uma vez por projeto)")
    args = ap.parse_args()

    feats = repo_features()
    config.TABLES.mkdir(parents=True, exist_ok=True)
    feats.to_csv(config.TABLES / "raise_repo_features.csv", index=False)
    print("\nCaracterísticas por projeto (metadados do RAISE):")
    print(feats[["dataset", "project", "repo", "main_language", "stars", "forks", "contributors", "releases",
                 "n_topics", "n_languages", "repo_age_years"]].round(1).to_string(index=False))

    rows, corr = [], []
    for ds in args.datasets:
        if not (feats.dataset == config.base_dataset(ds)).any():
            continue
        fallback = load_merged(ds).query("within == 'train'").sp.median()
        for proto in args.protocolos:
            print(f"\n== {ds}: ML ({proto}) só nos projetos com repositório ==")
            preds, ev = run_ml(ds, feats, args.models, proto)
            y = ev.sp
            P = {k: v.reindex(ev.index) for k, v in preds.items()}
            ref = "ProjectMedian" if proto == "within" else "Median"
            parte = f"A. ML + repositório ({proto})"
            for name in args.models:
                for a, b in [(name, name + "+raise"), (name, name + "_raiseonly"), (ref, name),
                             (ref, name + "+raise"), (ref, name + "_raiseonly")]:
                    rows.append({"dataset": ds, "parte": parte, **paired(P[a], P[b], a, b, y)})
            if proto != "within":
                continue
            L = llm_preds(ds, args.llm)
            if L:
                idx = ev.index.intersection(L["LLM ZS1"].index).intersection(L["LLM ZS1 + cartão"].index)
                yl = y.loc[idx]
                Lf = {k: v.loc[idx].fillna(fallback) for k, v in L.items()}
                pm = P["ProjectMedian"].loc[idx]
                for a, b, pa, pb in [("LLM ZS1", "LLM ZS1 + cartão", Lf["LLM ZS1"], Lf["LLM ZS1 + cartão"]),
                                     ("ProjectMedian", "LLM ZS1", pm, Lf["LLM ZS1"]),
                                     ("ProjectMedian", "LLM ZS1 + cartão", pm, Lf["LLM ZS1 + cartão"])]:
                    rows.append({"dataset": ds, "parte": "B. LLM + cartão (within)", **paired(pa, pb, a, b, yl)})

        # C. por projeto: características × mediana de SP do projeto (todo o dataset)
        d = load_merged(ds)
        fb = feats[feats.dataset == config.base_dataset(ds)]
        sp = d[d.project.isin(fb.project)].groupby("project").sp.median().rename("sp_mediana")
        fp = fb.set_index("project").join(sp)
        for c in FEATURES:
            if fp[c].nunique() > 1:
                rho, p = stats.spearmanr(fp[c], fp.sp_mediana)
                corr.append({"dataset": ds, "característica": c, "rho (Spearman)": rho, "p": p, "n projetos": len(fp)})
        por_lingua = fp.groupby("main_language").agg(projetos=("sp_mediana", "size"), sp_mediana=("sp_mediana", "median"))
        print(f"\n[{ds}] mediana de SP por linguagem principal (descritivo):")
        print(por_lingua.to_string())

    res = pd.DataFrame(rows).set_index(["dataset", "parte", "comparação"])
    res.to_csv(config.TABLES / "contexto_estimativa.csv")
    pd.set_option("display.width", 200)
    print("\n== Comparações pareadas (Δ negativo = B erra menos que A) ==")
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
