"""EXTENSÃO — metadados do issue tracker no ML (texto × metadados × texto + metadados).

Pergunta: as informações que o time registra no tracker junto com a issue (tipo, prioridade, componente,
quem abriu, quem pegou, checklist...) ajudam a estimar? Diferente das características do repositório
(31_estimativa_contexto.py), que são iguais para o projeto inteiro, estas variam de issue para issue.

NEO (GitLab): lê data/raw/neo_json/<projeto>.json (o mesmo JSON da API que o analise_jev.ipynb baixa).
Agile (Jira): lê data/processed/agile_tawos_meta.csv, gerado pelo 09_tawos_metadados.py a partir do dataset
TAWOS (Tawosi et al., MSR 2022), com os campos como estavam na data da estimativa. Só entram as issues
encontradas no TAWOS (12 dos 16 projetos do agile_limpo); texto e referências são recalculados nelas.

Vazamento (o que fica de fora): tudo que descreve o ANDAMENTO da issue ou só existe depois da estimativa:
estado, datas de fechamento, comentários, MRs, tempo gasto, votos, issues bloqueadas, a identidade da
sprint/milestone (as issues de uma mesma sprint são estimadas juntas, e o split within não é cronológico)
e os labels de fluxo (status, workflow, flow, qa, deployed, sprint, gitbot:..., "auto closed", ...).
Limitação: o JSON é um retrato do fim da vida da issue; labels podem ter mudado depois da estimativa.

Grupos de metadados (cada um pode ser removido para medir o que ele acrescenta):
    rotulos      labels descritivos: tipo, prioridade, severidade, produto/componente, time, plataforma, origem...
    estrutura    nº de itens de checklist na descrição, tem épico, tem milestone, tem prazo, confidencial
    autor        quem abriu a issue (só autores com >= MIN_DF issues no treino)
    responsavel  quem foi designado (fica num conjunto à parte: pode ter sido definido depois da estimativa)

Conjuntos treinados (mesmo modelo, mesmo split within, avaliados no within_eval):
    texto                 TF-IDF do título + descrição (como no 10_run_ml.py)
    meta                  só os metadados (rotulos + estrutura + autor)
    texto+meta            os dois
    texto+meta+resp       + responsável
Referências sem ML: mediana do projeto e mediana do projeto × tipo da issue (baseline condicional).

Saídas:
    results/predictions/<ds>/within_meta/<modelo><sufixo>.csv
    results/tables/metadados_estimativa.csv      MAE e comparações pareadas (Wilcoxon + IC95% bootstrap + Holm)
    results/tables/metadados_importancia.csv     quanto o MAE piora ao embaralhar cada grupo (permutação)
    results/tables/metadados_cobertura.csv       quantas issues têm cada grupo / labels descartados

Exemplo:
    python repro/32_ml_metadados.py --datasets neo_limpo --models KNN XGBoost
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse, stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged, save_predictions  # noqa: E402
from see.metadados import tipo_da_issue  # noqa: E402  (mesma regra usada pelo 30 e pelo notebook)

MIN_DF = 5   # um token de metadado precisa aparecer em pelo menos 5 issues do treino

# prefixos de labels com escopo ("prefixo::valor") que DESCREVEM a issue
PREFIXOS_OK = {"type", "priority", "severity", "impact", "kind", "product", "devops", "group", "section", "team",
               "dev-team", "platform", "dex-group", "dex", "backlog", "maintenance", "service", "feature",
               "category", "valuestream", "source", "pajamas"}
# prefixos que descrevem o ANDAMENTO (vazamento)
PREFIXOS_FLUXO = {"status", "workflow", "workflow-infra", "dex-status", "flow", "qa", "deployed", "sprint"}
# labels simples que descrevem andamento, automação ou o próprio peso (vazamento)
FLUXO_SIMPLES = re.compile(r"status|workflow|progress|closed|moved|missed|deployed|gitbot|auto[ -]?updated|"
                           r"unscheduled|stretch|deliverable|weight|sprint|\bdone\b|verified|review|blocked|"
                           r"doing|ready|to do|todo|triage|wip|merged|released|fixed|duplicate|wontfix|won't",
                           re.I)


def metadados_neo(ids: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uma linha por issue: listas de tokens por grupo + números. Também devolve os labels descartados."""
    pasta = config.DATA_RAW / "neo_json"
    if not pasta.exists():
        sys.exit(f"Falta {pasta}. Rode a seção 5 do analise_jev.ipynb (ela baixa os JSON) ou o 06_link_commits.py.")
    rows, descartados = [], {}
    for f in sorted(pasta.glob("*.json")):
        for it in json.loads(f.read_text(encoding="utf-8")):
            iid = f"{f.stem}-{it['id']}"
            if iid not in ids:
                continue
            labs = [l.strip() for l in (it.get("labels") or [])]
            rot = []
            for l in labs:
                low = l.lower()
                if "::" in low:
                    pre = low.split("::")[0].strip()
                    if pre in PREFIXOS_OK:
                        rot.append("lab:" + low)
                    else:
                        descartados[low] = descartados.get(low, 0) + 1
                elif FLUXO_SIMPLES.search(low):
                    descartados[low] = descartados.get(low, 0) + 1
                else:
                    rot.append("lab:" + low)
            tipo = tipo_da_issue(labs)
            rot.append(f"tipo:{tipo}")
            tasks = (it.get("task_completion_status") or {}).get("count") or 0
            estrutura = [t for t, ok in [("tem_epico", bool(it.get("epic") or it.get("epic_iid"))),
                                         ("tem_milestone", bool(it.get("milestone"))),
                                         ("tem_prazo", bool(it.get("due_date"))),
                                         ("confidencial", bool(it.get("confidential"))),
                                         ("tem_checklist", tasks > 0),
                                         (f"issue_type:{it.get('issue_type')}", it.get("issue_type") not in (None, "issue"))]
                         if ok]
            autor = [f"autor:{f.stem}:{(it.get('author') or {}).get('id')}"] if it.get("author") else []
            resp = [f"resp:{f.stem}:{a.get('id')}" for a in (it.get("assignees") or [])]
            rows.append({"issue_id": iid, "tipo": tipo, "rotulos": rot, "estrutura": estrutura, "autor": autor,
                         "responsavel": resp, "n_tasks": tasks})
    meta = pd.DataFrame(rows)
    desc = (pd.Series(descartados, name="issues").sort_values(ascending=False).rename_axis("label descartado")
            .reset_index())
    return meta, desc


def metadados_agile(ids: set) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Agile via TAWOS: tipo, prioridade, componentes e labels na data da estimativa; reporter; responsável."""
    path = config.DATA_PROCESSED / "agile_tawos_meta.csv"
    if not path.exists():
        return None, None
    t = pd.read_csv(path, dtype={"issue_id": str, "project": str, "reporter": str, "responsavel": str}).fillna("")
    t = t[t.issue_id.isin(ids) & ((t.tipo != "") | (t.sp_tawos != ""))]
    rows = []
    for r in t.itertuples():
        tipo = str(r.tipo).strip().lower() or "sem_tipo"
        rot = [f"tipo:{tipo}"]
        if r.prioridade:
            rot.append(f"prio:{str(r.prioridade).strip().lower()}")
        comps = [c.strip().lower() for c in str(r.componentes).split("|") if c.strip()]
        rot += [f"comp:{r.project}:{c}" for c in comps]
        rot += [f"lab:{l.lower()}" for l in str(r.labels).split() if l and not FLUXO_SIMPLES.search(l)]
        estrutura = ["tem_componente"] if comps else []
        autor = [f"autor:{r.project}:{r.reporter}"] if r.reporter else []
        resp = [f"resp:{r.project}:{r.responsavel}"] if r.responsavel else []
        rows.append({"issue_id": r.issue_id, "tipo": tipo, "rotulos": rot, "estrutura": estrutura, "autor": autor,
                     "responsavel": resp, "n_tasks": 0})
    return pd.DataFrame(rows), pd.DataFrame(columns=["label descartado", "issues"])


GRUPOS_META = ["rotulos", "estrutura", "autor"]
CONJUNTOS = {"": ["texto"], "_metaonly": GRUPOS_META, "+meta": ["texto", *GRUPOS_META],
             "+meta+resp": ["texto", *GRUPOS_META, "responsavel"]}
NOME_CONJUNTO = {"": "texto", "_metaonly": "meta", "+meta": "texto+meta", "+meta+resp": "texto+meta+resp"}


def blocos(df, tr, pr, meta):
    """Matrizes por grupo, ajustadas só no treino. Cada bloco de metadados é normalizado (norma L2 por linha),
    como o TF-IDF, para nenhum bloco dominar a distância do KNN."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.preprocessing import StandardScaler, normalize
    from importlib.util import module_from_spec, spec_from_file_location
    spec = spec_from_file_location("run_ml", config.ROOT / "10_run_ml.py")
    run_ml = module_from_spec(spec); spec.loader.exec_module(run_ml)
    out = {}
    t_tr, t_pr = run_ml.featurize(df.loc[tr, "text_full"], df.loc[pr, "text_full"], 0)
    out["texto"] = (t_tr, t_pr)
    m = meta.set_index("issue_id")
    ids_tr, ids_pr = df.loc[tr, "issue_id"], df.loc[pr, "issue_id"]
    for g in ["rotulos", "estrutura", "autor", "responsavel"]:
        vec = CountVectorizer(analyzer=lambda x: x, binary=True, min_df=MIN_DF)
        try:
            a = vec.fit_transform(m.loc[ids_tr, g])
        except ValueError:                       # nenhum token com frequência suficiente
            out[g] = (sparse.csr_matrix((len(ids_tr), 0)), sparse.csr_matrix((len(ids_pr), 0)))
            continue
        b = vec.transform(m.loc[ids_pr, g])
        if g == "estrutura":                     # + nº de itens do checklist (log), padronizado no treino
            sc = StandardScaler().fit(np.log1p(m.loc[ids_tr, ["n_tasks"]].to_numpy(float)))
            a = sparse.hstack([a, sparse.csr_matrix(sc.transform(np.log1p(m.loc[ids_tr, ["n_tasks"]].to_numpy(float))))])
            b = sparse.hstack([b, sparse.csr_matrix(sc.transform(np.log1p(m.loc[ids_pr, ["n_tasks"]].to_numpy(float))))])
        out[g] = (normalize(a.tocsr()), normalize(b.tocsr()))
    return out, run_ml


def junta(bl, grupos, lado):
    i = 0 if lado == "tr" else 1
    mats = [bl[g][i] for g in grupos if bl[g][i].shape[1]]
    return sparse.hstack(mats).tocsr()


def pareado(ea, eb, seed=0):
    d = (eb - ea).dropna()
    rng = np.random.default_rng(seed)
    boots = rng.choice(d.to_numpy(), (5000, len(d)), replace=True).mean(1) if len(d) >= 5 else np.array([np.nan])
    p = stats.wilcoxon(d).pvalue if len(d) >= 5 and (d != 0).any() else np.nan
    return {"MAE A": ea.mean(), "MAE B": eb.mean(), "Δ MAE (B − A)": d.mean(),
            "IC95% inf": np.percentile(boots, 2.5), "IC95% sup": np.percentile(boots, 97.5), "p (Wilcoxon)": p,
            "n": len(d)}


def holm(p):
    p = pd.Series(p, dtype=float); ok = p.dropna().sort_values(); m = len(ok)
    adj = pd.Series(np.nan, index=p.index); run = 0.0
    for k, (i, v) in enumerate(ok.items()):
        run = max(run, min(1.0, (m - k) * v)); adj[i] = run
    return adj


def roda(ds, models, permutacoes):
    df = load_merged(ds).reset_index(drop=True)
    if config.base_dataset(ds) == "neo":
        meta, desc = metadados_neo(set(df.issue_id))
    else:
        meta, desc = metadados_agile(set(df.issue_id))
        if meta is None:
            print(f"[{ds}] pulado: falta data/processed/agile_tawos_meta.csv (rode antes o 09_tawos_metadados.py).")
            return [], [], []
        antes = len(df)
        df = df[df.issue_id.isin(meta.issue_id)].reset_index(drop=True)   # só issues encontradas no TAWOS
        print(f"[{ds}] {len(df)} de {antes} issues com metadados do TAWOS; projetos: {sorted(df.project.unique())}")
    falta = set(df.issue_id) - set(meta.issue_id)
    if falta:
        print(f"[{ds}] {len(falta)} issues sem JSON: ficam com metadados vazios")
        meta = pd.concat([meta, pd.DataFrame({"issue_id": sorted(falta), "tipo": "sem_tipo", "rotulos": [[]] * len(falta),
                                              "estrutura": [[]] * len(falta), "autor": [[]] * len(falta),
                                              "responsavel": [[]] * len(falta), "n_tasks": 0})])
    tr = df.index[df.within == "train"]
    pr = df.index[df.within_eval == 1]
    y_tr, y = df.loc[tr, "sp"].to_numpy(float), df.loc[pr, "sp"].astype(float).reset_index(drop=True)
    t0 = time.time()
    bl, run_ml = blocos(df, tr, pr, meta)
    print(f"[{ds}] treino {len(tr)}, avaliação {len(pr)}; colunas: " +
          ", ".join(f"{g} {bl[g][0].shape[1]}" for g in bl) + f" ({time.time() - t0:.0f}s)")

    m = meta.set_index("issue_id")
    cob = [{"dataset": ds, "item": f"grupo {g}", "valor": 100 * (m.loc[df.issue_id, g].str.len() > 0).mean(),
            "unidade": "% das issues com algum token"} for g in ["rotulos", "estrutura", "autor", "responsavel"]]
    cob += [{"dataset": ds, "item": f"tipo = {t}", "valor": 100 * v, "unidade": "% das issues"}
            for t, v in m.loc[df.issue_id, "tipo"].value_counts(normalize=True).items()]
    cob += [{"dataset": ds, "item": f"descartado: {r['label descartado']}", "valor": r["issues"],
             "unidade": "nº de issues (label de andamento/vazamento)"} for _, r in desc.iterrows()]

    preds = {}
    proj_tr = df.loc[tr, "project"]
    med_proj = pd.Series(y_tr, index=proj_tr.values).groupby(level=0).median()
    preds["ProjectMedian"] = df.loc[pr, "project"].map(med_proj).fillna(np.median(y_tr)).to_numpy()
    tipo = df.issue_id.map(m.tipo)
    med_pt = pd.Series(y_tr).groupby([proj_tr.values, tipo.loc[tr].values]).median()
    n_pt = pd.Series(y_tr).groupby([proj_tr.values, tipo.loc[tr].values]).size()
    chave = list(zip(df.loc[pr, "project"], tipo.loc[pr]))
    preds["ProjectMedian×tipo"] = np.array([med_pt[k] if k in med_pt.index and n_pt[k] >= MIN_DF else med_proj.get(k[0], np.median(y_tr))
                                            for k in chave])
    imp = []
    for name in models:
        for suf, grupos in CONJUNTOS.items():
            x_tr, x_pr = junta(bl, grupos, "tr"), junta(bl, grupos, "pr")
            if x_tr.shape[1] == 0:
                continue
            t1 = time.time()
            model = run_ml.make_model(name); model.fit(x_tr, y_tr)
            yp = np.asarray(model.predict(x_pr), dtype=float)
            preds[f"{name}{suf}"] = yp
            out = df.loc[pr, ["issue_id", "project"]].copy()
            out["y_true"], out["y_pred"], out["family"], out["features"] = y.values, yp, "ML", NOME_CONJUNTO[suf]
            save_predictions(out, ds, "within_meta", f"{name}{suf}")
            print(f"   {name:8s} {NOME_CONJUNTO[suf]:16s} MAE {np.abs(y.values - yp).mean():.3f} ({time.time() - t1:.0f}s)")
            if suf == "+meta+resp" and permutacoes:   # importância por permutação de cada grupo (no modelo mais completo)
                base = np.abs(y.values - yp).mean()
                rng = np.random.default_rng(config.SEED)
                for g in grupos:
                    deltas = []
                    for _ in range(permutacoes):
                        perm = rng.permutation(x_pr.shape[0])
                        bl_p = {k: (v[0], v[1][perm] if k == g else v[1]) for k, v in bl.items()}
                        deltas.append(np.abs(y.values - model.predict(junta(bl_p, grupos, "pr"))).mean() - base)
                    imp.append({"dataset": ds, "modelo": name, "grupo": g, "Δ MAE ao embaralhar (média)": np.mean(deltas),
                                "desvio": np.std(deltas), "repetições": permutacoes})

    E = pd.DataFrame({k: np.abs(y.values - v) for k, v in preds.items()})
    rows = [{"dataset": ds, "tipo": "MAE", "comparação": k, "MAE A": E[k].mean(), "MdAE": E[k].median(), "n": len(E)}
            for k in E]
    fam = []
    for a, b in [("ProjectMedian", "ProjectMedian×tipo")]:
        fam.append({"dataset": ds, "tipo": "pareado", "comparação": f"{b} × {a}", **pareado(E[a], E[b])})
    for name in models:
        pares = [(f"{name}", f"{name}+meta", "texto → texto+meta"),
                 (f"{name}", f"{name}_metaonly", "texto → só meta"),
                 (f"{name}+meta", f"{name}+meta+resp", "+ responsável"),
                 ("ProjectMedian", f"{name}+meta", "mediana do projeto → texto+meta"),
                 ("ProjectMedian×tipo", f"{name}+meta", "mediana projeto×tipo → texto+meta")]
        for a, b, desc_ in pares:
            if a in E and b in E:
                fam.append({"dataset": ds, "tipo": "pareado", "comparação": f"{name}: {desc_}", **pareado(E[a], E[b])})
    fam = pd.DataFrame(fam)
    fam["p (Holm)"] = holm(fam["p (Wilcoxon)"]).values
    return rows + fam.to_dict("records"), imp, cob


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", default=["neo_limpo"], choices=config.ALL_DATASETS)
    ap.add_argument("--models", nargs="+", default=["KNN", "XGBoost"],
                    help="modelos do 10_run_ml.py (SVR e RandomForest são mais lentos)")
    ap.add_argument("--permutacoes", type=int, default=5, help="repetições da importância por permutação (0 = pula)")
    args = ap.parse_args()
    res, imp, cob = [], [], []
    for ds in args.datasets:
        r, i, c = roda(ds, args.models, args.permutacoes)
        res += r; imp += i; cob += c
    if not res:
        return
    config.TABLES.mkdir(parents=True, exist_ok=True)
    R = pd.DataFrame(res); R.to_csv(config.TABLES / "metadados_estimativa.csv", index=False)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 200, "display.max_rows", 200):
        print("\n" + R.drop(columns=["tipo"]).to_string(index=False))
        if imp:
            I = pd.DataFrame(imp); I.to_csv(config.TABLES / "metadados_importancia.csv", index=False)
            print("\nImportância por permutação (quanto o MAE PIORA ao embaralhar o grupo; maior = mais importante):")
            print(I.to_string(index=False))
    pd.DataFrame(cob).to_csv(config.TABLES / "metadados_cobertura.csv", index=False)
    print("\nTabelas em results/tables/metadados_*.csv; predições em results/predictions/<ds>/within_meta/")


if __name__ == "__main__":
    main()
