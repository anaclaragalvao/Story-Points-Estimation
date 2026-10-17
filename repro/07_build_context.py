"""EXTENSÃO (não faz parte do artigo) — contexto do projeto para cada issue, só com o que existia antes dela.

    python repro/07_build_context.py                                     # issues de teste (within) dos projetos ligados
    python repro/07_build_context.py --ids results/improvement/selecionadas.csv
    python repro/07_build_context.py --oraculo                           # inclui também o contexto-oráculo

Antes: 05 e 06 (e a coleta no RAISE). Lê o Postgres do RAISE (ver see/raise_db.py).

Para cada issue-alvo i, com data de referência t_i e ordem de criação o_i (etapa 06):
  1. Repositório: nome e descrição (metadados do RAISE).
  2. Áreas do código mais ativas nos 12 meses antes de t_i: pastas mais alteradas em
     commits com data < t_i (contadas por commit).
  3. Até K issues parecidas do MESMO projeto criadas antes de i (ordem < o_i), por
     similaridade TF-IDF de título + descrição: título, começo da descrição e os arquivos
     que os commits delas mudaram ANTES de t_i.
  4. Áreas tocadas por essas issues parecidas (resumo do item 3).
Nunca entram: story points (de nenhuma issue), commits com data >= t_i, commits ligados à
própria issue, texto de issues criadas depois dela.

Limites conhecidos:
  - O TF-IDF de cada projeto é ajustado com o texto de todas as issues do projeto (só o
    vocabulário e o IDF, nunca rótulos). O ranking usa apenas issues anteriores.
  - No Agile, t_i é um limite superior da criação (ver 06): o contexto pode incluir
    trabalho de outras issues feito entre a criação real e t_i.
  - Os metadados (descrição, README) são um retrato da data da coleta. O README fica
    fora por padrão; --readme o inclui, marcado como tal.
  - O RAISE perde arquivos de mesmo nome num mesmo commit (ver see/raise_db.py), e merges
    não têm arquivos: issues ligadas só por "Merge branch" aparecem sem arquivos.

Oráculo (--oraculo, campo separado): arquivos que os commits da própria issue mudaram. É
informação de depois do trabalho feito. Serve só como limite superior, nunca como condição
realista de estimativa.

Saída: data/processed/context/context_{dataset}.jsonl, uma linha por issue:
  issue_id, dataset, project, ref, t_ref, contexto, similares, n_similares_com_arquivos,
  n_areas_ativas, contexto_oraculo (só com --oraculo)
O texto do contexto é em inglês, como os prompts do notebook.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config, raise_db  # noqa: E402
from see.io import load_merged  # noqa: E402
from see.text import clean_light  # noqa: E402

K_SIMILARES = 3
MIN_SIM = 0.05
JANELA_ATIVIDADE = pd.Timedelta(days=365)
N_AREAS = 8
N_ARQUIVOS = 5
MAX_CHARS = 3500


def area(path: str) -> str:
    """Pasta do arquivo, encurtada para leitura: primeira pasta/…/duas últimas."""
    parts = [p for p in str(path).split("/") if p][:-1]
    if not parts:
        return "(repository root)"
    if len(parts) > 4:
        parts = [parts[0], "…"] + parts[-2:]
    return "/".join(parts)


def short_path(path: str) -> str:
    parts = [p for p in str(path).split("/") if p]
    if len(parts) > 5:
        parts = [parts[0], "…"] + parts[-3:]
    return "/".join(parts)


def snippet(text: str, n: int) -> str:
    t = " ".join(clean_light(text or "").split())
    return t if len(t) <= n else t[:n].rsplit(" ", 1)[0] + "…"


def files_of(shas, by_sha) -> pd.DataFrame:
    parts = [by_sha[s] for s in shas if s in by_sha]
    return pd.concat(parts) if parts else pd.DataFrame(columns=["sha", "date", "path", "added_lines", "deleted_lines"])


def load_inputs(ds: str):
    ctx = config.CONTEXT_DIR
    for name in ("issue_tref.csv", "issue_commits.csv"):
        if not (ctx / name).exists():
            sys.exit(f"{ctx / name} não existe. Rode antes: python repro/06_link_commits.py")
    tref = pd.read_csv(ctx / "issue_tref.csv", dtype={"issue_id": str, "project": str})
    tref = tref[tref.dataset == ds].copy()
    tref["t_ref"] = pd.to_datetime(tref.t_ref, utc=True, format="mixed")
    links = pd.read_csv(ctx / "issue_commits.csv", dtype={"issue_id": str, "project": str})
    links = links[links.dataset == ds].copy()
    links["commit_date"] = pd.to_datetime(links.commit_date, utc=True, format="mixed")
    data = load_merged(ds)[["issue_id", "project", "title", "description", "within"]]
    issues = tref.merge(data.drop(columns="project"), on="issue_id", how="left", validate="one_to_one")
    return issues, links


def build_project(proj, iss, links, files, meta, targets, args):
    """Contextos das issues-alvo de um projeto."""
    iss = iss.sort_values("ordem").reset_index(drop=True)
    text = (iss.title.fillna("") + ". " + iss.description.fillna("")).map(clean_light)
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_df=0.8)
    X = vec.fit_transform(text)
    pos = {iid: k for k, iid in enumerate(iss.issue_id)}
    ordem = iss.ordem.to_numpy()

    # arquivos por commit e commits por issue
    files = files.sort_values("date").reset_index(drop=True)
    files["area"] = files.path.map(area)
    fdates = files.date.to_numpy(dtype="datetime64[ns]")
    by_sha = {sha: g for sha, g in files.groupby("sha")}
    shas_of = links.groupby("issue_id").sha.apply(set).to_dict()

    repo_lines = []
    for _, r in meta.iterrows():
        desc = " ".join(str(r.description or "").split())
        repo_lines.append(f"Repository: {r.repo}" + (f" - {desc}" if desc else ""))
        if args.readme and r.readme:
            repo_lines.append(f"README (as of data collection): {snippet(r.readme, 600)}")

    out = []
    for iid in targets:
        i = pos[iid]
        row = iss.iloc[i]
        t = row.t_ref
        proprios = shas_of.get(iid, set())
        lines = ["Project context (only information available before this work item was created):"] + repo_lines

        # 2. áreas ativas nos 12 meses antes de t
        n_areas = 0
        if pd.notna(t) and len(files):
            lo = np.searchsorted(fdates, (t - JANELA_ATIVIDADE).tz_localize(None).to_datetime64())
            hi = np.searchsorted(fdates, t.tz_localize(None).to_datetime64())
            w = files.iloc[lo:hi]
            w = w[~w.sha.isin(proprios)]
            top = w.drop_duplicates(["sha", "area"]).area.value_counts().head(N_AREAS)
            n_areas = len(top)
            if n_areas:
                lines.append("Most active code areas in the previous 12 months: " +
                             ", ".join(f"{a} ({n} commits)" for a, n in top.items()))

        # 3. issues parecidas anteriores
        sims = (X[i] @ X.T).toarray().ravel()
        cand = np.where((ordem < ordem[i]) & (sims >= MIN_SIM))[0]
        cand = cand[np.argsort(-sims[cand], kind="stable")][:args.k]
        similares, com_arquivos, areas_sim = [], 0, {}
        if len(cand):
            lines.append("Similar earlier work items in this project:")
        for j in cand:
            sj = iss.iloc[j]
            item = f"- {sj.ref}: {snippet(sj.title, 120).rstrip('.')}"
            d = snippet(sj.description, 200)
            if d:
                item += f". {d}"
            lines.append(item)
            fj = files_of([s for s in shas_of.get(sj.issue_id, set()) if s not in proprios], by_sha)
            if len(fj) and pd.notna(t):
                fj = fj[fj.date < t]
            if len(fj):
                com_arquivos += 1
                fj = (fj.assign(lin=fj.added_lines + fj.deleted_lines).groupby("path").lin.sum()
                        .sort_values(ascending=False))
                shown = [short_path(p) for p in fj.index[:N_ARQUIVOS]]
                extra = f" (+{len(fj) - N_ARQUIVOS} more)" if len(fj) > N_ARQUIVOS else ""
                lines.append(f"  Files changed: {', '.join(shown)}{extra}")
                for a in {area(p) for p in fj.index}:
                    areas_sim[a] = areas_sim.get(a, 0) + 1
            similares.append({"issue_id": sj.issue_id, "ref": sj.ref, "sim": round(float(sims[j]), 3)})
        if areas_sim:
            top = sorted(areas_sim.items(), key=lambda kv: (-kv[1], kv[0]))[:N_AREAS]
            lines.append("Code areas touched by these similar items: " + ", ".join(a for a, _ in top))

        contexto = "\n".join(lines)
        if len(contexto) > MAX_CHARS:
            contexto = contexto[:MAX_CHARS].rsplit("\n", 1)[0]
        rec = {"issue_id": iid, "dataset": row.dataset, "project": proj, "ref": row.ref,
               "t_ref": None if pd.isna(t) else t.isoformat(), "contexto": contexto, "similares": similares,
               "n_similares_com_arquivos": com_arquivos, "n_areas_ativas": n_areas}

        if args.oraculo:
            fo = files_of(proprios, by_sha)
            if len(fo):
                fo = (fo.assign(lin=fo.added_lines + fo.deleted_lines).groupby("path").lin.sum()
                        .sort_values(ascending=False))
                rec["contexto_oraculo"] = ("Files changed by the commits that implemented this work item: " +
                                           ", ".join(short_path(p) for p in fo.index[:10]) +
                                           (f" (+{len(fo) - 10} more)" if len(fo) > 10 else ""))
            else:
                rec["contexto_oraculo"] = ""
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=config.DATASETS, help="só um dataset")
    ap.add_argument("--ids", type=Path, help="CSV com coluna issue_id (ex.: results/improvement/selecionadas.csv)")
    ap.add_argument("--k", type=int, default=K_SIMILARES, help="número de issues parecidas")
    ap.add_argument("--oraculo", action="store_true", help="inclui os arquivos da própria issue (limite superior)")
    ap.add_argument("--readme", action="store_true", help="inclui o começo do README (retrato da coleta)")
    args = ap.parse_args()

    repos_map = json.loads(config.RAISE_REPOS.read_text(encoding="utf-8"))
    pedidos = set(pd.read_csv(args.ids, dtype={"issue_id": str}).issue_id) if args.ids else None
    conn = raise_db.connect()
    for ds in ([args.dataset] if args.dataset else config.DATASETS):
        projs = {str(p): [r.lower() for r in rs] for p, rs in repos_map.get(ds, {}).items()}
        if not projs:
            continue
        issues, links = load_inputs(ds)
        if pedidos is not None:
            alvo = issues[issues.issue_id.isin(pedidos)]
            fora = pedidos & set(load_merged(ds).issue_id) - set(alvo.issue_id)
            if fora:
                print(f"{ds}: {len(fora)} issues pedidas estão em projetos sem repositório ligado (ficam fora)")
        else:
            alvo = issues[issues.within == "test"]
        sem_data = alvo.t_ref.isna()
        if sem_data.any():
            print(f"{ds}: {int(sem_data.sum())} issues sem data de referência (t_ref) ficam fora")
        alvo = alvo[~sem_data]

        rs_all = sorted({r for v in projs.values() for r in v})
        files_all = raise_db.load_files(conn, rs_all)
        meta_all = raise_db.load_metadata(conn, rs_all)
        recs = []
        for proj, g in alvo.groupby("project"):
            rs = projs[proj]
            recs += build_project(proj, issues[issues.project == proj], links[links.project == proj],
                                  files_all[files_all.repo.isin(rs)], meta_all[meta_all.repo.isin(rs)],
                                  list(g.issue_id), args)
            print(f"  {ds}/{proj}: {len(g)} contextos", flush=True)

        path = config.CONTEXT_DIR / f"context_{ds}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        df = pd.DataFrame(recs)
        if len(df):
            print(f"{ds}: {len(df)} contextos -> {path.relative_to(config.ROOT)}")
            print(f"   com issue parecida que tem arquivos: {100 * (df.n_similares_com_arquivos > 0).mean():.0f}%"
                  f" | com áreas ativas: {100 * (df.n_areas_ativas > 0).mean():.0f}%"
                  f" | tamanho mediano: {df.contexto.str.len().median():.0f} caracteres")
            ex = df.sort_values("n_similares_com_arquivos", ascending=False).iloc[0]
            print(f"\n--- exemplo ({ex.issue_id}, {ex.ref}) ---\n{ex.contexto}\n")


if __name__ == "__main__":
    main()
