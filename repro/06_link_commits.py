"""EXTENSÃO (não faz parte do artigo) — liga cada issue aos commits que a citam.

O RAISE (github.com/aisepucrio/raise) entra só como extrator: ele coleta os commits
dos repositórios de repro/raise_repos.json e grava no Postgres dele. Esta etapa lê
esse banco e descobre quais commits falam de quais issues.

    python repro/06_link_commits.py --coletar                 # 1. pede ao RAISE a coleta, em janelas de tempo
    python repro/06_link_commits.py --clones C:/clones_tcc    # 2. liga issues e commits (lê o Postgres do RAISE)

--clones (recomendado para o NEO): pasta com um clone de cada repositório, só com o
histórico (git clone --bare --filter=tree:0 https://github.com/<owner>/<repo>). No NEO,
96% das issues ligadas só aparecem na mensagem do merge da MR ("Closes #123"); o PyDriller
não lista arquivos de merges e o RAISE não guarda os pais dos commits. Com o clone, os
commits trazidos por cada merge ligado (git rev-list M ^primeiro_pai) também são ligados,
com regra "<regra>+merge". Os arquivos continuam vindo do RAISE.

Antes: python repro/05_agile_issuekeys.py e o RAISE rodando (docker compose up, no repo dele).
Driver do banco: pip install -r repro/requirements-raise.txt. Senha: ver see/raise_db.py.

Por que janelas: a tarefa de commits do RAISE guarda todos os commits coletados (com
diff) no resultado da tarefa, de uma vez. O moodle tem ~210 mil commits; coletar o
histórico inteiro num pedido só pode estourar a memória. --coletar manda um pedido
por repositório e janela (6 meses por padrão), espera cada um terminar e anota o que
já foi feito em data/processed/context/raise_coleta.jsonl (dá para interromper e
retomar). --planejar só grava os pedidos em JSON, para mandar pela interface do RAISE.

Regras de ligação (a coluna `regra` registra qual disparou):
  Agile  chave    a mensagem cita a chave Jira do projeto (MDL-1626).
  NEO    caminho  "minds/front#123" (caminho completo do projeto no GitLab)
         fecha    "Closes/Fixes/Resolves/Implements/Relates to/Refs/See #123" sem caminho
                  ou com o caminho do próprio projeto
         branch   "Merge branch '123-...'" (o GitLab nomeia a branch de uma issue <iid>-<título>)
  No NEO só valem commits entre a criação da issue (-1 dia) e o fechamento (+90 dias).
  Sem essa janela, um "#123" de outro projeto, ou de um PR do GitHub depois que o
  Meltano migrou para lá, viraria ligação falsa.

Saídas, em data/processed/context/:
  issue_commits.csv  issue_id, dataset, project, repo, sha, commit_date, regra
  issue_tref.csv     issue_id, dataset, project, ref, ordem, t_ref, t_ref_fonte,
                     n_commits, primeiro_commit, ultimo_commit
    ref    como a issue é citada (MDL-1626 no Agile, #123 no NEO)
    ordem  ordem de criação dentro do projeto: número da chave (Agile) ou data (NEO)
    t_ref  data de referência; só o que é anterior a ela pode entrar no contexto (etapa 07)
           NEO: data de criação, exata.
           Agile: o arquivo não tem datas. As chaves do Jira crescem com a criação, então a
           issue k foi criada antes de qualquer j > k do mesmo projeto. Usamos
           t_ref(k) = menor "primeiro commit" entre as issues j >= k. É um limite SUPERIOR da
           criação: entre a criação real e t_ref pode haver trabalho de outras issues
           (registrar como ameaça à validade). Os commits da própria issue ficam sempre fora.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config, raise_db  # noqa: E402
from see.io import load_processed  # noqa: E402

NEO_JANELA = (pd.Timedelta(days=1), pd.Timedelta(days=90))   # antes da criação, depois do fechamento
AGILE_PERIODO = ("2002-01-01", "2017-01-01")   # os dados do Choetkiertikul foram minerados em 2016
RAISE_API = "http://localhost:8000/api"

FECHA = re.compile(r"(?i)\b(?:clos(?:e|es|ed|ing)|fix(?:es|ed|ing)?|resolv(?:e|es|ed|ing)|"
                   r"implement(?:s|ed|ing)?|relate[sd]?\s+to|refs?|see|issue)\b\s*:?\s*"
                   r"([\w.-]+/[\w./-]+)?#(\d+)\b")
BRANCH = re.compile(r"Merge branch '(?:[\w./-]*/)?(\d+)-")


# --------------------------------------------------------------------------- issues
def load_repos() -> dict:
    m = json.loads(config.RAISE_REPOS.read_text(encoding="utf-8"))
    return {ds: {str(p): [r.lower() for r in rs] for p, rs in m.get(ds, {}).items()} for ds in config.DATASETS}


def agile_issues(projects) -> pd.DataFrame:
    path = config.DATA_PROCESSED / "agile_issuekeys.csv"
    if not path.exists():
        sys.exit(f"{path} não existe. Rode antes: python repro/05_agile_issuekeys.py")
    keys = pd.read_csv(path, dtype={"issue_id": str, "project": str})
    df = keys[keys.project.isin(projects)].copy()
    df["dataset"] = "agile"
    df["ref"] = df.issuekey
    df["ordem"] = df.key_num
    return df[["issue_id", "dataset", "project", "ref", "ordem", "key_num"]]


def neo_issues(projects) -> pd.DataFrame:
    """Issues do NEO com iid, caminho do projeto e datas, a partir dos JSON da API do GitLab."""
    data = load_processed("neo")[["issue_id", "project", "created"]]
    data = data[data.project.isin(projects)]
    rows = []
    for pid in projects:
        path = config.DATA_RAW / "neo_json" / f"{pid}.json"
        if not path.exists():   # mesmo arquivo que a seção 5 do analise_jev.ipynb baixa
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/{config.NEO_REPO}/{config.NEO_COMMIT}/util/json/{pid}.json"
            print("baixando", url)
            urllib.request.urlretrieve(url, path)
        for it in json.loads(path.read_text(encoding="utf-8")):
            rows.append({"issue_id": f"{pid}-{it['id']}", "iid": int(it["iid"]),
                         "caminho": it["references"]["full"].split("#")[0].lower(),
                         "closed_at": it.get("closed_at")})
    df = data.merge(pd.DataFrame(rows), on="issue_id", how="left", validate="one_to_one")
    assert df.iid.notna().all(), "issues do NEO sem iid no JSON do GitLab"
    df["dataset"] = "neo"
    df["created"] = pd.to_datetime(df.created, utc=True, format="mixed")
    # no JSON do neodataset as datas vêm como epoch em milissegundos
    df["closed_at"] = pd.to_datetime(pd.to_numeric(df.closed_at, errors="coerce"), unit="ms", utc=True)
    df["ref"] = "#" + df.iid.astype(int).astype(str)
    df["ordem"] = df.groupby("project").created.rank(method="first").astype(int)
    return df[["issue_id", "dataset", "project", "ref", "ordem", "iid", "caminho", "created", "closed_at"]]


# --------------------------------------------------------------------------- ligação
def link_agile(issues: pd.DataFrame, commits: pd.DataFrame, repos: dict) -> pd.DataFrame:
    out = []
    for proj, rs in repos.items():
        c = commits[commits.repo.isin(rs)]
        hits = c.message.str.findall(rf"(?<![A-Za-z0-9]){re.escape(proj)}-(\d+)\b").explode().dropna()
        if hits.empty:
            continue
        h = c.loc[hits.index, ["sha", "repo", "date"]].assign(key_num=hits.astype(int).values)
        iss = issues[issues.project == proj][["issue_id", "dataset", "project", "key_num"]]
        out.append(h.merge(iss, on="key_num").assign(regra="chave"))
    if not out:
        return pd.DataFrame(columns=["issue_id", "dataset", "project", "repo", "sha", "commit_date", "regra"])
    df = pd.concat(out, ignore_index=True).rename(columns={"date": "commit_date"})
    return df[["issue_id", "dataset", "project", "repo", "sha", "commit_date", "regra"]]


def link_neo(issues: pd.DataFrame, commits: pd.DataFrame, repos: dict) -> pd.DataFrame:
    prioridade = {"caminho": 0, "fecha": 1, "branch": 2}
    out = []
    for proj, rs in repos.items():
        iss = issues[issues.project == proj]
        if iss.empty:
            continue
        caminho = iss.caminho.iloc[0]
        cam_re = re.compile(rf"(?i){re.escape(caminho)}#(\d+)\b")
        c = commits[commits.repo.isin(rs)]
        hits = []
        for sha, repo, date, msg in zip(c.sha, c.repo, c.date, c.message):
            for iid in cam_re.findall(msg):
                hits.append((sha, repo, date, int(iid), "caminho"))
            for path, iid in FECHA.findall(msg):
                if not path or path.lower() == caminho:
                    hits.append((sha, repo, date, int(iid), "fecha"))
            for iid in BRANCH.findall(msg):
                hits.append((sha, repo, date, int(iid), "branch"))
        if not hits:
            continue
        h = pd.DataFrame(hits, columns=["sha", "repo", "commit_date", "iid", "regra"])
        h = (h.assign(p=h.regra.map(prioridade)).sort_values("p")
              .drop_duplicates(["sha", "iid"]).drop(columns="p"))
        h = h.merge(iss[["issue_id", "dataset", "project", "iid", "created", "closed_at"]], on="iid")
        fim = h.closed_at.fillna(pd.Timestamp.now(tz="UTC")) + NEO_JANELA[1]
        h = h[(h.commit_date >= h.created - NEO_JANELA[0]) & (h.commit_date <= fim)]
        out.append(h)
    if not out:
        return pd.DataFrame(columns=["issue_id", "dataset", "project", "repo", "sha", "commit_date", "regra"])
    return pd.concat(out, ignore_index=True)[["issue_id", "dataset", "project", "repo", "sha", "commit_date", "regra"]]


def _clone_dir(clones: Path, repo: str):
    owner, name = repo.split("/")
    for cand in (clones / name, clones / f"{owner}_{name}", clones / owner / name,
                 clones / f"{name}.git", clones / f"{owner}_{name}.git"):
        if cand.exists():
            return cand
    lower = {p.name.lower(): p for p in clones.iterdir()}
    for key in (name, f"{owner}_{name}", f"{name}.git", f"{owner}_{name}.git"):
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def expand_merges(links: pd.DataFrame, commits: pd.DataFrame, clones: Path, issues: pd.DataFrame) -> pd.DataFrame:
    """Liga também os commits trazidos por cada merge ligado (git rev-list M ^primeiro_pai).

    O PyDriller não lista arquivos de merges e o RAISE não guarda os pais dos commits, então
    um "Closes #123" na mensagem do merge de uma MR não diria quais arquivos a issue mexeu.
    A topologia vem de um clone local (basta `git clone --bare --filter=tree:0`)."""
    import subprocess
    merges = links[links.sha.map(commits.set_index("sha")["merge"]).fillna(False).astype(bool)]
    if merges.empty:
        return links
    known = set(commits.sha)
    extra = []
    for repo, g in merges.groupby("repo"):
        d = _clone_dir(clones, repo)
        if d is None:
            print(f"AVISO: clone de {repo} não encontrado em {clones}; merges sem expansão")
            continue
        for sha in g.sha.unique():
            par = subprocess.run(["git", "-C", str(d), "rev-parse", f"{sha}^1"], capture_output=True, text=True)
            if par.returncode:
                continue
            rl = subprocess.run(["git", "-C", str(d), "rev-list", "--no-merges", sha, f"^{par.stdout.strip()}"],
                                capture_output=True, text=True).stdout.split()
            if 0 < len(rl) <= 200:   # merges enormes (integração de branches longas) ficam de fora
                for row in g[g.sha == sha].itertuples():
                    extra += [(row.issue_id, row.dataset, row.project, repo, s, f"{row.regra}+merge")
                              for s in rl if s in known]
    if not extra:
        return links
    ex = pd.DataFrame(extra, columns=["issue_id", "dataset", "project", "repo", "sha", "regra"])
    ex = ex.merge(commits[["sha", "date"]].rename(columns={"date": "commit_date"}), on="sha")
    if "created" in issues:   # mesma janela de datas do NEO
        ex = ex.merge(issues[["issue_id", "created", "closed_at"]], on="issue_id")
        fim = ex.closed_at.fillna(pd.Timestamp.now(tz="UTC")) + NEO_JANELA[1]
        ex = ex[(ex.commit_date >= ex.created - NEO_JANELA[0]) & (ex.commit_date <= fim)]
    out = pd.concat([links, ex[links.columns]], ignore_index=True)
    return out.drop_duplicates(["issue_id", "sha"])


def reference_dates(issues: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    g = links.groupby("issue_id").agg(n_commits=("sha", "nunique"), primeiro_commit=("commit_date", "min"),
                                      ultimo_commit=("commit_date", "max"))
    df = issues.merge(g, left_on="issue_id", right_index=True, how="left")
    df["n_commits"] = df.n_commits.fillna(0).astype(int)
    for col in ("primeiro_commit", "ultimo_commit"):
        df[col] = pd.to_datetime(df[col], utc=True)
    if df.dataset.iloc[0] == "neo":
        df["t_ref"] = df.created
        df["t_ref_fonte"] = "criacao"
    else:
        big = np.iinfo("int64").max
        parts = []
        for _, p in df.groupby("project"):
            p = p.sort_values("ordem", ascending=False)
            # menor primeiro commit entre as issues de ordem >= k (mínimo acumulado de trás para frente)
            ns = p.primeiro_commit.dt.tz_localize(None).to_numpy(dtype="datetime64[ns]").view("int64").copy()
            ns[p.primeiro_commit.isna().to_numpy()] = big
            cm = np.minimum.accumulate(ns)
            t = pd.Series(pd.to_datetime(np.where(cm == big, 0, cm), utc=True), index=p.index)
            t[cm == big] = pd.NaT
            parts.append(p.assign(t_ref=t))
        df = pd.concat(parts)
        df["t_ref_fonte"] = np.where(df.t_ref.notna(), "limite_superior_commits", "sem_data")
    cols = ["issue_id", "dataset", "project", "ref", "ordem", "t_ref", "t_ref_fonte",
            "n_commits", "primeiro_commit", "ultimo_commit"]
    return df[cols].sort_values(["project", "ordem"])


# --------------------------------------------------------------------------- coleta no RAISE
def _utc(x) -> pd.Timestamp:
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def janelas(inicio, fim, meses):
    inicio = _utc(inicio)
    inicio = pd.Timestamp(year=inicio.year, month=inicio.month, day=1, tz="UTC")
    fim = _utc(fim)
    while inicio < fim:
        prox = min(inicio + pd.DateOffset(months=meses), fim)
        yield inicio, prox - pd.Timedelta(seconds=1)
        inicio = prox


def plan_collection(repos: dict, meses: int) -> list:
    pedidos = []
    for ds, projs in repos.items():
        if not projs:
            continue
        if ds == "neo":
            iss = neo_issues(list(projs))
            per = iss.groupby("project").agg(ini=("created", "min"), fim=("closed_at", "max"), fim2=("created", "max"))
        for proj, rs in projs.items():
            if ds == "neo":
                ini = per.loc[proj, "ini"] - pd.Timedelta(days=365)
                fim = max(per.loc[proj, "fim"] if pd.notna(per.loc[proj, "fim"]) else per.loc[proj, "fim2"],
                          per.loc[proj, "fim2"]) + pd.Timedelta(days=120)
            else:
                ini, fim = AGILE_PERIODO
            for repo in rs:
                for a, b in janelas(ini, fim, meses):
                    pedidos.append({"dataset": ds, "project": proj, "body": {
                        "targets": [repo], "collect_types": ["commits"],
                        "start_date": a.strftime("%Y-%m-%dT%H:%M:%SZ"), "end_date": b.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "filters": {}, "options": {}}})
    # metadados (descrição e README), um pedido por repositório
    for repo in sorted({r for projs in repos.values() for rs in projs.values() for r in rs}):
        pedidos.append({"dataset": "-", "project": "-", "body": {
            "targets": [repo], "collect_types": ["metadata"], "start_date": None, "end_date": None,
            "filters": {}, "options": {}}})
    return pedidos


def _http(method, url, body=None):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def run_collection(pedidos: list, api: str):
    log = config.CONTEXT_DIR / "raise_coleta.jsonl"
    feitos = set()
    if log.exists():
        for line in log.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            if r["status"] == "SUCCESS":
                feitos.add(json.dumps(r["body"], sort_keys=True))
    todo = [p for p in pedidos if json.dumps(p["body"], sort_keys=True) not in feitos]
    print(f"{len(pedidos) - len(todo)} pedidos já concluídos, {len(todo)} a fazer")
    for n, p in enumerate(todo, 1):
        b = p["body"]
        print(f"[{n}/{len(todo)}] {b['targets'][0]} {b['collect_types'][0]} {b['start_date']} -> {b['end_date']}", flush=True)
        t0 = time.time()
        try:
            resp = _http("POST", f"{api}/github/collect/", b)
        except Exception as e:   # RAISE fora do ar ou pedido recusado: anota e segue
            print(f"    erro ao pedir a coleta: {e}")
            resp = {}
        task_ids = [t["task_id"] for r in resp.get("results", []) for t in r.get("tasks", [])]
        status = "FAILURE" if not task_ids else "PENDING"
        while task_ids and status not in ("SUCCESS", "FAILURE", "REVOKED"):
            time.sleep(10)
            try:
                status = _http("GET", f"{api}/jobs/tasks/{task_ids[0]}/").get("status", "PENDING")
            except Exception as e:
                print(f"    erro ao consultar a tarefa ({e}); tentando de novo")
        print(f"    {status} em {time.time() - t0:.0f} s")
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps({"body": b, "task_ids": task_ids, "status": status,
                                "segundos": round(time.time() - t0)}) + "\n")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planejar", action="store_true", help="só grava os pedidos de coleta em JSON")
    ap.add_argument("--coletar", action="store_true", help="manda os pedidos ao RAISE, um por vez, e espera")
    ap.add_argument("--meses", type=int, default=6, help="tamanho da janela de coleta (meses)")
    ap.add_argument("--api", default=RAISE_API, help="URL da API do RAISE")
    ap.add_argument("--dataset", choices=config.DATASETS, help="só um dataset")
    ap.add_argument("--clones", type=Path, help="pasta com clones git dos repositórios, para expandir merges (NEO)")
    args = ap.parse_args()

    config.CONTEXT_DIR.mkdir(parents=True, exist_ok=True)
    repos = load_repos()
    if args.dataset:
        repos = {k: (v if k == args.dataset else {}) for k, v in repos.items()}

    if args.planejar or args.coletar:
        pedidos = plan_collection(repos, args.meses)
        path = config.CONTEXT_DIR / "raise_pedidos.json"
        path.write_text(json.dumps(pedidos, indent=1), encoding="utf-8")
        print(f"{len(pedidos)} pedidos -> {path.relative_to(config.ROOT)}")
        if args.coletar:
            run_collection(pedidos, args.api)
        return

    conn = raise_db.connect()
    all_links, all_tref = [], []
    for ds, projs in repos.items():
        if not projs:
            continue
        rs = sorted({r for v in projs.values() for r in v})
        commits = raise_db.load_commits(conn, rs)
        faltam = sorted(set(rs) - set(commits.repo))
        if faltam:
            print(f"AVISO ({ds}): sem commits no RAISE para {faltam}. Rode --coletar.")
        if ds == "agile":
            issues = agile_issues(list(projs))
            links = link_agile(issues, commits, projs)
        else:
            issues = neo_issues(list(projs))
            links = link_neo(issues, commits, projs)
        links = links.drop_duplicates(["issue_id", "sha"])
        if args.clones:
            antes = len(links)
            links = expand_merges(links, commits, args.clones, issues)
            print(f"{ds}: +{len(links) - antes} commits vindos de branches de merges ligados")
        tref = reference_dates(issues, links)
        all_links.append(links)
        all_tref.append(tref)

        resumo = tref.groupby("project").agg(issues=("issue_id", "size"), ligadas=("n_commits", lambda s: (s > 0).sum()),
                                             com_t_ref=("t_ref", lambda s: s.notna().sum()))
        resumo["ligadas_%"] = (100 * resumo.ligadas / resumo.issues).round(0)
        resumo["repos"] = [", ".join(projs[p]) for p in resumo.index]
        print(f"\n== {ds}: {len(commits)} commits lidos do RAISE ==")
        print(resumo.sort_values("ligadas_%", ascending=False).to_string())
        print("regras:", links.drop_duplicates("issue_id").regra.value_counts().to_dict())

    links = pd.concat(all_links, ignore_index=True)
    tref = pd.concat(all_tref, ignore_index=True)
    feitos = set(tref.dataset)
    for name, df in (("issue_commits.csv", links), ("issue_tref.csv", tref)):
        path = config.CONTEXT_DIR / name
        if path.exists():   # com --dataset, preserva as linhas do outro dataset
            old = pd.read_csv(path, dtype={"issue_id": str, "project": str})
            df = pd.concat([old[~old.dataset.isin(feitos)], df], ignore_index=True)
        df.to_csv(path, index=False)
    links = pd.read_csv(config.CONTEXT_DIR / "issue_commits.csv")
    tref = pd.read_csv(config.CONTEXT_DIR / "issue_tref.csv")
    print(f"\n{len(links)} ligações issue-commit, {int((tref.n_commits > 0).sum())} issues ligadas "
          f"-> {(config.CONTEXT_DIR / 'issue_commits.csv').relative_to(config.ROOT)}, issue_tref.csv")


if __name__ == "__main__":
    main()
