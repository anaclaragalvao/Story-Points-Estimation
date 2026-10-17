"""EXTENSÃO (não faz parte do artigo) — "cartão do projeto" a partir dos metadados do RAISE.

O texto das issues não traz o que quem trabalha no projeto sabe: o que é o sistema, quais
são os módulos, o vocabulário do domínio. Esta etapa monta um contexto FIXO por projeto,
a partir do repositório no GitHub, para a condição "melhorar com contexto" da seção 12 do
analise_jev.ipynb.

O RAISE (github.com/aisepucrio/raise) entra só como extrator, e só a parte de metadados
(descrição, README, linguagens, tópicos). Nada de commits.

    # 0. RAISE rodando: na pasta do raise, `docker compose up --build` (token do GitHub no .env)
    python repro/08_project_cards.py --coletar      # 1. pede ao RAISE os metadados de cada repositório
    python repro/08_project_cards.py --exportar     # 2. baixa a tabela githubmetadata do RAISE (JSON)
    python repro/08_project_cards.py                # 3. monta os cartões (usa o Llama 3 do Ollama)

Os passos 1 e 2 também podem ser feitos pela interface web do RAISE (http://localhost:5173):
colete "metadata" de cada repositório de repro/raise_repos.json e exporte a tabela
githubmetadata em JSON para data/raw/raise/githubmetadata.json.

Opções do passo 3:
    --sem-llm                 só o cartão "bruto" (descrição + tópicos + linguagens + começo do README)
    --provider/--model        quem resume (padrão: ollama / llama3, igual ao REWRITER do notebook)
    --force                   refaz os cartões que já existem

Saída: data/processed/project_cards.csv, uma linha por projeto do dataset:
    dataset, project, repo, cartao_bruto, cartao, model
`cartao` é o resumo do LLM (vazio com --sem-llm); a seção 12 usa `cartao` e, se estiver
vazio, `cartao_bruto`.

Limites (registrar no TCC):
  - Retrato ATUAL do repositório: o README pode citar módulos que surgiram depois das issues.
  - Só projetos com repositório no GitHub (ver raise_repos.json).
  - Um cartão por projeto: o contexto é igual para todas as issues do projeto.
"""
import argparse
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402

RAISE_API = "http://localhost:8000/api"
EXPORT = config.DATA_RAW / "raise" / "githubmetadata.json"
CARDS = config.DATA_PROCESSED / "project_cards.csv"
README_CHARS = 4000

PROMPT_CARD = '''You are preparing background notes for developers who will refine and estimate work items of a software project.
Using ONLY the repository information below, write a short project brief (at most 150 words) covering:
1. what the system is and who uses it;
2. its main components or modules;
3. its main technologies;
4. domain terms a newcomer should know.
Do not invent components, features or facts that are not in the information below. Write plain text in English, without headings.

Repository: {repo}
Description: {description}
Topics: {topics}
Languages: {languages}
README (truncated):
{readme}'''


# --------------------------------------------------------------------------- RAISE
def _http(method, url, body=None, timeout=120):
    req = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8")


def load_map() -> pd.DataFrame:
    m = json.loads(config.RAISE_REPOS.read_text(encoding="utf-8"))
    rows = [{"dataset": ds, "project": str(p), "repo": r}
            for ds in config.DATASETS for p, rs in m.get(ds, {}).items() for r in rs]
    return pd.DataFrame(rows)


def collect(repos, api):
    for n, repo in enumerate(repos, 1):
        print(f"[{n}/{len(repos)}] metadados de {repo}", flush=True)
        try:
            resp = json.loads(_http("POST", f"{api}/github/collect/", {
                "targets": [repo], "collect_types": ["metadata"], "start_date": None, "end_date": None,
                "filters": {}, "options": {}}))
        except Exception as e:
            print(f"    erro ao pedir a coleta ({e}). O RAISE está rodando em {api}?")
            continue
        ids = [t["task_id"] for r in resp.get("results", []) for t in r.get("tasks", [])]
        status, t0 = ("PENDING" if ids else "FAILURE"), time.time()
        while ids and status not in ("SUCCESS", "FAILURE", "REVOKED") and time.time() - t0 < 600:
            time.sleep(5)
            try:
                status = json.loads(_http("GET", f"{api}/jobs/tasks/{ids[0]}/")).get("status", "PENDING")
            except Exception:
                pass
        print(f"    {status} em {time.time() - t0:.0f} s")


def export(api):
    text = _http("POST", f"{api}/github/export/", {"table": "githubmetadata", "format": "json"})
    EXPORT.parent.mkdir(parents=True, exist_ok=True)
    EXPORT.write_text(text, encoding="utf-8")
    print(f"{len(json.loads(text))} repositórios -> {EXPORT.relative_to(config.ROOT)}")


# --------------------------------------------------------------------------- cartões
def clean_readme(text: str) -> str:
    """README em texto corrido: sem HTML, imagens, badges, blocos de código e tabelas."""
    t = text or ""
    t = re.sub(r"```.*?```", " ", t, flags=re.S)                 # blocos de código
    t = re.sub(r"<!--.*?-->", " ", t, flags=re.S)
    t = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", t)                  # imagens/badges
    t = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", " ", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", t)               # links -> texto
    t = re.sub(r"(?m)^\s*\[[^\]]+\]:\s*\S+.*$", " ", t)          # definições de link [1]: http...
    t = re.sub(r"\[([^\]]+)\]\[[^\]]*\]", r"\1", t)              # links de referência [texto][1]
    t = re.sub(r"<[^>]+>", " ", t)                               # tags HTML
    t = html.unescape(t)
    lines = []
    for ln in t.splitlines():
        s = ln.strip()
        if not s or s.startswith("|") or re.fullmatch(r"[-=*_#\s]+", s):
            continue
        s = re.sub(r"^#+\s*", "", s)                             # títulos markdown
        s = re.sub(r"https?://\S+", "", s).strip()
        if s:
            lines.append(s)
    out = "\n".join(lines)
    return out[:README_CHARS].rsplit(" ", 1)[0] + " ..." if len(out) > README_CHARS else out


def fmt_languages(v) -> str:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return v
    langs = (v or {}).get("languages", []) if isinstance(v, dict) else (v or [])
    langs = sorted(langs, key=lambda d: -float(d.get("percentage", 0)))[:6]
    return ", ".join(f"{d['language']} ({float(d.get('percentage', 0)):.0f}%)" for d in langs)


def fmt_topics(v) -> str:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            return v
    return ", ".join(v or [])


def raw_card(meta: dict) -> str:
    parts = [f"Repository: {meta['repo']}"]
    if meta["description"]:
        parts.append(f"Description: {meta['description']}")
    if meta["topics"]:
        parts.append(f"Topics: {meta['topics']}")
    if meta["languages"]:
        parts.append(f"Languages: {meta['languages']}")
    if meta["readme"]:
        first = " ".join(meta["readme"].split())[:600].rsplit(" ", 1)[0]
        parts.append(f"README (start): {first} ...")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coletar", action="store_true", help="pede ao RAISE os metadados de cada repositório")
    ap.add_argument("--exportar", action="store_true", help="baixa a tabela githubmetadata do RAISE")
    ap.add_argument("--api", default=RAISE_API, help="URL da API do RAISE")
    ap.add_argument("--arquivo", type=Path, default=EXPORT, help="JSON exportado do RAISE (githubmetadata)")
    ap.add_argument("--sem-llm", action="store_true", help="só o cartão bruto, sem resumo do LLM")
    ap.add_argument("--provider", default="ollama")
    ap.add_argument("--model", default="llama3")
    ap.add_argument("--force", action="store_true", help="refaz cartões já existentes")
    args = ap.parse_args()

    mapa = load_map()
    if args.coletar:
        collect(sorted(mapa.repo.unique()), args.api)
    if args.exportar:
        export(args.api)
    if args.coletar or args.exportar:
        return

    if not args.arquivo.exists():
        sys.exit(f"{args.arquivo} não existe. Rode antes --coletar e --exportar (ou exporte pela interface do RAISE).")
    meta = {}
    for r in json.loads(args.arquivo.read_text(encoding="utf-8")):
        repo = str(r.get("repository", "")).lower()
        meta[repo] = {"repo": r.get("repository"), "description": " ".join(str(r.get("description") or "").split()),
                      "topics": fmt_topics(r.get("topics")), "languages": fmt_languages(r.get("languages")),
                      "readme": clean_readme(r.get("readme"))}

    old = pd.read_csv(CARDS, dtype={"project": str}).fillna("") if CARDS.exists() and not args.force else None
    client = None
    if not args.sem_llm:
        from see.llm_clients import LLMClient, call_with_retry
        client = LLMClient(args.provider, args.model, temperature=0.0, max_tokens=400, seed=config.SEED)

    rows = []
    for (ds, proj), g in mapa.groupby(["dataset", "project"], sort=False):
        ms = [meta[r.lower()] for r in g.repo if r.lower() in meta]
        if not ms:
            print(f"  {ds}/{proj}: sem metadados no RAISE para {list(g.repo)} (colete e exporte de novo)")
            continue
        m = ms[0]   # um repositório por projeto em raise_repos.json
        row = {"dataset": ds, "project": proj, "repo": m["repo"], "cartao_bruto": raw_card(m), "cartao": "", "model": ""}
        if old is not None:
            prev = old[(old.dataset == ds) & (old.project == proj) & (old.cartao != "")]
            if len(prev):
                row.update(cartao=prev.iloc[0].cartao, model=prev.iloc[0].model)
        if client and not row["cartao"]:
            prompt = PROMPT_CARD.format(repo=m["repo"], description=m["description"] or "(none)",
                                        topics=m["topics"] or "(none)", languages=m["languages"] or "(none)",
                                        readme=m["readme"] or "(no README)")
            try:
                text, lat, _ = call_with_retry(client, prompt)
                row.update(cartao=" ".join(text.split()), model=f"{args.provider}:{args.model}")
                print(f"  {ds}/{proj}: cartão gerado ({lat:.0f} s)", flush=True)
            except Exception as e:
                print(f"  {ds}/{proj}: erro no LLM ({str(e)[:120]}); fica só o cartão bruto")
        rows.append(row)

    out = pd.DataFrame(rows)
    CARDS.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CARDS, index=False)
    print(f"\n{len(out)} cartões ({(out.cartao != '').sum()} resumidos pelo LLM) -> {CARDS.relative_to(config.ROOT)}")
    falt = sorted(set(mapa.project) - set(out.project))
    if falt:
        print("sem cartão:", falt)
    if len(out):
        ex = out.iloc[0]
        print(f"\n--- exemplo ({ex.dataset}/{ex.project}) ---\n{ex.cartao or ex.cartao_bruto}")


if __name__ == "__main__":
    main()
