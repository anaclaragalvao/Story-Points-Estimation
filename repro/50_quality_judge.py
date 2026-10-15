"""EXTENSÃO — qualidade da entrada: avalia cada issue com um juiz "System One" (Jev ou Laya).

Jev (TypeSafe AI) é o juiz de decisões estudado em Li et al., "JEV-as-a-Judge: Accept
When Confident, Escalate When Unsure" (arXiv 2609.26550). Ele não gera texto: devolve
probabilidades para perguntas tipadas (choice, score, noul). Laya (Convai Innovations,
Apache 2.0) é um modelo aberto que expõe a MESMA API e roda localmente, de graça.

    # Laya local (grátis). Num terminal separado, suba o servidor:
    #   pip install "laya[serve]"
    #   set LAYA_MODELS=typed-decisions      (no cmd; no Linux/Mac: export ...)
    #   laya-serve
    python repro/50_quality_judge.py --engine laya --dataset agile --limit 20      # teste
    python repro/50_quality_judge.py --engine laya                                  # 2.188 issues

    # Jev (pago; custo estimado no README)
    set TYPESAFE_API_KEY=sua_chave
    python repro/50_quality_judge.py --engine jev

    python repro/50_quality_judge.py --engine laya --dry-run     # mostra a requisição e sai

Por padrão avalia só as issues de avaliação (`within_eval`: 1.166 Agile + 1.022 NEO),
que são as mesmas em que medimos o erro dos modelos. `--subset test` avalia o test set inteiro
e `--subset all` o dataset inteiro. Funciona também com as versões limpas (agile_limpo,
neo_limpo): issues já julgadas em qualquer rodada são reaproveitadas, sem pagar de novo.
Saída: results/quality/{dataset}{_test|_all}__{engine}.csv

As perguntas ficam em repro/quality_criteria.json (editável). Retomável como o script de
LLM: as respostas vão para results/quality_raw/*.jsonl na hora; se parar, rode de novo.
Saída final: results/quality/{dataset}__{engine}.csv, uma linha por issue, com as respostas
e duas notas gerais (see/quality.py): indice_qualidade (0–100, composto) e nota_geral_juiz
(0–100, avaliação holística do juiz), além da faixa ruim / regular / boa.

ATENÇÃO (do próprio artigo do Jev): juízes desse tipo são bons em decisões que dá para
"ler" no texto e fracos em julgar prosa sem referência. Antes de usar os números no TCC,
valide com ~100 issues rotuladas por você (51_quality_report.py sample / report).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged  # noqa: E402
from see.prompts import truncate  # noqa: E402
from see.quality import add_quality_scores  # noqa: E402
from see.text import clean_light  # noqa: E402

ENGINES = {
    "jev": {"base_url": "https://api.typesafe.ai", "model": "jev-latest", "key_env": "TYPESAFE_API_KEY"},
    "laya": {"base_url": "http://localhost:8000", "model": "typed-decisions", "key_env": "LAYA_API_KEY"},
}
CRITERIA_FILE = config.ROOT / "quality_criteria.json"
RAW_DIR = config.RESULTS / "quality_raw"
OUT_DIR = config.RESULTS / "quality"


def load_questions(path: Path) -> dict:
    q = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in q.items() if not k.startswith("_")}


def make_state(row, max_chars: int, fmt: str):
    title = clean_light(row.title)
    desc = truncate(clean_light(row.description), max_chars) or "(no description)"
    if fmt == "object":
        return {"title": title, "description": desc}
    return f"Title: {title}\nDescription: {desc}"


class Judge:
    def __init__(self, engine, base_url, model, timeout=120):
        spec = ENGINES[engine]
        self.url = (base_url or spec["base_url"]).rstrip("/") + "/v1/systemone"
        self.model = model or spec["model"]
        key = os.environ.get(spec["key_env"])
        if engine == "jev" and not key:
            raise SystemExit(f"Defina a variável de ambiente {spec['key_env']} com a sua chave do Jev.")
        self.headers = {"Content-Type": "application/json"}
        if key:
            self.headers["Authorization"] = f"Bearer {key}"
        self.timeout = timeout

    def ask(self, state, questions, retries=6):
        body = {"state": state, "model": self.model, "questions": questions}
        for attempt in range(1, retries + 1):
            t0 = time.perf_counter()
            try:
                r = requests.post(self.url, json=body, headers=self.headers, timeout=self.timeout)
            except (requests.ConnectionError, requests.Timeout) as e:
                err, wait = f"conexão: {e}", None
            else:
                if r.status_code == 200:
                    return r.json(), time.perf_counter() - t0
                if r.status_code != 429 and r.status_code < 500:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
                err = f"HTTP {r.status_code}"
                try:
                    wait = float(r.headers.get("retry-after"))
                except (TypeError, ValueError):
                    wait = None
            if attempt == retries:
                raise RuntimeError(err)
            wait = max(wait or 0, 3 * 2 ** (attempt - 1))
            print(f"    re-tentando em {wait:.0f}s ({err})", flush=True)
            time.sleep(wait)


def flatten(answers: dict, questions: dict) -> dict:
    """Uma coluna por resultado: choice/score/noul + confiança + probabilidades."""
    out = {}
    for qid, spec in questions.items():
        a = answers.get(qid, {})
        t = spec["type"]
        if t == "choice":
            out[f"{qid}"] = a.get("choice")
            out[f"{qid}_conf"] = a.get("confidence")
            for opt, p in (a.get("probabilities") or {}).items():
                out[f"{qid}_p_{opt}"] = p
        elif t == "score":
            out[f"{qid}"] = a.get("score")
            out[f"{qid}_conf"] = a.get("confidence")
        elif t == "noul":
            out[f"{qid}"] = a.get("noul")
    return out


def read_done(path: Path, questions: dict = None) -> dict:
    """issue_id -> registro mais recente sem erro. Se as perguntas mudaram (ex.: você
    acrescentou uma no quality_criteria.json), registros sem alguma delas contam como
    não feitos e a issue é perguntada de novo."""
    done = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("error"):
                    continue
                if questions and not set(questions) <= set(rec.get("answers", {})):
                    continue
                done[rec["issue_id"]] = rec
    return done


def read_done_all(raw_dir: Path, engine: str, questions: dict) -> dict:
    """Junta as respostas de TODOS os arquivos do juiz para o dataset-base (eval, test, all):
    uma issue já julgada não é paga de novo, nem quando muda o subconjunto ou se usa a versão limpa."""
    done = {}
    for f in sorted(raw_dir.glob(f"{engine}*.jsonl")):
        done.update(read_done(f, questions))
    return done


def export(df_sel, raw_path, questions, dataset, engine, done=None):
    done = done if done is not None else read_done(raw_path, questions)
    rows = []
    for r in df_sel.itertuples():
        rec = done.get(r.issue_id)
        if rec is None:
            continue
        row = {"issue_id": r.issue_id, "project": r.project, "sp": r.sp,
               "engine": rec["engine"], "model": rec.get("model_returned") or rec["model"],
               "latency_s": rec["latency_s"]}
        row.update(flatten(rec["answers"], questions))
        rows.append(row)
    if not rows:
        print("  nada para exportar ainda")
        return
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{dataset}__{engine}.csv"
    add_quality_scores(pd.DataFrame(rows)).to_csv(path, index=False)
    missing = len(df_sel) - len(rows)
    print(f"  salvo: {path.relative_to(config.ROOT)} ({len(rows)} issues" + (f", faltam {missing})" if missing else ")"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, choices=list(ENGINES))
    ap.add_argument("--dataset", nargs="+", default=config.DATASETS, choices=config.ALL_DATASETS)
    ap.add_argument("--subset", default="eval", choices=["eval", "test", "all"],
                    help="eval = issues de avaliação (padrão); test = test set inteiro do within; all = dataset inteiro")
    ap.add_argument("--criteria", default=str(CRITERIA_FILE))
    ap.add_argument("--base-url", help="padrão: API do Jev ou http://localhost:8000 (Laya)")
    ap.add_argument("--model", help="padrão: jev-latest (Jev) ou typed-decisions (Laya)")
    ap.add_argument("--state-format", default="text", choices=["text", "object"],
                    help="text = 'Title: ...\\nDescription: ...'; object = {title, description}")
    ap.add_argument("--max-desc-chars", type=int, default=2000)
    ap.add_argument("--rpm", type=float, default=0, help="limite de requisições por minuto")
    ap.add_argument("--limit", type=int, default=0, help="só as N primeiras issues (teste)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--export-only", action="store_true")
    args = ap.parse_args()

    questions = load_questions(Path(args.criteria))
    judge = None
    for dataset in args.dataset:
        df = load_merged(dataset)
        sel = {"all": df, "test": df[df.within == "test"], "eval": df[df.within_eval == 1]}[args.subset]
        if args.limit:
            sel = sel.head(args.limit)
        suffix = "" if args.subset == "eval" else f"_{args.subset}"
        raw_dir = RAW_DIR / config.base_dataset(dataset)        # cache compartilhado: agile e agile_limpo
        raw_path = raw_dir / f"{args.engine}{suffix}.jsonl"

        if args.dry_run:
            row = next(sel.itertuples())
            body = {"state": make_state(row, args.max_desc_chars, args.state_format),
                    "model": args.model or ENGINES[args.engine]["model"], "questions": questions}
            print(json.dumps(body, indent=2, ensure_ascii=False))
            return
        if args.export_only:
            export(sel, raw_path, questions, dataset + suffix, args.engine,
                   read_done_all(raw_dir, args.engine, questions))
            continue
        if judge is None:
            judge = Judge(args.engine, args.base_url, args.model)

        done = read_done_all(raw_dir, args.engine, questions)
        todo = sel[~sel.issue_id.isin(done)]
        print(f"== {dataset} ({args.engine}): {len(done)} prontas, {len(todo)} a fazer", flush=True)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        min_interval = 60.0 / args.rpm if args.rpm else 0.0
        last = 0.0
        with open(raw_path, "a", encoding="utf-8") as f:
            for n, row in enumerate(todo.itertuples(), start=1):
                wait = min_interval - (time.time() - last)
                if wait > 0:
                    time.sleep(wait)
                last = time.time()
                rec = {"issue_id": row.issue_id, "engine": args.engine, "model": judge.model,
                       "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
                try:
                    resp, latency = judge.ask(make_state(row, args.max_desc_chars, args.state_format), questions)
                    rec.update(answers=resp.get("answers", {}), model_returned=resp.get("model"),
                               usage=resp.get("usage"), latency_s=round(latency, 3), error=None)
                except Exception as e:
                    rec.update(answers={}, latency_s=None, error=str(e)[:500])
                    print(f"    erro em {row.issue_id}: {str(e)[:200]}", flush=True)
                    if str(e).startswith("HTTP 4"):
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        sys.exit("Erro na requisição (4xx): confira a chave, o modelo e o formato das perguntas.")
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                if n % 50 == 0 or n == len(todo):
                    print(f"    {n}/{len(todo)}", flush=True)
        export(sel, raw_path, questions, dataset + suffix, args.engine, read_done_all(raw_dir, args.engine, questions))


if __name__ == "__main__":
    main()
