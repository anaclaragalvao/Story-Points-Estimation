"""Etapa LLM — estimativa por prompting (zero-shot e few-shot), sem fine-tuning.

Pode ser rodada em QUALQUER computador que tenha a pasta repro/ com os dados
preparados (etapas 01 e 02) e os splits (já versionados no repositório).

Exemplos:
    # Ollama local (modelo pequeno, roda na CPU)
    python repro/30_run_llm.py --provider ollama --model llama3.2:3b --dataset agile

    # APIs com camada gratuita (exporte a chave antes)
    export GEMINI_API_KEY=...;  python repro/30_run_llm.py --provider gemini --model gemini-2.5-flash
    export GROQ_API_KEY=...;    python repro/30_run_llm.py --provider groq --model llama-3.1-8b-instant --rpm 25

    # Ver o prompt sem chamar nada
    python repro/30_run_llm.py --provider ollama --model x --dry-run

    # Few-shot como no artigo (cross-project, exemplos de OUTROS projetos)
    python repro/30_run_llm.py --protocol fewshot --template FS1 --provider ollama --model llama3.2:3b

    # Sensibilidade ao prompt (RQ2): os 10 templates zero-shot
    python repro/30_run_llm.py --template ZS1 ZS2 ZS3 ZS4 ZS5 ZS6 ZS7 ZS8 ZS9 ZS10 ...

Retomável: cada resposta é gravada na hora em results/llm_raw/.../*.jsonl.
Se cair a internet, acabar a cota ou você fechar o notebook, rode o MESMO comando
de novo e ele continua de onde parou. Para juntar resultados feitos em outro
computador, copie os .jsonl para a mesma pasta e rode com --export-only.
"""
import argparse
import hashlib
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged, log_cost, save_predictions  # noqa: E402
from see.llm_clients import PRESETS, LLMClient, call_with_retry  # noqa: E402
from see.prompts import FS_TEMPLATES, ZS_TEMPLATES, few_shot, parse_estimate, truncate, zero_shot  # noqa: E402
from see.text import clean_light  # noqa: E402

PROTOCOL_COLUMN = {"within": "within_eval", "cross": "cross_eval", "lopo": "lopo_eval",
                   "fewshot": "fewshot_eval", "perturb": "within_eval"}


def perturbed_views(df: pd.DataFrame, dataset: str):
    """Para o protocolo perturb: uma cópia do df com título/descrição perturbados por seed."""
    files = sorted((config.ROOT / "data" / "perturbed").glob(f"{dataset}_s*.csv"))
    if not files:
        sys.exit("Não há textos perturbados. Rode antes: python repro/40_perturb_texts.py")
    for f in files:
        pert = pd.read_csv(f, dtype={"issue_id": str}, keep_default_na=False).set_index("issue_id")
        view = df.copy()
        mask = view.issue_id.isin(pert.index)
        view.loc[mask, "title"] = view.loc[mask, "issue_id"].map(pert["title"])
        view.loc[mask, "description"] = view.loc[mask, "issue_id"].map(pert["description"])
        yield f"perturb_s{f.stem.split('_s')[-1]}", view


def slug(s: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in s)


def instances(df: pd.DataFrame, protocol: str) -> pd.DataFrame:
    col = PROTOCOL_COLUMN[protocol.split("_")[0]]
    return df[df[col] == 1]


def pick_examples(df: pd.DataFrame, row, shots_from: str, max_chars: int):
    """2 a 4 exemplos rotulados, sorteados de forma estável por issue."""
    if shots_from == "other_projects":   # como no artigo: só projetos de treino do cross-project
        pool = df[(df.cross == "train") & (df.project != row.project)]
    else:                                # variação: exemplos do treino do MESMO projeto
        pool = df[(df.project == row.project) & (df.within == "train") & (df.issue_id != row.issue_id)]
    rng = np.random.default_rng([config.SEED, zlib.crc32(row.issue_id.encode())])
    lo, hi = config.FEWSHOT_K_RANGE
    k = min(int(rng.integers(lo, hi + 1)), len(pool))
    chosen = pool.iloc[rng.choice(len(pool), k, replace=False)]
    return [(clean_light(r.title), truncate(clean_light(r.description), max_chars), r.sp)
            for r in chosen.itertuples()]


def build_prompt(df, row, template, shots_from, max_chars):
    title = clean_light(row.title)
    desc = truncate(clean_light(row.description), max_chars)
    if template in FS_TEMPLATES:
        return few_shot(template, title, desc, pick_examples(df, row, shots_from, max_chars // 3))
    return zero_shot(template, title, desc)


def read_done(path: Path) -> dict:
    """issue_id -> registro mais recente SEM erro."""
    done = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if not rec.get("error"):
                    done[rec["issue_id"]] = rec
    return done


def export(df_inst: pd.DataFrame, raw_path: Path, dataset, protocol, run_name, provider, model, template):
    done = read_done(raw_path)
    if not done:
        print("  nada para exportar ainda")
        return
    pred = df_inst[["issue_id", "project", "sp"]].rename(columns={"sp": "y_true"}).copy()
    pred = pred[pred.issue_id.isin(done)]
    pred["y_pred"] = [done[i]["parsed"] for i in pred.issue_id]
    pred["raw_response"] = [done[i]["response"][:200] for i in pred.issue_id]
    pred["family"], pred["provider"], pred["llm"], pred["template"] = "LLM", provider, model, template
    path = save_predictions(pred, dataset, protocol, run_name)
    lat = [done[i]["latency_s"] for i in pred.issue_id]
    n_invalid = int(pred.y_pred.isna().sum())
    missing = len(df_inst) - len(pred)
    log_cost(dataset, protocol, "LLM", run_name, n_train=0, n_pred=len(pred), train_seconds=0,
             infer_seconds=round(float(np.mean(lat)), 3),
             notes=f"provider={provider}; infer_seconds = latência média por issue; invalidas={n_invalid}")
    print(f"  salvo: {path.relative_to(config.ROOT)}  ({len(pred)} issues, {n_invalid} sem número"
          + (f", faltam {missing}" if missing else "") + ")")


def run_one(df, dataset, protocol, template, args, client):
    inst = instances(df, protocol)
    if args.limit:
        inst = inst.head(args.limit)
    shots_tag = "" if template in ZS_TEMPLATES else f"__{args.shots_from}"
    run_name = f"{slug(args.model)}__{template}{shots_tag}"
    raw_path = config.LLM_RAW / dataset / protocol / f"{run_name}.jsonl"

    if args.dry_run:
        row = next(inst.itertuples())
        print("-" * 70 + f"\n{run_name} | {row.issue_id} | SP real = {row.sp}\n" + "-" * 70)
        print(build_prompt(df, row, template, args.shots_from, args.max_desc_chars))
        return
    if args.export_only:
        export(inst, raw_path, dataset, protocol, run_name, args.provider, args.model, template)
        return

    done = read_done(raw_path)
    todo = inst[~inst.issue_id.isin(done)]
    print(f"== {dataset}/{protocol}/{run_name}: {len(done)} prontas, {len(todo)} a fazer", flush=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    min_interval = 60.0 / args.rpm if args.rpm else 0.0
    last = 0.0
    with open(raw_path, "a", encoding="utf-8") as f:
        for n, row in enumerate(todo.itertuples(), start=1):
            prompt = build_prompt(df, row, template, args.shots_from, args.max_desc_chars)
            wait = min_interval - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            rec = {"issue_id": row.issue_id, "provider": args.provider, "model": args.model,
                   "template": template, "temperature": args.temperature,
                   "prompt_sha1": hashlib.sha1(prompt.encode()).hexdigest(),
                   "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
            try:
                text, latency, attempts = call_with_retry(client, prompt)
                rec.update(response=text, parsed=parse_estimate(text), latency_s=round(latency, 3),
                           attempts=attempts, error=None)
            except Exception as e:  # registra o erro e segue; o próximo comando re-tenta essa issue
                rec.update(response="", parsed=None, latency_s=None, attempts=None, error=str(e)[:500])
                print(f"    erro em {row.issue_id}: {str(e)[:200]}", flush=True)
                if "HTTP 4" in str(e) and "429" not in str(e):
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    sys.exit("Erro de requisição (4xx) — confira o nome do modelo e a chave de API.")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if n % 25 == 0 or n == len(todo):
                print(f"    {n}/{len(todo)}  última resposta: {rec['response'][:40]!r} (real {row.sp:g})", flush=True)
    export(inst, raw_path, dataset, protocol, run_name, args.provider, args.model, template)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, choices=["ollama", *PRESETS, "openai_compat"])
    ap.add_argument("--model", required=True, help="nome do modelo no provedor (ex.: llama3.2:3b)")
    ap.add_argument("--dataset", nargs="+", default=config.DATASETS, choices=config.ALL_DATASETS)
    ap.add_argument("--protocol", nargs="+", default=["within"], choices=list(PROTOCOL_COLUMN))
    ap.add_argument("--template", nargs="+", default=None,
                    help="ZS1..ZS10 (zero-shot) ou FS1..FS3 (few-shot). Padrão: ZS1, ou FS1 no protocolo fewshot")
    ap.add_argument("--shots-from", default="other_projects", choices=["other_projects", "same_project"],
                    help="origem dos exemplos few-shot (artigo: other_projects)")
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="o artigo diz apenas 'configurações padrão'; fixamos 0 para reprodutibilidade")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--max-desc-chars", type=int, default=2000, help="corta descrições longas (economiza cota)")
    ap.add_argument("--rpm", type=float, default=0, help="limite de requisições por minuto (0 = sem limite)")
    ap.add_argument("--limit", type=int, default=0, help="só as N primeiras issues (teste rápido)")
    ap.add_argument("--base-url")
    ap.add_argument("--api-key-env")
    ap.add_argument("--dry-run", action="store_true", help="mostra o primeiro prompt e sai")
    ap.add_argument("--export-only", action="store_true", help="só converte os .jsonl em predições")
    args = ap.parse_args()

    client = None
    if not (args.dry_run or args.export_only):
        client = LLMClient(args.provider, args.model, temperature=args.temperature, max_tokens=args.max_tokens,
                           base_url=args.base_url, api_key_env=args.api_key_env, seed=config.SEED)
    for dataset in args.dataset:
        df = load_merged(dataset)
        for protocol in args.protocol:
            templates = args.template or (["FS1"] if protocol == "fewshot" else ["ZS1"])
            if protocol == "perturb":
                for proto, view in perturbed_views(df, dataset):
                    for template in templates:
                        run_one(view, dataset, proto, template, args, client)
                continue
            for template in templates:
                if protocol == "fewshot" and template not in FS_TEMPLATES:
                    sys.exit("O protocolo fewshot usa templates FS1..FS3.")
                proto = protocol if template in ZS_TEMPLATES or protocol == "fewshot" else f"{protocol}_fewshot"
                if args.shots_from == "same_project" and template in FS_TEMPLATES:
                    proto += "_sameproj"
                run_one(df, dataset, proto, template, args, client)


if __name__ == "__main__":
    main()
