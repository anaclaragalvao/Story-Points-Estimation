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

    # EXTENSÃO: o mesmo prompt com o cartão do projeto antes (gere os cartões com 08_project_cards.py).
    # Só issues de projetos com cartão; salva como <modelo>__ZS1__cartao. Compare com 31_estimativa_contexto.py.
    python repro/30_run_llm.py --provider ollama --model llama3 --cartao

    # EXTENSÃO (controle): cartão PLACEBO = o cartão de OUTRO projeto do mesmo dataset. Se o cartão real
    # ganhar do placebo, o ganho vem da informação do projeto, não de "ter um texto de contexto" no prompt.
    # Salva como <modelo>__ZS1__cartaoplacebo (mesmas issues do --cartao).
    python repro/30_run_llm.py --provider ollama --model llama3 --dataset agile_limpo neo_limpo --cartao-placebo

    # EXTENSÃO: few-shot com exemplos do treino do MESMO projeto (estimativa por analogia, como o time faz).
    #   same_project          = exemplos sorteados (já existia)
    #   similar_same_project  = os exemplos MAIS PARECIDOS com a issue (TF-IDF ajustado só no treino do projeto)
    # O número de exemplos (2 a 4) é o mesmo nos dois, por issue: só muda QUAIS exemplos entram.
    # --so-projetos-com-cartao restringe às issues do --cartao (comparação pareada e menos chamadas).
    python repro/30_run_llm.py --provider ollama --model llama3 --dataset agile_limpo neo_limpo \
        --protocol within --template FS1 --shots-from similar_same_project --so-projetos-com-cartao
    # ... e junto com o cartão:
    python repro/30_run_llm.py --provider ollama --model llama3 --dataset agile_limpo neo_limpo \
        --protocol within --template FS1 --shots-from similar_same_project --cartao
    # A análise pareada de tudo isso está em analise_extensoes.ipynb (parte 1).

    # EXTENSÃO: variações da estimativa por analogia (cada uma gera um arquivo com nome próprio):
    #   --retriever ollama-embed   busca os parecidos por embeddings do Ollama em vez de TF-IDF
    #                              (antes: ollama pull nomic-embed-text; os embeddings ficam em cache)
    #   --k 8                      número fixo de exemplos (padrão: 2 a 4, como no artigo)
    #   --dist-sp                  põe no prompt a distribuição de story points do time (treino do projeto);
    #                              com ZS1 testa se "saber a escala do time" basta, sem exemplos
    python repro/30_run_llm.py --provider ollama --model llama3 --dataset agile_limpo neo_limpo \
        --protocol within --template FS1 --shots-from similar_same_project --retriever ollama-embed --k 8
    python repro/30_run_llm.py --provider ollama --model llama3 --dataset agile_limpo neo_limpo --dist-sp

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
from see.prompts import FS_TEMPLATES, ZS_TEMPLATES, few_shot, format_sp, parse_estimate, truncate, zero_shot  # noqa: E402
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


# EXTENSÃO: opções da estimativa por analogia. main() preenche a partir da linha de comando; quem importa
# este arquivo (ex.: o notebook) fica com os padrões, que são o comportamento original.
OPTS = {"retriever": "tfidf", "embed_model": "nomic-embed-text", "k": 0, "dist_sp": False,
        "ollama_url": None, "dataset": None}

_SIMILAR_INDEX = {}   # (id do df, projeto, recuperador) -> (treino do projeto, índice, matriz)


def _texto(title, description) -> str:
    return clean_light(title) + " " + clean_light(description)


def _ollama_url() -> str:
    url = OPTS["ollama_url"] or __import__("os").environ.get("OLLAMA_HOST", "http://localhost:11434")
    return (url if url.startswith("http") else "http://" + url).rstrip("/")


def _embed(texts, kind):
    """Embeddings normalizados pelo Ollama (/api/embed). kind = 'query' ou 'document' (prefixo do nomic)."""
    import requests
    pre = {"query": "search_query: ", "document": "search_document: "}[kind] if "nomic" in OPTS["embed_model"] else ""
    out = []
    for i in range(0, len(texts), 64):
        batch = [pre + t[:2000] for t in texts[i:i + 64]]
        r = requests.post(f"{_ollama_url()}/api/embed", json={"model": OPTS["embed_model"], "input": batch}, timeout=600)
        if r.status_code == 404 and "not found" in r.text.lower():
            sys.exit(f"Modelo de embeddings ausente no Ollama. Rode antes: ollama pull {OPTS['embed_model']}")
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    m = np.asarray(out, dtype=np.float32)
    return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)


def _embed_cached(texts, project):
    """Embeddings dos textos do treino de um projeto, em cache: results/embeddings/<modelo>/<dataset>/<projeto>.npz."""
    folder = config.RESULTS / "embeddings" / slug(OPTS["embed_model"]) / str(OPTS["dataset"] or "x")
    path = folder / f"{slug(str(project))}.npz"
    keys = np.array([hashlib.sha1(t.encode()).hexdigest() for t in texts])
    if path.exists():
        z = np.load(path)
        if len(z["keys"]) == len(keys) and (z["keys"] == keys).all():
            return z["emb"].astype(np.float32)
    print(f"   embeddings do treino de {project} ({len(texts)} textos) ...", flush=True)
    emb = _embed(texts, "document")
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, keys=keys, emb=emb.astype(np.float16))
    return emb


def _similar_index(df: pd.DataFrame, project):
    key = (id(df), project, OPTS["retriever"])
    if key not in _SIMILAR_INDEX:
        pool = df[(df.project == project) & (df.within == "train")]
        texts = [_texto(t, d) for t, d in zip(pool.title, pool.description)]
        if pool.empty:
            _SIMILAR_INDEX[key] = (pool, None, None)
        elif OPTS["retriever"] == "ollama-embed":
            _SIMILAR_INDEX[key] = (pool, "embed", _embed_cached(texts, project))
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit(texts)   # só o treino do projeto
            _SIMILAR_INDEX[key] = (pool, vec, vec.transform(texts))
    return _SIMILAR_INDEX[key]


def sp_distribution(df: pd.DataFrame, project) -> str:
    """EXTENSÃO --dist-sp: como o time costuma pontuar (só o treino do projeto)."""
    sp = df[(df.project == project) & (df.within == "train")].sp.astype(float)
    if sp.empty:
        return ""
    freq = sp.value_counts(normalize=True).head(6).sort_index()
    usados = ", ".join(f"{format_sp(v)} ({100 * f:.0f}%)" for v, f in freq.items())
    return ("Story point history of this team (work items previously estimated in the same project):\n"
            f"median = {format_sp(sp.median())}; most used values: {usados}\n\n")


def pick_examples(df: pd.DataFrame, row, shots_from: str, max_chars: int):
    """2 a 4 exemplos rotulados por issue (k estável por issue).

    other_projects        como no artigo: sorteados dos projetos de treino do cross-project
    same_project          sorteados do treino do MESMO projeto
    similar_same_project  os k mais parecidos do treino do MESMO projeto (similaridade de cosseno TF-IDF);
                          o mais parecido fica por último, logo antes da issue a estimar
    """
    rng = np.random.default_rng([config.SEED, zlib.crc32(row.issue_id.encode())])
    lo, hi = config.FEWSHOT_K_RANGE
    if shots_from == "similar_same_project":
        pool, vec, mat = _similar_index(df, row.project)
        k = min(int(rng.integers(lo, hi + 1)), len(pool))   # mesmo k do same_project para esta issue
        if OPTS["k"]:
            k = min(OPTS["k"], len(pool))
        if k == 0:
            return []
        if isinstance(vec, str):                             # embeddings
            sims = mat @ _embed([_texto(row.title, row.description)], "query")[0]
        else:
            sims = (mat @ vec.transform([_texto(row.title, row.description)]).T).toarray().ravel()
        top = np.argsort(-sims, kind="stable")[:k][::-1]
        chosen = pool.iloc[top]
    else:
        if shots_from == "other_projects":
            pool = df[(df.cross == "train") & (df.project != row.project)]
        else:
            pool = df[(df.project == row.project) & (df.within == "train") & (df.issue_id != row.issue_id)]
        k = min(int(rng.integers(lo, hi + 1)), len(pool))
        if OPTS["k"]:
            k = min(OPTS["k"], len(pool))
        chosen = pool.iloc[rng.choice(len(pool), k, replace=False)]
    return [(clean_light(r.title), truncate(clean_light(r.description), max_chars), r.sp)
            for r in chosen.itertuples()]


CARD_PREFIX = ("Project context (background about the software project this work item belongs to):\n"
               "{card}\n\n")


def load_cards() -> dict:
    """EXTENSÃO: (dataset base, projeto) -> cartão do projeto (08_project_cards.py). Vale também para *_limpo."""
    path = config.DATA_PROCESSED / "project_cards.csv"
    if not path.exists():
        sys.exit("Falta data/processed/project_cards.csv. Rode antes: python repro/08_project_cards.py")
    c = pd.read_csv(path, dtype={"project": str}).fillna("")
    return {(r.dataset, r.project): (r.cartao or r.cartao_bruto) for r in c.itertuples()}


def placebo_map(df: pd.DataFrame, dataset: str, cards: dict) -> dict:
    """EXTENSÃO --cartao-placebo: projeto -> projeto cujo cartão será usado no lugar do verdadeiro.

    Entre os projetos do dataset que têm cartão, em ordem alfabética, cada um recebe o cartão do SEGUINTE
    (o último recebe o do primeiro). Nenhum projeto recebe o próprio cartão, e todo cartão é usado uma vez.
    """
    base = config.base_dataset(dataset)
    projs = sorted({p for p in df.project.unique() if (base, p) in cards}, key=str)
    if len(projs) < 2:
        sys.exit(f"{dataset}: o placebo precisa de pelo menos 2 projetos com cartão.")
    return {p: projs[(i + 1) % len(projs)] for i, p in enumerate(projs)}


def build_prompt(df, row, template, shots_from, max_chars, card=None):
    title = clean_light(row.title)
    desc = truncate(clean_light(row.description), max_chars)
    if template in FS_TEMPLATES:
        prompt = few_shot(template, title, desc, pick_examples(df, row, shots_from, max_chars // 3))
    else:
        prompt = zero_shot(template, title, desc)
    if OPTS["dist_sp"]:
        prompt = sp_distribution(df, row.project) + prompt
    return CARD_PREFIX.format(card=card) + prompt if card else prompt


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


def run_one(df, dataset, protocol, template, args, client, cards=None):
    inst = instances(df, protocol)
    base = config.base_dataset(dataset)
    if cards is not None:   # EXTENSÃO --cartao / --cartao-placebo / --so-projetos-com-cartao: só projetos com cartão
        inst = inst[[(base, p) in cards for p in inst.project]]
    if args.limit:
        inst = inst.head(args.limit)
    shots_tag = "" if template in ZS_TEMPLATES else f"__{args.shots_from}"
    if template in FS_TEMPLATES and args.shots_from == "similar_same_project" and args.retriever != "tfidf":
        shots_tag += "__emb"
    if template in FS_TEMPLATES and args.k:
        shots_tag += f"__k{args.k}"
    if args.dist_sp:
        shots_tag += "__distsp"
    card_tag = "__cartao" if args.cartao else "__cartaoplacebo" if args.cartao_placebo else ""
    run_name = f"{slug(args.model)}__{template}{shots_tag}{card_tag}"
    if args.cartao:
        card_of = lambda r: cards[(base, r.project)]  # noqa: E731
    elif args.cartao_placebo:
        troca = placebo_map(df, dataset, cards)
        print(f"   placebo ({dataset}): " + "; ".join(f"{p} recebe o cartão de {q}" for p, q in troca.items()))
        card_of = lambda r: cards[(base, troca[r.project])]  # noqa: E731
    else:
        card_of = lambda r: None  # noqa: E731
    raw_path = config.LLM_RAW / dataset / protocol / f"{run_name}.jsonl"

    if args.dry_run:
        row = next(inst.itertuples())
        print("-" * 70 + f"\n{run_name} | {row.issue_id} | SP real = {row.sp}\n" + "-" * 70)
        print(build_prompt(df, row, template, args.shots_from, args.max_desc_chars, card_of(row)))
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
            prompt = build_prompt(df, row, template, args.shots_from, args.max_desc_chars, card_of(row))
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
    ap.add_argument("--shots-from", default="other_projects",
                    choices=["other_projects", "same_project", "similar_same_project"],
                    help="origem dos exemplos few-shot (artigo: other_projects; EXTENSÃO: similar_same_project)")
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
    ap.add_argument("--retriever", default="tfidf", choices=["tfidf", "ollama-embed"],
                    help="EXTENSÃO: como achar os exemplos parecidos (similar_same_project)")
    ap.add_argument("--embed-model", default="nomic-embed-text", help="modelo de embeddings no Ollama")
    ap.add_argument("--k", type=int, default=0, help="EXTENSÃO: nº fixo de exemplos few-shot (0 = 2 a 4, como no artigo)")
    ap.add_argument("--dist-sp", action="store_true",
                    help="EXTENSÃO: põe no prompt a distribuição de story points do time (treino do projeto)")
    ctx = ap.add_mutually_exclusive_group()
    ctx.add_argument("--cartao", action="store_true",
                     help="EXTENSÃO: põe o cartão do projeto (08_project_cards.py) antes do prompt; "
                          "só issues de projetos com cartão")
    ctx.add_argument("--cartao-placebo", action="store_true",
                     help="EXTENSÃO (controle): põe o cartão de OUTRO projeto do mesmo dataset; mesmas issues do --cartao")
    ctx.add_argument("--so-projetos-com-cartao", action="store_true",
                     help="EXTENSÃO: sem cartão no prompt, mas só as issues do --cartao (para comparar pareado)")
    args = ap.parse_args()
    cards = load_cards() if (args.cartao or args.cartao_placebo or args.so_projetos_com_cartao) else None
    OPTS.update(retriever=args.retriever, embed_model=args.embed_model, k=args.k, dist_sp=args.dist_sp,
                ollama_url=args.base_url if args.provider == "ollama" else None)

    client = None
    if not (args.dry_run or args.export_only):
        client = LLMClient(args.provider, args.model, temperature=args.temperature, max_tokens=args.max_tokens,
                           base_url=args.base_url, api_key_env=args.api_key_env, seed=config.SEED)
    for dataset in args.dataset:
        df = load_merged(dataset)
        OPTS["dataset"] = dataset
        for protocol in args.protocol:
            templates = args.template or (["FS1"] if protocol == "fewshot" else ["ZS1"])
            if protocol == "perturb":
                for proto, view in perturbed_views(df, dataset):
                    for template in templates:
                        run_one(view, dataset, proto, template, args, client, cards)
                continue
            for template in templates:
                if protocol == "fewshot" and template not in FS_TEMPLATES:
                    sys.exit("O protocolo fewshot usa templates FS1..FS3.")
                proto = protocol if template in ZS_TEMPLATES or protocol == "fewshot" else f"{protocol}_fewshot"
                if args.shots_from == "same_project" and template in FS_TEMPLATES:
                    proto += "_sameproj"
                elif args.shots_from == "similar_same_project" and template in FS_TEMPLATES:
                    proto += "_simproj"
                run_one(df, dataset, proto, template, args, client, cards)


if __name__ == "__main__":
    main()
