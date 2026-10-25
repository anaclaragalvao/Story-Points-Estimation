"""EXTENSÃO — fidelidade das reescritas: o LLM inventou ou perdeu informação ao reescrever as issues?

Três checagens automáticas, sem nenhuma chamada de LLM (usam as reescritas já salvas em results/improvement/):

1. Similaridade semântica (original × reescrita): cosseno entre embeddings do Ollama (nomic-embed-text).
   Mede se o assunto se manteve. O "parafrasear" (mesma informação, outras palavras) é a régua de referência.
2. Entidades técnicas: números, versões, arquivos, caminhos, identificadores de código, URLs, chaves de issue.
   - preservadas = quantas entidades do original sobreviveram (perda de informação);
   - inventadas = entidades da reescrita que não estão no original (nem no cartão do projeto, nas versões
     *_contexto, que podiam usá-lo).
3. Consistência factual por NLI (inferência textual), como no SummaC (Laban et al., TACL 2022): cada frase da
   reescrita é testada contra trechos do original (e do cartão, nas versões *_contexto). A frase é
   "sustentada" se algum trecho a implica (P(entailment) >= 0,5), "contradiz" se algum trecho a contradiz, e
   "nova" se nenhum trecho a sustenta. Modelo padrão: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli (roda na CPU).

Uma reescrita é marcada como SUSPEITA de invenção se tem entidade inventada OU se mais da metade das suas frases
não é sustentada pelo original (--limiar-nli). Os limiares são calibrados pela amostra humana (abaixo).

Saídas:
    results/improvement/fidelidade_reescritas.csv     uma linha por issue × versão (com as frases e entidades suspeitas)
    results/tables/fidelidade_resumo.csv              resumo por dataset × versão
    results/improvement/fidelidade_amostra_humana.xlsx  40 reescritas para você rotular (só é criada se não existir)
    results/improvement/fidelidade_nli_cache.jsonl    cache do NLI (rodar de novo não recalcula)

Pré-requisitos (uma vez):
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install transformers sentencepiece
    ollama pull nomic-embed-text
    (o modelo de NLI, ~700 MB, é baixado sozinho na primeira execução)

Exemplos:
    python repro/52_fidelidade_reescritas.py                      # as três checagens (~30–60 min na CPU)
    python repro/52_fidelidade_reescritas.py --sem-nli            # só similaridade e entidades (minutos)
    python repro/52_fidelidade_reescritas.py --avaliar            # depois de preencher a amostra humana
Depois: seção 2h do analise_extensoes.ipynb (análise de sensibilidade sem as reescritas suspeitas).
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged  # noqa: E402
from see.text import clean_light  # noqa: E402

EXP = config.RESULTS / "improvement"
VERSOES = ["parafrasear", "melhorar", "melhorar_contexto", "estruturar_contexto"]
COM_CARTAO = {"melhorar_contexto", "estruturar_contexto"}
SAIDA = EXP / "fidelidade_reescritas.csv"
AMOSTRA = EXP / "fidelidade_amostra_humana.xlsx"
CACHE_NLI = EXP / "fidelidade_nli_cache.jsonl"

# ----------------------------------------------------------------------------------------------- textos
CAMPOS_MODELO = ["Type", "Goal", "Scope", "Affected components", "Technologies", "External dependencies",
                 "Acceptance criteria", "Open questions"]
_ROTULO = re.compile(r"(?im)^[\s*#>\-]*(" + "|".join(re.escape(c) for c in CAMPOS_MODELO) + r")\s*\**\s*:\s*")


def limpa_reescrita(texto: str, versao: str) -> str:
    """Tira da versão estruturada o que não é afirmação: rótulos dos campos, 'Not stated' e as dúvidas em aberto."""
    if versao != "estruturar_contexto":
        return texto
    linhas = []
    pular = False
    for ln in (texto or "").splitlines():
        m = _ROTULO.match(ln)
        if m:
            pular = m.group(1).lower() == "open questions"
            ln = ln[m.end():]
        if pular:
            continue
        if re.fullmatch(r"\W*not stated\W*", ln.strip(), re.I):
            continue
        linhas.append(ln)
    return "\n".join(linhas)


def frases(texto: str) -> list[str]:
    """Frases-afirmação: corta em . ! ? e quebras de linha; descarta perguntas e pedaços com menos de 4 palavras."""
    out = []
    for pedaco in re.split(r"(?<=[.!?])\s+|\n+", texto or ""):
        s = re.sub(r"^[\s*\-•\d.)]+", "", pedaco).strip()
        if len(s.split()) >= 4 and not s.endswith("?"):
            out.append(s)
    return out


def trechos(texto: str, janela=3, passo=2) -> list[str]:
    """Premissas do NLI: janelas de 3 frases seguidas do original (o modelo vê até ~512 tokens por par)."""
    fr = [s for s in re.split(r"(?<=[.!?])\s+|\n+", texto or "") if s.strip()]
    if not fr:
        return []
    if len(fr) <= janela:
        return [" ".join(fr)]
    return [" ".join(fr[i:i + janela]) for i in range(0, len(fr) - janela + passo, passo) if fr[i:i + janela]]


# ----------------------------------------------------------------------------------------------- entidades
_ENT = [
    re.compile(r"https?://\S+"),
    re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b"),                                                   # chave de issue
    re.compile(r"[\w./\\-]+\.(?:java|js|ts|tsx|jsx|py|php|xml|json|ya?ml|css|scss|less|html?|md|sql|sh|go|rb|cpp|"
               r"c|h|kt|swift|gradle|properties|txt|log|jar|war|zip|tar|gz|ini|conf|cfg|vue|rs|cs)\b", re.I),
    re.compile(r"(?:/[\w.-]+){2,}|[\w.-]+(?:/[\w.-]+){2,}"),                                # caminhos (2+ barras)
    re.compile(r"\bv?\d+(?:\.\d+)+\b"),                                                      # versões 1.2.3
    re.compile(r"\b\d{2,}\b"),                                                               # números com 2+ dígitos
    re.compile(r"\b[a-z]+(?:[A-Z][a-z0-9]+)+\b"),                                            # camelCase
    re.compile(r"\b[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+\b"),                                    # PascalCase
    re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*_[a-zA-Z0-9_]+\b"),                                    # snake_case
    re.compile(r"\b[a-zA-Z_]\w+(?:\.[a-zA-Z_]\w+)+(?:\(\))?"),                                 # a.b.c, metodo()
]
_NAO_ENT = {"e.g", "i.e", "etc"}


def entidades(texto: str) -> set[str]:
    out = set()
    for rx in _ENT:
        for m in rx.findall(texto or ""):
            t = m.strip(".,;:()[]{}'\"").lower()
            if len(t) >= 2 and t not in _NAO_ENT:
                out.add(t)
    return out


# ----------------------------------------------------------------------------------------------- dados
def carrega_reescritas(modelo: str) -> pd.DataFrame:
    rows = []
    for v in VERSOES:
        arqs = sorted(EXP.glob(f"rewrites_{v}_{modelo}_*.jsonl"), key=lambda f: f.stat().st_mtime)
        if not arqs:
            print(f"   (sem reescritas '{v}' do {modelo}; fica de fora)")
            continue
        vistos = {}
        for ln in arqs[-1].read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                if r.get("ok"):
                    vistos[r["issue_id"]] = r
        rows += [{"issue_id": i, "versao": v, "titulo": r.get("title", ""), "descricao": r.get("description", ""),
                  "arquivo": arqs[-1].name} for i, r in vistos.items()]
        print(f"   {v}: {len(vistos)} reescritas válidas ({arqs[-1].name})")
    return pd.DataFrame(rows)


def originais_e_cartoes(ids: set):
    orig = {}
    for ds in config.DATASETS_LIMPOS:
        d = load_merged(ds)
        d = d[d.issue_id.isin(ids)]
        for r in d.itertuples():
            orig[r.issue_id] = {"dataset": ds, "project": str(r.project), "titulo": clean_light(r.title),
                                "descricao": clean_light(r.description or "")}
    cards = {}
    path = config.DATA_PROCESSED / "project_cards.csv"
    if path.exists():
        c = pd.read_csv(path, dtype={"project": str}).fillna("")
        cards = {(r.dataset, r.project): (r.cartao or r.cartao_bruto) for r in c.itertuples()}
    return orig, cards


# ----------------------------------------------------------------------------------------------- similaridade
def similaridade(pares: list[tuple[str, str]], modelo: str, url: str) -> list[float]:
    import requests

    def emb(textos):
        out = []
        for i in range(0, len(textos), 32):
            r = requests.post(f"{url}/api/embed", json={"model": modelo,
                              "input": ["clustering: " + t[:2000] for t in textos[i:i + 32]]}, timeout=600)
            if r.status_code == 404:
                sys.exit(f"Modelo de embeddings ausente. Rode: ollama pull {modelo}")
            r.raise_for_status()
            out += r.json()["embeddings"]
        m = np.asarray(out, dtype=float)
        return m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
    a = emb([p[0] for p in pares])
    b = emb([p[1] for p in pares])
    return list((a * b).sum(1))


# ----------------------------------------------------------------------------------------------- NLI
class NLI:
    def __init__(self, modelo: str, lote: int = 16):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            sys.exit("Falta o torch/transformers. Rode:\n  pip install torch --index-url "
                     "https://download.pytorch.org/whl/cpu\n  pip install transformers sentencepiece\n"
                     "(ou rode com --sem-nli)")
        print(f"   carregando o modelo de NLI {modelo} (a primeira vez baixa ~700 MB) ...", flush=True)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(modelo)
        self.mod = AutoModelForSequenceClassification.from_pretrained(modelo).eval()
        nomes = {i: n.lower() for i, n in self.mod.config.id2label.items()}
        self.i_ent = next(i for i, n in nomes.items() if n.startswith("entail"))
        self.i_con = next(i for i, n in nomes.items() if n.startswith("contra"))
        self.lote = lote
        self.cache = {}
        if CACHE_NLI.exists():
            for ln in CACHE_NLI.read_text(encoding="utf-8").splitlines():
                if ln.strip():
                    r = json.loads(ln); self.cache[r["k"]] = (r["e"], r["c"])
        self.f = open(CACHE_NLI, "a", encoding="utf-8")

    def probs(self, pares: list[tuple[str, str]]) -> list[tuple[float, float]]:
        """(P(entailment), P(contradiction)) para cada par (premissa, hipótese), com cache."""
        chaves = [hashlib.sha1((p + "\x1f" + h).encode()).hexdigest() for p, h in pares]
        faltam = [(k, p, h) for k, (p, h) in zip(chaves, pares) if k not in self.cache]
        for i in range(0, len(faltam), self.lote):
            bloco = faltam[i:i + self.lote]
            enc = self.tok([p for _, p, _ in bloco], [h for _, _, h in bloco], truncation="only_first",
                           max_length=512, padding=True, return_tensors="pt")
            with self.torch.no_grad():
                pr = self.torch.softmax(self.mod(**enc).logits, dim=-1).numpy()
            for (k, _, _), row in zip(bloco, pr):
                e, c = float(row[self.i_ent]), float(row[self.i_con])
                self.cache[k] = (e, c)
                self.f.write(json.dumps({"k": k, "e": round(e, 4), "c": round(c, 4)}) + "\n")
            self.f.flush()
        return [self.cache[k] for k in chaves]


def avalia_frases(nli: NLI, premissas: list[str], hipoteses: list[str], limiar=0.5):
    """Para cada frase da reescrita: sustentada / contradiz / nova (o melhor trecho do original decide)."""
    if not hipoteses or not premissas:
        return []
    pares = [(p, h) for h in hipoteses for p in premissas]
    pr = nli.probs(pares)
    out, k = [], 0
    for h in hipoteses:
        bloco = pr[k:k + len(premissas)]; k += len(premissas)
        e = max(x[0] for x in bloco); c = max(x[1] for x in bloco)
        out.append(("sustentada" if e >= limiar else "contradiz" if c >= limiar else "nova", e, c, h))
    return out


# ----------------------------------------------------------------------------------------------- principal
def roda(args):
    print("lendo as reescritas ...")
    R = carrega_reescritas(args.modelo_reescrita)
    if R.empty:
        sys.exit("Nenhuma reescrita encontrada em results/improvement/. Rode antes as seções 9 e 12 do analise_jev.ipynb.")
    orig, cards = originais_e_cartoes(set(R.issue_id))
    R = R[R.issue_id.isin(orig)].reset_index(drop=True)
    R["dataset"] = R.issue_id.map(lambda i: orig[i]["dataset"])
    R["project"] = R.issue_id.map(lambda i: orig[i]["project"])
    R["texto_orig"] = R.issue_id.map(lambda i: (orig[i]["titulo"] + ". " + orig[i]["descricao"]).strip())
    R["texto_reesc"] = [(t + ". " + limpa_reescrita(d, v)).strip() for t, d, v in zip(R.titulo, R.descricao, R.versao)]
    R["cartao"] = [cards.get((config.base_dataset(ds), p), "") if v in COM_CARTAO else ""
                   for ds, p, v in zip(R.dataset, R.project, R.versao)]

    # 2. entidades
    # uma entidade "existe" do outro lado se aparece no texto dele (sem diferenciar maiúsculas):
    # "CrashPlan" na reescrita não é invenção se o original diz "Crashplan"
    # comparação só por letras e dígitos ("node.js" = "nodejs") e aceitando plural ("exceptions" = "exception")
    norm = lambda t: re.sub(r"[^a-z0-9]", "", (t or "").lower())
    presente = lambda e, t: (lambda k: k in t or (k.endswith("s") and k[:-1] in t))(norm(e))
    eo = [entidades(t) for t in R.texto_orig]
    er = [entidades(t) for t in R.texto_reesc]
    to, tr, tc = [norm(t) for t in R.texto_orig], [norm(t) for t in R.texto_reesc], [norm(t) for t in R.cartao]
    perd = [sorted(e for e in a if not presente(e, t)) for a, t in zip(eo, tr)]
    inv = [sorted(e for e in b if not presente(e, o) and not presente(e, c)) for b, o, c in zip(er, to, tc)]
    R["ent_original"] = [len(a) for a in eo]
    R["ent_preservadas_pct"] = [100 * (len(a) - len(p)) / len(a) if a else np.nan for a, p in zip(eo, perd)]
    R["ent_inventadas_n"] = [len(x) for x in inv]
    R["ent_inventadas"] = [" | ".join(x[:10]) for x in inv]
    R["ent_perdidas"] = [" | ".join(x[:10]) for x in perd]

    # 1. similaridade
    if args.sem_similaridade:
        R["similaridade"] = np.nan
    else:
        print("similaridade (Ollama) ...", flush=True)
        R["similaridade"] = similaridade(list(zip(R.texto_orig, R.texto_reesc)), args.modelo_embed, args.ollama)

    # 3. NLI
    if args.sem_nli:
        for c in ["n_frases", "frac_sustentada", "frac_nova", "frac_contradiz", "frases_nao_sustentadas"]:
            R[c] = np.nan
    else:
        nli = NLI(args.modelo_nli, args.lote)
        res = []
        for n, r in enumerate(R.itertuples(), 1):
            prem = trechos(r.texto_orig) + (trechos(r.cartao) if r.cartao else [])
            av = avalia_frases(nli, prem, frases(r.texto_reesc), args.limiar_entail)
            tot = max(1, len(av))
            res.append({"n_frases": len(av),
                        "frac_sustentada": sum(a[0] == "sustentada" for a in av) / tot if av else np.nan,
                        "frac_nova": sum(a[0] == "nova" for a in av) / tot if av else np.nan,
                        "frac_contradiz": sum(a[0] == "contradiz" for a in av) / tot if av else np.nan,
                        "frases_nao_sustentadas": " || ".join(a[3] for a in av if a[0] != "sustentada")[:1500]})
            if n % 50 == 0 or n == len(R):
                print(f"   NLI {n}/{len(R)}", flush=True)
        R = pd.concat([R, pd.DataFrame(res)], axis=1)

    nao_sust = 1 - R.frac_sustentada
    R["suspeita"] = (R.ent_inventadas_n >= 1) | (nao_sust > args.limiar_nli).fillna(False)
    R["motivo"] = ["; ".join(m for m, ok in [("entidade inventada", a >= 1),
                                             (f"{100 * b:.0f}% das frases não sustentadas", pd.notna(b) and b > args.limiar_nli)] if ok)
                   for a, b in zip(R.ent_inventadas_n, nao_sust)]
    cols = ["issue_id", "dataset", "project", "versao", "suspeita", "motivo", "similaridade", "n_frases",
            "frac_sustentada", "frac_nova", "frac_contradiz", "ent_original", "ent_preservadas_pct",
            "ent_inventadas_n", "ent_inventadas", "ent_perdidas", "frases_nao_sustentadas", "texto_orig", "texto_reesc"]
    R[cols].to_csv(SAIDA, index=False)

    resumo = (R.groupby(["dataset", "versao"])
                .agg(reescritas=("issue_id", "size"), similaridade_mediana=("similaridade", "median"),
                     frases_sustentadas_pct=("frac_sustentada", lambda s: 100 * s.mean()),
                     com_contradicao_pct=("frac_contradiz", lambda s: 100 * (s > 0).mean()),
                     entidades_preservadas_pct=("ent_preservadas_pct", "mean"),
                     com_entidade_inventada_pct=("ent_inventadas_n", lambda s: 100 * (s >= 1).mean()),
                     suspeitas_pct=("suspeita", lambda s: 100 * s.mean()))
                .reindex(VERSOES, level="versao").round(1))
    config.TABLES.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(config.TABLES / "fidelidade_resumo.csv")
    with pd.option_context("display.width", 200):
        print("\n" + resumo.to_string())
    print(f"\nsalvo: {SAIDA.relative_to(config.ROOT)} e results/tables/fidelidade_resumo.csv")
    print("Leitura: o parafrasear (mesma informação) é a régua; uma versão bem abaixo dele em frases sustentadas")
    print("ou com muitas entidades inventadas está acrescentando conteúdo que não estava na issue.")

    if not AMOSTRA.exists() or args.nova_amostra:
        rng = np.random.default_rng(config.SEED)
        partes = []
        for v, g in R.groupby("versao"):
            sus, ok = g[g.suspeita], g[~g.suspeita]
            k = min(5, len(sus))
            partes.append(sus.iloc[rng.permutation(len(sus))[:k]])
            partes.append(ok.iloc[rng.permutation(len(ok))[:10 - k]])
        A = pd.concat(partes).sample(frac=1, random_state=config.SEED)    # embaralha: quem rotula não vê o grupo
        A = A[["issue_id", "versao", "texto_orig", "texto_reesc"]].assign(
            **{"inventou_algo (sim/nao)": "", "perdeu_informacao (sim/nao)": "", "observacao": ""})
        instr = pd.DataFrame({"Como rotular": [
            "Compare o texto original com a reescrita, uma linha por vez.",
            "inventou_algo = sim se a reescrita afirma algo que NÃO está no original (requisito, componente, número, comportamento, critério de aceite). Nas versões *_contexto, informação vinda do cartão do projeto também conta como invenção se não for sobre esta issue.",
            "perdeu_informacao = sim se um detalhe técnico do original sumiu (versão, arquivo, passo para reproduzir, mensagem de erro).",
            "Não olhe as métricas antes de rotular. Depois rode: python repro/52_fidelidade_reescritas.py --avaliar"]})
        with pd.ExcelWriter(AMOSTRA) as w:
            instr.to_excel(w, sheet_name="instrucoes", index=False)
            A.to_excel(w, sheet_name="rotular", index=False)
        print(f"amostra para rotulagem humana: {AMOSTRA.relative_to(config.ROOT)} ({len(A)} reescritas)")


def avaliar():
    """Compara as checagens automáticas com a sua rotulagem da amostra."""
    if not AMOSTRA.exists() or not SAIDA.exists():
        sys.exit("Rode antes o script sem --avaliar e preencha a amostra.")
    A = pd.read_excel(AMOSTRA, sheet_name="rotular", dtype=str).fillna("")
    col = "inventou_algo (sim/nao)"
    A = A[A[col].str.strip().str.lower().isin(["sim", "nao", "não"])]
    if A.empty:
        sys.exit("Nenhuma linha rotulada ainda (coluna 'inventou_algo' com sim/nao).")
    R = pd.read_csv(SAIDA, dtype={"issue_id": str})
    M = A.merge(R, on=["issue_id", "versao"], how="left")
    y = M[col].str.strip().str.lower().eq("sim").astype(int)
    from sklearn.metrics import cohen_kappa_score, roc_auc_score
    print(f"{len(M)} reescritas rotuladas; {int(y.sum())} com invenção segundo você\n")
    pred = M.suspeita.astype(str).str.lower().eq("true").astype(int)
    vp, fp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    print(f"marcação automática 'suspeita': precisão {vp / max(1, vp + fp):.2f}, recall {vp / max(1, vp + fn):.2f}, "
          f"kappa {cohen_kappa_score(y, pred):.2f}")
    if y.nunique() == 2:
        for nome, s in [("1 − frases sustentadas (NLI)", 1 - M.frac_sustentada), ("entidades inventadas", M.ent_inventadas_n),
                        ("1 − similaridade", 1 - M.similaridade)]:
            s = pd.to_numeric(s, errors="coerce")
            if s.notna().sum() == len(s):
                print(f"AUC {nome}: {roc_auc_score(y, s):.2f}  (0,5 = acaso; 1 = separa perfeitamente)")
    print("\nSe a precisão ou o recall ficarem baixos, ajuste --limiar-nli e rode de novo (sem --nova-amostra).")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo-reescrita", default="llama3", help="modelo que fez as reescritas (nome nos arquivos)")
    ap.add_argument("--modelo-nli", default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
    ap.add_argument("--modelo-embed", default="nomic-embed-text")
    ap.add_argument("--ollama", default="http://localhost:11434")
    ap.add_argument("--limiar-entail", type=float, default=0.5, help="P(entailment) para a frase contar como sustentada")
    ap.add_argument("--limiar-nli", type=float, default=0.5, help="fração de frases não sustentadas acima da qual é suspeita")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--sem-nli", action="store_true")
    ap.add_argument("--sem-similaridade", action="store_true")
    ap.add_argument("--nova-amostra", action="store_true", help="sobrescreve a amostra humana (cuidado: perde a rotulagem)")
    ap.add_argument("--avaliar", action="store_true", help="compara as checagens com a amostra rotulada")
    args = ap.parse_args()
    avaliar() if args.avaliar else roda(args)


if __name__ == "__main__":
    main()
