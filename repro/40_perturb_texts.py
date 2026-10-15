"""Etapa de robustez — gera versões perturbadas do subconjunto de teste (Algoritmo 1).

    python repro/40_perturb_texts.py

Para cada seed (42, 123, 2025) e cada issue de `within_eval`:
  1. troca até 10% das palavras por sinônimos do WordNet com a mesma classe gramatical;
  2. ruído de caractere: cada caractere tem chance `--char-p` de ser apagado ou trocado
     (o Algoritmo 1 usa 0,05 por caractere; o texto do artigo fala em ruído "mínimo");
  3. remove cada sinal de pontuação com probabilidade 0,05.

Saída: data/perturbed/{dataset}_s{seed}.csv (issue_id, title, description,
text_light, text_full). Os scripts de ML, Transformers e LLM usam esses arquivos
com --protocol perturb, e os resultados ficam em results/predictions/*/perturb_s{seed}/.

Requer os dados do NLTK: python -m nltk.downloader wordnet omw-1.4 averaged_perceptron_tagger_eng
"""
import argparse
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_merged  # noqa: E402
from see.text import clean_light  # noqa: E402

PERTURBED = config.ROOT / "data" / "perturbed"
LETTERS = string.ascii_lowercase


def wordnet_pos(tag: str):
    from nltk.corpus import wordnet as wn
    return {"N": wn.NOUN, "V": wn.VERB, "J": wn.ADJ, "R": wn.ADV}.get(tag[:1])


def synonym_swap(text: str, rng, max_rate: float) -> str:
    import nltk
    from nltk.corpus import wordnet as wn
    words = text.split()
    if not words:
        return text
    tags = nltk.pos_tag(words)
    n_swap = int(np.floor(max_rate * len(words)))
    if n_swap == 0:
        return text
    for i in rng.choice(len(words), n_swap, replace=False):
        word, tag = tags[i]
        pos = wordnet_pos(tag)
        core = word.strip(string.punctuation)
        if pos is None or not core.isalpha():
            continue
        lemmas = sorted({l.name().replace("_", " ") for s in wn.synsets(core.lower(), pos=pos)
                         for l in s.lemmas()} - {core.lower()})
        if lemmas:
            syn = lemmas[rng.integers(len(lemmas))]
            if core[:1].isupper():
                syn = syn[:1].upper() + syn[1:]
            words[i] = word.replace(core, syn)
    return " ".join(words)


def char_noise(text: str, rng, p: float) -> str:
    out = []
    for ch in text:
        if not ch.isspace() and rng.random() < p:
            if rng.random() < 0.5:
                continue                                      # apaga
            out.append(LETTERS[rng.integers(len(LETTERS))])   # troca
        else:
            out.append(ch)
    return "".join(out)


def drop_punct(text: str, rng, p: float) -> str:
    return "".join(ch for ch in text if not (ch in string.punctuation and rng.random() < p))


def perturb(text: str, rng, args) -> str:
    t = synonym_swap(text, rng, args.max_rate)
    t = char_noise(t, rng, args.char_p)
    return drop_punct(t, rng, args.punct_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", nargs="+", default=config.DATASETS, choices=config.ALL_DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=config.ROBUSTNESS_SEEDS)
    ap.add_argument("--max-rate", type=float, default=0.10)
    ap.add_argument("--char-p", type=float, default=0.05)
    ap.add_argument("--punct-p", type=float, default=0.05)
    args = ap.parse_args()
    PERTURBED.mkdir(parents=True, exist_ok=True)
    for dataset in args.dataset:
        df = load_merged(dataset)
        sub = df[df.within_eval == 1]
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            title = [perturb(clean_light(t), rng, args) for t in sub.title]
            desc = [perturb(clean_light(d), rng, args) for d in sub.description]
            out = pd.DataFrame({"issue_id": sub.issue_id.values, "title": title, "description": desc})
            out["text_light"] = (out.title + ". " + out.description).str.strip(" .")
            out["text_full"] = out.text_light.str.lower()
            path = PERTURBED / f"{dataset}_s{seed}.csv"
            out.to_csv(path, index=False)
            print(f"{dataset} seed={seed}: {len(out)} textos -> {path.relative_to(config.ROOT)}")
            print(f"   antes : {sub.text_light.iloc[0][:110]}")
            print(f"   depois: {out.text_light.iloc[0][:110]}")


if __name__ == "__main__":
    main()
