"""Etapa 3 — gera TODOS os splits uma única vez e salva em arquivo.

    python repro/03_make_splits.py

Todas as famílias de modelos (ML, Transformers, LLMs) leem o mesmo arquivo, então
comparam exatamente as mesmas issues — mesmo rodando em computadores diferentes.

Saída: data/splits/{dataset}_splits.csv com uma linha por issue:

    within        train / val / test   80/10/10 dentro de cada projeto, estratificado por SP
    within_eval   1 = está nos 50% do test set usados na avaliação principal (~5% do total)
    cross         train / val / test   80/10/10 dos PROJETOS (projeto inteiro num split só)
    cross_eval    1 = amostra de ~2% do total, sorteada entre as issues dos projetos de teste
    lopo_eval     1 = uma das 30 (Agile) / 15 (NEO) amostras do projeto quando ele é deixado de fora
    fewshot_eval  1 = uma das 100 instâncias do experimento few-shot (tiradas do test cross-project)

"Estratificação adaptativa" (termo do artigo, sem definição): dentro de cada
projeto ordenamos as issues por SP e, a cada bloco de 10, sorteamos 8 treino /
1 validação / 1 teste. Assim as distribuições de SP ficam parecidas entre os
splits mesmo em projetos pequenos (o artigo mostra isso na Tabela 4).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_processed  # noqa: E402


def systematic_labels(sp: pd.Series, pattern: list, rng: np.random.Generator) -> pd.Series:
    """Ordena por SP (desempate aleatório) e sorteia o `pattern` em blocos consecutivos."""
    order = np.lexsort((rng.random(len(sp)), sp.to_numpy()))
    labels = np.empty(len(sp), dtype=object)
    k = len(pattern)
    for start in range(0, len(sp), k):
        block = order[start:start + k]
        labels[block] = rng.permutation(pattern)[: len(block)]
    return pd.Series(labels, index=sp.index)


def stratified_sample(sp: pd.Series, frac: float, rng: np.random.Generator) -> pd.Index:
    """Amostra estratificada por SP (amostragem sistemática sobre a lista ordenada)."""
    if len(sp) == 0:
        return sp.index
    k = max(1, round(1 / frac))
    pattern = [1] + [0] * (k - 1)
    picked = systematic_labels(sp, pattern, rng)
    return picked[picked == 1].index


def make_splits(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rng = np.random.default_rng(config.SEED)
    out = df[["issue_id", "project", "sp"]].copy()

    # ---------------- within-project ----------------
    pattern = ["train"] * 8 + ["val"] + ["test"]
    out["within"] = ""
    for _, idx in out.groupby("project").groups.items():
        out.loc[idx, "within"] = systematic_labels(out.loc[idx, "sp"], pattern, rng)
    test = out[out.within == "test"]
    out["within_eval"] = 0
    for _, g in test.groupby("project"):
        out.loc[stratified_sample(g["sp"], config.EVAL_SUBSET_FRAC, rng), "within_eval"] = 1

    # ---------------- cross-project ----------------
    projects = sorted(out["project"].unique())
    perm = list(rng.permutation(projects))
    n_test = max(1, round(config.TEST_FRAC * len(projects)))
    n_val = max(1, round(config.VAL_FRAC * len(projects)))
    proj_split = {p: "test" for p in perm[:n_test]}
    proj_split.update({p: "val" for p in perm[n_test:n_test + n_val]})
    proj_split.update({p: "train" for p in perm[n_test + n_val:]})
    out["cross"] = out["project"].map(proj_split)
    # O artigo avalia em "20% do test set", que corresponde a ~2% do dataset
    # (20% de um test de 10%). Mantemos esse tamanho (~2% do total), sorteado
    # entre as issues dos projetos de teste.
    out["cross_eval"] = 0
    cross_test = out[out.cross == "test"]
    target = round(config.CROSS_EVAL_FRAC * config.TEST_FRAC * len(out))
    frac = min(1.0, target / len(cross_test))
    out.loc[stratified_sample(cross_test["sp"], frac, rng), "cross_eval"] = 1

    # ---------------- LOPO ----------------
    out["lopo_eval"] = 0
    n_per = config.LOPO_SAMPLES[dataset]
    for _, g in out.groupby("project"):
        take = g.index if len(g) <= n_per else rng.choice(g.index, n_per, replace=False)
        out.loc[take, "lopo_eval"] = 1

    # ---------------- few-shot (cross-project) ----------------
    out["fewshot_eval"] = 0
    n_fs = min(config.FEWSHOT_N_INSTANCES, len(cross_test))
    out.loc[rng.choice(cross_test.index, n_fs, replace=False), "fewshot_eval"] = 1

    return out.drop(columns=["sp"])


def report(df: pd.DataFrame, sp: pd.DataFrame, dataset: str):
    m = df.merge(sp, on="issue_id")
    m["words"] = m["text_light"].str.split().str.len()

    def row(name, x):
        return (f"  {name:<26}{len(x):>7}{x.sp.mean():>9.2f}{x.sp.median():>8.1f}"
                f"{x.sp.std():>8.2f}{x.words.mean():>9.1f}")

    print(f"\n[{dataset}]  {'split':<26}{'issues':>7}{'mean SP':>9}{'med':>8}{'std':>8}{'words':>9}")
    for s in ["train", "val", "test"]:
        print(row(f"within {s}", m[m.within == s]))
    print(row("within test (subset aval.)", m[m.within_eval == 1]))
    print(row("cross test (subset aval.)", m[m.cross_eval == 1]))
    print(row("LOPO (todas as amostras)", m[m.lopo_eval == 1]))
    print(row("few-shot", m[m.fewshot_eval == 1]))
    test_projects = sorted(m.loc[m.cross == "test", "project"].unique())
    val_projects = sorted(m.loc[m.cross == "val", "project"].unique())
    print(f"  cross-project: test={test_projects}  val={val_projects}")


def main():
    config.SPLITS.mkdir(parents=True, exist_ok=True)
    for dataset in config.DATASETS:
        df = load_processed(dataset)
        splits = make_splits(df, dataset)
        splits.to_csv(config.SPLITS / f"{dataset}_splits.csv", index=False)
        report(df[["issue_id", "sp", "text_light"]], splits, dataset)
    print(f"\nSplits salvos em {config.SPLITS}")


if __name__ == "__main__":
    main()
