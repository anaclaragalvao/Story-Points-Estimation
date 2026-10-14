"""Notas gerais de qualidade a partir das respostas do juiz (Jev ou Laya).

indice_qualidade (0–100), índice composto e transparente: média simples de quatro
componentes, cada um levado para a escala 0–1:
    clareza do objetivo       goal_clarity / 2
    dá para estimar           estimability / 2
    critérios de aceitação    P(has_acceptance_criteria)
    sem ambiguidade           1 − P(is_ambiguous)
O tamanho do escopo e o tipo da issue NÃO entram: descrevem a issue, não a qualidade
do texto. Se faltar algum componente, a média usa os disponíveis.

nota_geral_juiz (0–100), avaliação holística pedida ao juiz (pergunta overall_quality,
5 níveis de 0 a 4), convertida para 0–100. É o tipo de julgamento em que o artigo do
Jev mostra o juiz mais fraco (prosa sem referência), por isso convém compará-la com o índice.

faixa: ruim (< 40), regular (40–70), boa (≥ 70). Os cortes são uma convenção nossa,
para facilitar a leitura; o índice contínuo é o que deve ir para as análises.
"""
import numpy as np
import pandas as pd

COMPONENTS = {
    "goal_clarity": lambda s: s / 2,
    "estimability": lambda s: s / 2,
    "has_acceptance_criteria": lambda s: s,
    "is_ambiguous": lambda s: 1 - s,
}
FAIXAS = [(0, 40, "ruim"), (40, 70, "regular"), (70, 100.0001, "boa")]


def add_quality_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parts = [fn(df[c].astype(float)) for c, fn in COMPONENTS.items() if c in df]
    if parts:
        df["indice_qualidade"] = (100 * pd.concat(parts, axis=1).mean(axis=1, skipna=True)).round(1)
        df["faixa"] = faixa(df["indice_qualidade"])
    if "overall_quality" in df:
        df["nota_geral_juiz"] = (100 * df["overall_quality"].astype(float) / 4).round(1)
    return df


def faixa(values: pd.Series) -> pd.Series:
    out = pd.Series(pd.NA, index=values.index, dtype="object")
    for lo, hi, name in FAIXAS:
        out[(values >= lo) & (values < hi)] = name
    return out


def summarize(df: pd.DataFrame) -> dict:
    """Resumo para o dataset inteiro."""
    out = {"n": len(df)}
    if "indice_qualidade" in df:
        q = df["indice_qualidade"]
        out.update({"indice_media": round(q.mean(), 1), "indice_mediana": round(q.median(), 1)})
        for _, _, name in FAIXAS:
            out[f"%_{name}"] = round(100 * (df["faixa"] == name).mean(), 1)
    if "nota_geral_juiz" in df:
        out.update({"nota_juiz_media": round(df["nota_geral_juiz"].mean(), 1),
                    "nota_juiz_mediana": round(df["nota_geral_juiz"].median(), 1)})
        if "indice_qualidade" in df and df[["indice_qualidade", "nota_geral_juiz"]].dropna().shape[0] > 2:
            from scipy.stats import spearmanr
            x = df[["indice_qualidade", "nota_geral_juiz"]].dropna()
            out["spearman_indice_x_nota_juiz"] = round(float(spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0]), 2)
    return out
