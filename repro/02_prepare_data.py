"""Etapa 2 — converte os dois datasets para um formato único.

    python repro/02_prepare_data.py

Saída: data/processed/{agile,neo}.csv com as colunas
    issue_id, dataset, project, order, created, title, description, sp,
    text_light  (título + descrição sem código/URLs/markup — Transformers e LLMs)
    text_full   (text_light em minúsculas — TF-IDF do ML clássico)

`order` é a posição da issue dentro do projeto no arquivo original. No NEO
também guardamos `created`, que permite um split temporal como variação.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.text import clean_full, clean_light  # noqa: E402

COLS = ["issue_id", "dataset", "project", "order", "created", "title", "description", "sp",
        "text_light", "text_full"]


def finish(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df["dataset"] = name
    df["title"] = df["title"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["text_light"] = (df["title"].map(clean_light) + ". " + df["description"].map(clean_light)).str.strip(" .")
    df["text_full"] = df["text_light"].str.lower()
    df["order"] = df.groupby("project").cumcount()
    return df[COLS]


def prepare_agile() -> pd.DataFrame:
    raw = pd.read_csv(config.DATA_RAW / "agile" / "data.csv", low_memory=False)
    df = pd.DataFrame({
        "project": raw["project"].astype(str),
        "title": raw["title"],
        "description": raw["user_story"],
        "sp": raw["point"].astype(float),
        "created": pd.NA,  # o arquivo do mrthlinh não traz datas
    })
    # O arquivo não tem chave de issue: criamos uma estável a partir da ordem.
    df["issue_id"] = df["project"] + "-" + df.groupby("project").cumcount().astype(str).str.zfill(5)
    return finish(df, "agile")


def prepare_neo() -> pd.DataFrame:
    parts = []
    for pid in config.NEO_PROJECT_IDS:
        d = pd.read_csv(config.DATA_RAW / "neo" / f"{pid}.csv", low_memory=False)
        d["project"] = pid
        parts.append(d)
    raw = pd.concat(parts, ignore_index=True)
    n_all = len(raw)
    raw = raw[~raw["project"].isin(config.NEO_EXCLUDED_PROJECTS)]
    n_subset = len(raw)
    n_nan = int(raw["storypoints"].isna().sum())
    raw = raw.dropna(subset=["storypoints"])
    print(f"NEO: {n_all} issues no total -> {n_subset} sem o projeto GitLab "
          f"-> {len(raw)} após remover {n_nan} sem story point")
    df = pd.DataFrame({
        "issue_id": raw["project"] + "-" + raw["issuekey"].astype(str),
        "project": raw["project"],
        "title": raw["title"],
        "description": raw["description"],
        "sp": raw["storypoints"].astype(float),
        "created": raw["created"],
    })
    assert not df["issue_id"].duplicated().any()
    return finish(df, "neo")


def describe(df: pd.DataFrame):
    words = df["text_light"].str.split().str.len()
    print(f"  {df['dataset'].iloc[0]}: {len(df)} issues, {df['project'].nunique()} projetos | "
          f"SP média={df.sp.mean():.2f} mediana={df.sp.median():.1f} desvio={df.sp.std():.2f} | "
          f"palavras médias={words.mean():.1f}")


def main():
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, fn in [("agile", prepare_agile), ("neo", prepare_neo)]:
        df = fn()
        df.to_csv(config.DATA_PROCESSED / f"{name}.csv", index=False)
        describe(df)
    print(f"Arquivos em {config.DATA_PROCESSED}")


if __name__ == "__main__":
    main()
