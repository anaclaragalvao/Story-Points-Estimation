"""EXTENSÃO — tipo da issue e calibração do time por tipo (usados pelo 30, pelo 32 e pelo analise_extensoes).

tipo da issue
    Agile: campo Type do Jira na data da estimativa (data/processed/agile_tawos_meta.csv, do 09_tawos_metadados.py)
    NEO:   labels do GitLab (data/raw/neo_json), pela mesma regra da seção 5 do analise_jev.ipynb;
           sem label de tipo -> "sem_tipo"; mais de um -> "misto"

mediana projeto × tipo
    mediana do SP no treino do mesmo projeto e tipo, se houver pelo menos MIN_GRUPO issues;
    senão, mediana do projeto (é o baseline "ProjectMedian×tipo" do 32_ml_metadados.py)
"""
import json

import numpy as np
import pandas as pd

from . import config

MIN_GRUPO = 5


def tipo_da_issue(labels):
    """bug / feature / maintenance / documentation / misto / sem_tipo, a partir dos labels do GitLab."""
    L = [l.lower().strip() for l in labels]
    tags = set()
    if any(l in ("type::bug", "bug", "type::regression", "regression", "kind::bug") for l in L):
        tags.add("bug")
    if any(l in ("type::feature", "feature", "feature request", "feature-request", "type::user feature",
                 "feature::addition", "feature::enhancement", "enhancement", "type::enhancement", "kind::feature")
           for l in L):
        tags.add("feature")
    if any(l in ("type::maintenance", "type::chore", "maintenance::refactor", "backlog::technicaldebt",
                 "technical debt", "tech debt", "refactor", "refactoring", "type::tooling", "kind::tech debt")
           for l in L):
        tags.add("maintenance")
    if any(l in ("documentation", "type::documentation", "docs") for l in L):
        tags.add("documentation")
    return tags.pop() if len(tags) == 1 else ("misto" if tags else "sem_tipo")


def tipos(dataset: str) -> pd.Series:
    """issue_id -> tipo. No Agile, só as issues encontradas no TAWOS (as outras ficam de fora da Series)."""
    if config.base_dataset(dataset) == "agile":
        path = config.DATA_PROCESSED / "agile_tawos_meta.csv"
        if not path.exists():
            raise FileNotFoundError("Falta data/processed/agile_tawos_meta.csv: rode antes o 09_tawos_metadados.py")
        t = pd.read_csv(path, dtype={"issue_id": str}).fillna("")
        t = t[(t.tipo != "") | (t.sp_tawos != "")]
        return pd.Series([str(x).strip().lower() or "sem_tipo" for x in t.tipo], index=t.issue_id, name="tipo")
    pasta = config.DATA_RAW / "neo_json"
    if not pasta.exists():
        raise FileNotFoundError(f"Falta {pasta}: rode a seção 5 do analise_jev.ipynb (ela baixa os JSON)")
    out = {}
    for f in sorted(pasta.glob("*.json")):
        for it in json.loads(f.read_text(encoding="utf-8")):
            out[f"{f.stem}-{it['id']}"] = tipo_da_issue(it.get("labels") or [])
    return pd.Series(out, name="tipo")


def mediana_projeto_tipo(train: pd.DataFrame, alvo: pd.DataFrame, tipo: pd.Series) -> np.ndarray:
    """Baseline mediana projeto × tipo para as issues de `alvo` (com fallback para a mediana do projeto)."""
    t_tr = train.issue_id.map(tipo)
    med_proj = train.groupby("project").sp.median()
    g = train.assign(tipo=t_tr.values).dropna(subset=["tipo"]).groupby(["project", "tipo"]).sp
    med, n = g.median(), g.size()
    geral = float(train.sp.median())
    out = []
    for p, t in zip(alvo.project, alvo.issue_id.map(tipo)):
        k = (p, t)
        out.append(med[k] if isinstance(t, str) and k in med.index and n[k] >= MIN_GRUPO else med_proj.get(p, geral))
    return np.asarray(out, dtype=float)
