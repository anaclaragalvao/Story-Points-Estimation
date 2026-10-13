"""EXTENSÃO (não faz parte do artigo) — características dos repositórios do NEO.

    python repro/04_repo_features.py

O NEO traz, além das stories, um JSON por projeto com os metadados da API do
GitLab (pasta util/json head do repositório giseldo/neodataset). Esta etapa
transforma esses metadados em features numéricas por issue.

Saída: data/processed/neo_repo_features.csv, uma linha por issue:

  log_stars, log_forks, log_open_issues   popularidade e tamanho da comunidade (log1p)
  n_topics                                número de tópicos declarados no repositório
  project_age_days                        dias desde a primeira issue do projeto até esta issue
  org_gitlab, org_minds, org_fpc          organização dona do repositório (o resto = 0)

Cuidados (também no README):
  1. Os metadados são um retrato de junho de 2023 (data da mineração). Estrelas,
     forks e issues abertas são posteriores à maioria das issues; não são o que o
     time "sabia" ao estimar. Só project_age_days usa apenas informação do passado.
  2. Exceto a idade, as features são constantes dentro de cada projeto. No
     protocolo within elas funcionam como um identificador do projeto (o que
     ajuda, porque cada time usa sua própria escala de SP). No cross-project e no
     LOPO há só ~30 valores distintos para aprender, e o risco de sobreajuste é alto.
  3. O Agile não tem nada equivalente; a extensão vale só para o NEO.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_processed  # noqa: E402

REPO_FEATURES = ["log_stars", "log_forks", "log_open_issues", "n_topics", "project_age_days",
                 "org_gitlab", "org_minds", "org_fpc"]


def load_repo_table() -> pd.DataFrame:
    rows = []
    for pid in config.NEO_PROJECT_IDS:
        path = config.DATA_RAW / "neo_repos" / f"{pid}.json"
        if not path.exists():
            sys.exit(f"{path} não existe. Rode antes: python repro/01_download_data.py")
        d = json.loads(path.read_text(encoding="utf-8"))
        top = d["namespace"]["full_path"].split("/")[0].lower()
        rows.append({
            "project": pid,
            "repo_name": d["name_with_namespace"],
            "repo_description": (d.get("description") or "").strip(),
            "repo_created_at": pd.to_datetime(d["created_at"], utc=True),
            "stars": d.get("star_count"),
            "forks": d.get("forks_count"),
            "open_issues": d.get("open_issues_count"),
            "n_topics": len(d.get("topics") or []),
            "org_gitlab": int(top in ("gitlab-org", "gitlab-com")),
            "org_minds": int(top == "minds"),
            "org_fpc": int(top == "freepascal.org"),
        })
    return pd.DataFrame(rows)


def main():
    repos = load_repo_table()
    # forks ausentes em 2 projetos: imputamos a mediana (registrado aqui)
    for col in ["stars", "forks", "open_issues"]:
        n_missing = int(repos[col].isna().sum())
        if n_missing:
            print(f"  {col}: {n_missing} projeto(s) sem valor -> mediana")
        repos[col] = repos[col].fillna(repos[col].median())
        repos[f"log_{col}"] = np.log1p(repos[col].astype(float))

    issues = load_processed("neo")[["issue_id", "project", "created"]]
    df = issues.merge(repos, on="project", how="left", validate="many_to_one")
    # Idade = dias desde a PRIMEIRA issue do projeto no dataset. A data de criação
    # do repositório no GitLab não serve: KiCad, duplicity e FPC migraram para o
    # GitLab trazendo issues mais antigas que o próprio repositório.
    created = pd.to_datetime(df["created"], utc=True, format="mixed")
    first = created.groupby(df["project"]).transform("min")
    df["project_age_days"] = (created - first).dt.days
    out = df[["issue_id", "project", "repo_name", "repo_description", *REPO_FEATURES]]
    path = config.DATA_PROCESSED / "neo_repo_features.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} issues, {out.project.nunique()} projetos -> {path.relative_to(config.ROOT)}")
    print(repos[["project", "repo_name", "stars", "forks", "open_issues", "n_topics"]]
          .sort_values("stars", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()
