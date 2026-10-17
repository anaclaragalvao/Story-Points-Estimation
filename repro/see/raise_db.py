"""Leitura do banco Postgres do RAISE (github.com/aisepucrio/raise), usado só como extrator.

O RAISE clona um repositório do GitHub, percorre os commits com o PyDriller e grava
tudo em Postgres. As etapas 06 e 07 leem estas tabelas (nomes padrão do Django):

    github_githubcommit        sha, repository_name, message, date, insertions, deletions, merge
    github_githubmodifiedfile  commit_id, old_path, new_path, added_lines, deleted_lines
    github_githubmetadata      repository, description, readme

Conexão por variáveis de ambiente (nada de senha em arquivo):
    RAISE_DB_HOST (localhost)  RAISE_DB_PORT (5432)  RAISE_DB_NAME (raise)
    RAISE_DB_USER (postgres)   RAISE_DB_PASSWORD (perguntada se não estiver definida)
Os valores são os POSTGRES_* do .env do raise; o docker-compose dele expõe a porta 5432.

Cuidados do RAISE (versão 29d8406), que afetam o contexto:
  - grava os arquivos modificados com a chave (commit, nome do arquivo SEM a pasta):
    dois arquivos com o mesmo nome no mesmo commit (index.js, __init__.py) viram uma linha só;
  - o PyDriller não lista arquivos de commits de merge (devolve lista vazia);
  - repository_name fica como foi digitado na coleta; aqui tudo é comparado em minúsculas.
"""
import getpass
import os

import pandas as pd


def connect():
    try:
        import psycopg
    except ImportError as e:
        raise SystemExit("Falta o driver do Postgres: pip install -r repro/requirements-raise.txt") from e
    password = os.environ.get("RAISE_DB_PASSWORD")
    if password is None:
        password = getpass.getpass("Senha do Postgres do RAISE (POSTGRES_PASSWORD do .env do raise): ")
    return psycopg.connect(host=os.environ.get("RAISE_DB_HOST", "localhost"),
                           port=int(os.environ.get("RAISE_DB_PORT", "5432")),
                           dbname=os.environ.get("RAISE_DB_NAME", "raise"),
                           user=os.environ.get("RAISE_DB_USER", "postgres"),
                           password=password)


def _query(conn, sql: str, params) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [d.name for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def _lower(repos) -> list:
    return sorted({r.lower() for r in repos})


def load_commits(conn, repos) -> pd.DataFrame:
    """Uma linha por commit: sha, repo, message, date (UTC), insertions, deletions, merge."""
    df = _query(conn, """
        SELECT c.sha, lower(c.repository_name) AS repo, c.message, c.date,
               c.insertions, c.deletions, c.merge
        FROM github_githubcommit c
        WHERE lower(c.repository_name) = ANY(%s)""", (_lower(repos),))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["message"] = df["message"].fillna("")
    return df.drop_duplicates("sha")


def load_files(conn, repos) -> pd.DataFrame:
    """Uma linha por arquivo modificado: sha, repo, date (UTC), path, added_lines, deleted_lines."""
    df = _query(conn, """
        SELECT c.sha, lower(c.repository_name) AS repo, c.date,
               COALESCE(f.new_path, f.old_path) AS path, f.added_lines, f.deleted_lines
        FROM github_githubmodifiedfile f
        JOIN github_githubcommit c ON c.id = f.commit_id
        WHERE lower(c.repository_name) = ANY(%s)""", (_lower(repos),))
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["path"] = df["path"].fillna("").str.replace("\\", "/", regex=False)
    return df


def load_metadata(conn, repos) -> pd.DataFrame:
    """repo, description, readme. Retrato do momento da coleta (posterior às issues)."""
    return _query(conn, """
        SELECT lower(repository) AS repo, description, readme
        FROM github_githubmetadata
        WHERE lower(repository) = ANY(%s)""", (_lower(repos),))
