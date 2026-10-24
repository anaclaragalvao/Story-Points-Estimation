"""EXTENSÃO — metadados do Jira para o Agile, a partir do dataset TAWOS (Tawosi et al., MSR 2022).

O Agile (Choetkiertikul) só traz título, descrição e story points. O TAWOS é um banco MySQL com as issues
dos mesmos Jiras públicos, incluindo tipo, prioridade, componentes, pessoas, a data em que o SP foi dado
(Estimation_Date) e o histórico de mudanças de cada campo (Change_Log). Este script:

1. conecta no MySQL do TAWOS (o container Docker "tawos", porta 3307);
2. cruza as issues do Agile pela chave real do Jira (data/processed/agile_issuekeys.csv, do 05_agile_issuekeys.py);
3. reconstrói os campos COMO ESTAVAM NA DATA DA ESTIMATIVA, desfazendo as mudanças registradas depois dela
   (tipo, prioridade, componentes) e lendo os labels da última mudança antes dela; o responsável só é usado
   se não mudou depois da estimativa;
4. grava data/processed/agile_tawos_meta.csv, que o 32_ml_metadados.py usa para o Agile.

Depois de rodar, o MySQL não é mais necessário (pode parar o container: docker stop tawos).

Pré-requisitos:
    pip install pymysql
    docker start tawos        (se o container estiver parado)

Exemplo:
    python repro/09_tawos_metadados.py
    python repro/09_tawos_metadados.py --host 127.0.0.1 --port 3307 --user root --password tawos
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402

SAIDA = config.DATA_PROCESSED / "agile_tawos_meta.csv"
CAMPOS_LOG = ("issuetype", "priority", "component", "assignee", "labels")


def conecta(args):
    try:
        import pymysql
    except ImportError:
        sys.exit("Falta o pymysql: pip install pymysql")
    try:
        return pymysql.connect(host=args.host, port=args.port, user=args.user, password=args.password,
                               database=args.database, charset="utf8mb4")
    except Exception as e:
        sys.exit(f"Não conectou no MySQL ({e}).\nO container está rodando? docker start tawos")


def consulta(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def baixa(conn, projetos):
    marcas = ", ".join(["%s"] * len(projetos))
    issues = consulta(conn, f"""
        SELECT i.ID, i.Issue_Key, p.Project_Key, i.Type, i.Priority, i.Creation_Date, i.Estimation_Date,
               i.Story_Point, i.Story_Point_Changed_After_Estimation, i.Title_Changed_After_Estimation,
               i.Description_Changed_After_Estimation, i.Reporter_ID, i.Creator_ID, i.Assignee_ID
        FROM Issue i JOIN Project p ON p.ID = i.Project_ID
        WHERE p.Project_Key IN ({marcas})""", projetos)
    comps = consulta(conn, f"""
        SELECT ic.Issue_ID, c.Name
        FROM Issue_Component ic JOIN Component c ON c.ID = ic.Component_ID
        JOIN Issue i ON i.ID = ic.Issue_ID JOIN Project p ON p.ID = i.Project_ID
        WHERE p.Project_Key IN ({marcas})""", projetos)
    log = consulta(conn, f"""
        SELECT cl.Issue_ID, LOWER(cl.Field) AS Field, cl.From_String, cl.To_String, cl.Creation_Date
        FROM Change_Log cl JOIN Issue i ON i.ID = cl.Issue_ID JOIN Project p ON p.ID = i.Project_ID
        WHERE p.Project_Key IN ({marcas}) AND LOWER(cl.Field) IN ({", ".join(["%s"] * len(CAMPOS_LOG))})""",
        [*projetos, *CAMPOS_LOG])
    return issues, comps, log


def na_estimativa(issues: pd.DataFrame, comps: pd.DataFrame, log: pd.DataFrame) -> pd.DataFrame:
    """Valores dos campos na data da estimativa (sem data de estimativa, usa os valores atuais)."""
    for c in ["Creation_Date", "Estimation_Date"]:
        issues[c] = pd.to_datetime(issues[c], errors="coerce")
    log = log.copy()
    log["Creation_Date"] = pd.to_datetime(log.Creation_Date, errors="coerce")
    est = issues.set_index("ID").Estimation_Date
    log["depois"] = log.Creation_Date > log.Issue_ID.map(est)
    comp_atual = comps.groupby("Issue_ID").Name.apply(lambda s: set(map(str, s))).to_dict()
    por_issue = {k: g.sort_values("Creation_Date") for k, g in log.groupby("Issue_ID")}
    rows = []
    for r in issues.itertuples():
        g = por_issue.get(r.ID)
        tipo, prio, comp = r.Type, r.Priority, set(comp_atual.get(r.ID, set()))
        resp, resp_ok, labels = r.Assignee_ID, True, None
        if g is not None and pd.notna(r.Estimation_Date):
            depois = g[g.depois].iloc[::-1]                  # desfaz da mais recente para a mais antiga
            for m in depois.itertuples():
                if m.Field == "issuetype":
                    tipo = m.From_String
                elif m.Field == "priority":
                    prio = m.From_String
                elif m.Field == "component":
                    if isinstance(m.To_String, str) and m.To_String:
                        comp.discard(m.To_String)
                    if isinstance(m.From_String, str) and m.From_String:
                        comp.add(m.From_String)
                elif m.Field == "assignee":
                    resp_ok = False                          # mudou depois: não sabemos quem era
            antes = g[~g.depois & (g.Field == "labels")]
            if len(antes):
                labels = antes.iloc[-1].To_String or ""
        rows.append({"issuekey": r.Issue_Key, "tawos_project": r.Project_Key, "tipo": tipo, "prioridade": prio,
                     "componentes": " | ".join(sorted(c for c in comp if c)), "labels": labels or "",
                     "reporter": r.Reporter_ID, "creator": r.Creator_ID,
                     "responsavel": r.Assignee_ID if resp_ok else None, "responsavel_mudou": not resp_ok,
                     "creation_date": r.Creation_Date, "estimation_date": r.Estimation_Date,
                     "sp_tawos": r.Story_Point,
                     "sp_mudou_depois": r.Story_Point_Changed_After_Estimation,
                     "titulo_mudou_depois": r.Title_Changed_After_Estimation,
                     "descricao_mudou_depois": r.Description_Changed_After_Estimation})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3307)
    ap.add_argument("--user", default="root")
    ap.add_argument("--password", default="tawos", help="senha do MySQL local do container (não é segredo)")
    ap.add_argument("--database", default="TAWOS")
    args = ap.parse_args()

    chaves = config.DATA_PROCESSED / "agile_issuekeys.csv"
    if not chaves.exists():
        sys.exit("Falta data/processed/agile_issuekeys.csv. Rode antes: python repro/05_agile_issuekeys.py")
    keys = pd.read_csv(chaves, dtype={"issue_id": str, "project": str})
    projetos = sorted(keys.project.unique())
    conn = conecta(args)
    print("baixando issues, componentes e changelog do TAWOS ...", flush=True)
    issues, comps, log = baixa(conn, projetos)
    conn.close()
    print(f"   {len(issues)} issues, {len(comps)} componentes, {len(log)} mudanças registradas")

    meta = na_estimativa(issues, comps, log)
    out = keys[["issue_id", "project", "issuekey"]].merge(meta, on="issuekey", how="left")
    achou = out.sp_tawos.notna() | out.tipo.notna()
    print("\nCobertura por projeto (issues do Agile encontradas no TAWOS):")
    cob = out.assign(achou=achou).groupby("project").agg(issues=("issue_id", "size"), no_tawos=("achou", "sum"))
    cob["%"] = (100 * cob.no_tawos / cob.issues).round(1)
    print(cob.to_string())

    # o SP do Agile (2016) bate com o do TAWOS (2020)? diferenças = SP mudou depois da coleta do Agile
    sp_agile = pd.read_csv(config.DATA_PROCESSED / "agile.csv", dtype={"issue_id": str}, usecols=["issue_id", "sp"])
    chk = out[achou].merge(sp_agile, on="issue_id", how="left")
    igual = np.isclose(chk.sp.astype(float), pd.to_numeric(chk.sp_tawos, errors="coerce").astype(float))
    print(f"\nSP igual ao do TAWOS em {100 * igual.mean():.1f}% das issues encontradas"
          f" | com data de estimativa: {100 * chk.estimation_date.notna().mean():.1f}%"
          f" | responsável mudou depois da estimativa: {100 * chk.responsavel_mudou.mean():.1f}%"
          f" | com labels conhecidos: {100 * (chk.labels != '').mean():.1f}%")
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SAIDA, index=False)
    print(f"\nsalvo: {SAIDA.relative_to(config.ROOT)} ({int(achou.sum())} de {len(out)} issues com metadados)")
    print("Próximo passo: python repro/32_ml_metadados.py --datasets agile_limpo neo_limpo")


if __name__ == "__main__":
    main()
