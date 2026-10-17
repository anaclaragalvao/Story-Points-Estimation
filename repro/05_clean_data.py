"""EXTENSÃO — gera as versões limpas dos datasets: agile_limpo e neo_limpo.

    python repro/05_clean_data.py          (depois de 02_prepare_data.py)

As versões originais (agile, neo) continuam intactas e são as da reprodução do artigo.
As limpas usam os MESMOS splits (cada issue fica no mesmo train/val/test), apenas sem as
issues removidas; assim dá para comparar original × limpo nas mesmas issues que sobraram.

Regras (aplicadas nesta ordem; cada issue removida registra a primeira regra que a pegou):
  R1 sp_invalido      projeto cujo "story point" não é estimativa do time (config.PROJETOS_SP_INVALIDO)
  R2 sp_zero          SP = 0 (não é estimativa e quebra o PRED(25))
  R3 dup_conflito     duplicata exata (título + descrição) com SPs diferentes: remove o grupo inteiro
  R4 dup_identica     duplicata exata com o mesmo SP: mantém a primeira (inclui o projeto GHS, cópia do JSW)
  R5 sem_descricao    descrição vazia depois da limpeza de texto (vazia, só URL ou só markup)

Saídas:
  data/processed/{agile,neo}_limpo.csv
  data/splits/{agile,neo}_limpo_splits.csv
  data/processed/limpeza_log.csv          (issue_id, dataset, projeto, sp, regra)
  data/processed/limpeza_resumo.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_processed, load_splits  # noqa: E402
from see.text import clean_light  # noqa: E402


def regras(df: pd.DataFrame, dataset: str) -> pd.Series:
    regra = pd.Series(pd.NA, index=df.index, dtype="object")

    def marca(mask, nome):
        regra[mask & regra.isna()] = nome

    marca(df.project.isin(config.PROJETOS_SP_INVALIDO.get(dataset, [])), "sp_invalido")
    marca(df.sp == 0, "sp_zero")
    vivos = regra.isna()
    chave = (df.title.str.strip().str.lower() + "||" + df.description.str.strip().str.lower())
    n_sp = df[vivos].groupby(chave[vivos]).sp.transform("nunique")
    conflito = pd.Series(False, index=df.index)
    conflito.loc[n_sp.index] = n_sp > 1
    marca(conflito, "dup_conflito")
    vivos = regra.isna()
    repetida = pd.Series(False, index=df.index)
    repetida.loc[vivos[vivos].index] = chave[vivos].duplicated(keep="first")
    marca(repetida, "dup_identica")
    sem_desc = df.description.map(clean_light).str.split().str.len().fillna(0) == 0
    marca(sem_desc, "sem_descricao")
    return regra


def main():
    logs, resumo = [], []
    for dataset in config.DATASETS:
        df = load_processed(dataset)
        sp = load_splits(dataset)
        r = regras(df, dataset)
        manter = r.isna()
        out = f"{dataset}_limpo"
        df[manter].to_csv(config.DATA_PROCESSED / f"{out}.csv", index=False)
        sp[sp.issue_id.isin(df.loc[manter, "issue_id"])].to_csv(config.SPLITS / f"{out}_splits.csv", index=False)
        log = df.loc[~manter, ["issue_id", "project", "sp"]].assign(dataset=dataset, regra=r[~manter])
        logs.append(log)
        ev = sp.set_index("issue_id").within_eval
        linha = {"dataset": dataset, "issues_original": len(df)}
        for nome in ["sp_invalido", "sp_zero", "dup_conflito", "dup_identica", "sem_descricao"]:
            linha[f"-{nome}"] = int((r == nome).sum())
        linha.update({"issues_limpo": int(manter.sum()), "%_mantido": round(100 * manter.mean(), 1),
                      "projetos_limpo": df.loc[manter, "project"].nunique(),
                      "avaliacao_limpo": int(ev.loc[df.loc[manter, "issue_id"]].sum()),
                      "test_limpo": int((sp.set_index("issue_id").within.loc[df.loc[manter, "issue_id"]] == "test").sum()),
                      "SP_media": round(df.loc[manter, "sp"].mean(), 2), "SP_mediana": df.loc[manter, "sp"].median(),
                      "SP_desvio": round(df.loc[manter, "sp"].std(), 2)})
        resumo.append(linha)
    pd.concat(logs).to_csv(config.DATA_PROCESSED / "limpeza_log.csv", index=False)
    res = pd.DataFrame(resumo).set_index("dataset")
    res.to_csv(config.DATA_PROCESSED / "limpeza_resumo.csv")
    with pd.option_context("display.width", 200):
        print(res.T.to_string())
    print(f"\nArquivos em {config.DATA_PROCESSED} e {config.SPLITS}")


if __name__ == "__main__":
    main()
