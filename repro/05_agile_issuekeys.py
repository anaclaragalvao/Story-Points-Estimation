"""EXTENSÃO (não faz parte do artigo) — recupera a chave Jira de cada issue do Agile.

    python repro/05_agile_issuekeys.py
    python repro/05_agile_issuekeys.py --from-dir CLONE_DE_morakotch_datasets   # sem rede

O arquivo do mrthlinh usado na reprodução não tem a chave da issue (por isso o
02_prepare_data cria ids sintéticos como MDL-00000) nem datas. Os CSVs originais
de Choetkiertikul et al. (2019) têm a chave (issuekey, title, description,
storypoint). Esta etapa baixa os 16 CSVs num commit fixo e casa cada issue do
nosso agile.csv com a original.

Casamento: o mrthlinh embaralhou a ordem das linhas, então a junção é pelo texto:
projeto + título + descrição (300 primeiros caracteres) + SP, todos normalizados
(só letras e dígitos, minúsculas). Issues com o mesmo texto e o mesmo SP são
pareadas pela ordem de ocorrência. O que sobrar tenta projeto + título + SP.
A etapa falha se alguma issue ficar sem chave ou se uma chave for usada duas vezes.

O jirasoftware.csv traz JSW e GHS: GHS é a chave antiga do GreenHopper, nome
antigo do Jira Software. As 68 issues "GHS" que duplicam JSW são as mesmas issues
com duas chaves, por causa da renomeação do projeto.

Saída: data/processed/agile_issuekeys.csv
    issue_id   id sintético do 02 (MDL-00000)
    project    código do projeto
    issuekey   chave Jira real (MDL-1626)
    key_num    número da chave; cresce com a data de criação dentro do projeto
    casamento  texto | titulo  (qual regra casou)
"""
import argparse
import hashlib
import re
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402
from see.io import load_processed  # noqa: E402

RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_originals(from_dir: Path | None, force: bool) -> Path:
    dest_dir = config.DATA_RAW / "agile_orig"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, digest in config.AGILE_ORIG_SHA256.items():
        dest = dest_dir / f"{name}.csv"
        if force or not dest.exists():
            if from_dir:
                shutil.copy(Path(from_dir) / config.AGILE_ORIG_DIR / f"{name}.csv", dest)
            else:
                print(f"Baixando {name}.csv...")
                url = RAW.format(repo=config.AGILE_ORIG_REPO, commit=config.AGILE_ORIG_COMMIT,
                                 path=urllib.parse.quote(f"{config.AGILE_ORIG_DIR}/{name}.csv"))
                tmp = dest.with_suffix(".part")
                with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f)
                tmp.replace(dest)
        if sha256(dest) != digest:
            sys.exit(f"ERRO: hash de {dest.name} não confere. Arquivo diferente do usado na extensão.")
    return dest_dir


def norm(s) -> str:
    return re.sub(r"\W+", "", "" if pd.isna(s) else str(s)).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", type=Path, help="clone local de morakotch/datasets (em vez de baixar)")
    ap.add_argument("--force", action="store_true", help="baixar de novo mesmo se já existir")
    args = ap.parse_args()

    orig_dir = fetch_originals(args.from_dir, args.force)
    orig = pd.concat([pd.read_csv(orig_dir / f"{n}.csv") for n in config.AGILE_ORIG_SHA256], ignore_index=True)
    orig["project"] = orig.issuekey.str.split("-").str[0]
    orig["sp"] = orig.storypoint.astype(float)

    mine = load_processed("agile")[["issue_id", "project", "title", "description", "sp"]].copy()
    for df in (orig, mine):
        df["kt"] = df.title.map(norm)
        df["kd"] = df.description.map(norm).str[:300]

    # 1) projeto + título + descrição + SP (pareando repetidos pela ordem de ocorrência)
    k1 = ["project", "kt", "kd", "sp"]
    for df in (orig, mine):
        df["occ"] = df.groupby(k1).cumcount()
    m = mine.merge(orig[k1 + ["occ", "issuekey"]], on=k1 + ["occ"], how="left", validate="one_to_one")
    m["casamento"] = m.issuekey.notna().map({True: "texto", False: None})

    # 2) o que sobrou: projeto + título + SP, só entre as chaves ainda não usadas
    rest = m.issuekey.isna()
    if rest.any():
        k2 = ["project", "kt", "sp"]
        o2 = orig[~orig.issuekey.isin(m.issuekey.dropna())].copy()
        o2["occ2"] = o2.groupby(k2).cumcount()
        left = m.loc[rest, ["issue_id"] + k2].copy()
        left["occ2"] = left.groupby(k2).cumcount()
        fix = left.merge(o2[k2 + ["occ2", "issuekey"]], on=k2 + ["occ2"], how="left").set_index("issue_id")
        m.loc[rest, "issuekey"] = m.loc[rest, "issue_id"].map(fix.issuekey)
        m.loc[rest & m.issuekey.notna(), "casamento"] = "titulo"

    sem = m[m.issuekey.isna()]
    if len(sem):
        print(sem.groupby("project").size().to_string())
        sys.exit(f"ERRO: {len(sem)} issues sem chave.")
    if m.issuekey.duplicated().any():
        sys.exit(f"ERRO: {int(m.issuekey.duplicated().sum())} chaves usadas mais de uma vez.")

    m["key_num"] = m.issuekey.str.split("-").str[1].astype(int)
    out = m[["issue_id", "project", "issuekey", "key_num", "casamento"]]
    path = config.DATA_PROCESSED / "agile_issuekeys.csv"
    out.to_csv(path, index=False)
    print(f"{len(out)} issues com chave ({out.casamento.value_counts().to_dict()}) -> {path.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
