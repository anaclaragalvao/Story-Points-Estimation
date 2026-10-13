"""Etapa 1 — baixa os dois datasets de story points em versões fixas (por commit).

    python repro/01_download_data.py

- Agile User Story Point Dataset: arquivo data_csv/data do repositório de
  L. Truong (mrthlinh), a mesma fonte citada no artigo [40]. Contém as 23.313
  issues de Choetkiertikul et al. (2019).
- NEO Dataset: 34 CSVs (um por projeto) do repositório giseldo/neodataset [41].

Sem rede? Rode com --from-dir apontando para clones locais desses repositórios.
"""
import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from see import config  # noqa: E402

RAW = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def fetch(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", nargs=2, metavar=("AGILE_CLONE", "NEO_CLONE"),
                    help="copiar de clones locais em vez de baixar")
    ap.add_argument("--force", action="store_true", help="baixar de novo mesmo se já existir")
    args = ap.parse_args()

    # ---------------- Agile ----------------
    agile_dest = config.DATA_RAW / "agile" / "data.csv"
    if args.force or not agile_dest.exists():
        if args.from_dir:
            agile_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(args.from_dir[0]) / config.AGILE_FILE, agile_dest)
        else:
            print("Baixando Agile...")
            fetch(RAW.format(repo=config.AGILE_REPO, commit=config.AGILE_COMMIT, path=config.AGILE_FILE), agile_dest)
    digest = sha256(agile_dest)
    if digest != config.AGILE_SHA256:
        sys.exit(f"ERRO: hash do Agile não confere ({digest}). Arquivo diferente do usado na reprodução.")
    print(f"Agile OK  -> {agile_dest}")

    # ---------------- NEO ----------------
    neo_dir = config.DATA_RAW / "neo"
    for pid in config.NEO_PROJECT_IDS:
        dest = neo_dir / f"{pid}.csv"
        if dest.exists() and not args.force:
            continue
        if args.from_dir:
            neo_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(args.from_dir[1]) / "csv" / f"{pid}.csv", dest)
        else:
            print(f"Baixando NEO {pid}...")
            fetch(RAW.format(repo=config.NEO_REPO, commit=config.NEO_COMMIT, path=f"csv/{pid}.csv"), dest)
    print(f"NEO OK    -> {neo_dir} ({len(config.NEO_PROJECT_IDS)} projetos)")

    # ------- NEO: metadados dos repositórios (extensão; não usado no artigo) -------
    meta_dir = config.DATA_RAW / "neo_repos"
    for pid in config.NEO_PROJECT_IDS:
        dest = meta_dir / f"{pid}.json"
        if dest.exists() and not args.force:
            continue
        if args.from_dir:
            meta_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(Path(args.from_dir[1]) / "util" / "json head" / f"{pid}.json", dest)
        else:
            fetch(RAW.format(repo=config.NEO_REPO, commit=config.NEO_COMMIT,
                             path=f"util/json%20head/{pid}.json"), dest)
    print(f"NEO repos -> {meta_dir}")


if __name__ == "__main__":
    main()
