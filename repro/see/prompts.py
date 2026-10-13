"""Monta os prompts a partir dos templates ORIGINAIS do repositório (pasta Prompts/).

- ZS1..ZS10  zero-shot. O artigo usa ZS1 na avaliação principal e os 10 no teste
             de sensibilidade ao prompt (RQ2).
- FS1..FS3   few-shot. Os arquivos trazem "Example 1, 2, ..., 4" com reticências;
             aqui geramos de 2 a 4 exemplos reais, mantendo o cabeçalho e o rodapé
             de cada template.
"""
import re

from . import config

ZS_TEMPLATES = [f"ZS{i}" for i in range(1, 11)]
FS_TEMPLATES = ["FS1", "FS2", "FS3"]


def _read(path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()


def load_template(name: str) -> str:
    if name in ZS_TEMPLATES:
        return _read(config.PROMPTS_DIR / "Zero-Shot templates" / f"{name}.txt")
    if name in FS_TEMPLATES:
        return _read(config.PROMPTS_DIR / "Few-Shots templates" / f"{name}.txt")
    raise ValueError(f"template desconhecido: {name}")


def truncate(text: str, max_chars: int) -> str:
    text = text or ""
    if max_chars and len(text) > max_chars:
        return text[:max_chars].rsplit(" ", 1)[0] + " ..."
    return text


def zero_shot(template: str, title: str, description: str) -> str:
    t = load_template(template)
    return t.replace("<title>", title).replace("<description>", description)


def few_shot(template: str, title: str, description: str, examples: list) -> str:
    """examples: lista de (title, description, sp)."""
    t = load_template(template)
    lines = t.split("\n")
    header = lines[0]
    footer_start = next(i for i, ln in enumerate(lines) if ln.lower().startswith("now estimate"))
    footer = "\n".join(lines[footer_start:])
    blocks = []
    for i, (ex_title, ex_desc, ex_sp) in enumerate(examples, start=1):
        blocks.append(f"Example {i}:\nTitle: {ex_title}\nDescription: {ex_desc}\nStory Points: {format_sp(ex_sp)}")
    body = "\n\n".join(blocks)
    footer = footer.replace("<title>", title).replace("<description>", description)
    return f"{header}\n{body}\n\n{footer}"


def format_sp(sp) -> str:
    sp = float(sp)
    return str(int(sp)) if sp.is_integer() else str(sp)


_NUMBER = re.compile(r"-?\d+(?:[.,]\d+)?")


def parse_estimate(text: str):
    """Primeiro número da resposta. Remove blocos <think> de modelos de raciocínio."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    m = _NUMBER.search(text)
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", "."))
    except ValueError:
        return None
