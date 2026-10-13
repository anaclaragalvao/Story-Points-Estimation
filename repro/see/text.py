"""Limpeza de texto.

O artigo (Seção III-B.2) diz apenas: remover markup, URLs e fragmentos de código,
converter para minúsculas e normalizar espaços. Implementamos isso em duas versões:

- clean_light: remove código, URLs e markup, mas mantém maiúsculas/pontuação.
  Usada como entrada dos Transformers e dos LLMs.
- clean_full:  clean_light + minúsculas. Usada no TF-IDF do ML clássico.
"""
import re

_CODE_BLOCK = re.compile(
    r"```.*?```"                         # markdown (GitLab / NEO)
    r"|\{code(?::[^}]*)?\}.*?\{code\}"   # Jira {code}...{code}
    r"|\{noformat\}.*?\{noformat\}"      # Jira {noformat}
    r"|<pre>.*?</pre>",
    flags=re.DOTALL | re.IGNORECASE,
)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_URL = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")    # [texto](url) -> texto
_HTML_TAG = re.compile(r"<[^>]+>")
_JIRA_MARKUP = re.compile(r"\{[a-z]+(?::[^}]*)?\}|\[~[^\]]*\]|^h[1-6]\.\s*", flags=re.IGNORECASE | re.MULTILINE)
_MD_MARKUP = re.compile(r"^#{1,6}\s*|^\s*[-*+]\s+|\*\*|__|^>\s?", flags=re.MULTILINE)
_SPACES = re.compile(r"\s+")


def clean_light(text) -> str:
    if text is None or (isinstance(text, float) and text != text):  # NaN
        return ""
    t = str(text)
    t = _CODE_BLOCK.sub(" ", t)
    t = _INLINE_CODE.sub(" ", t)
    t = _MD_LINK.sub(r"\1", t)
    t = _URL.sub(" ", t)
    t = _HTML_TAG.sub(" ", t)
    t = _JIRA_MARKUP.sub(" ", t)
    t = _MD_MARKUP.sub(" ", t)
    return _SPACES.sub(" ", t).strip()


def clean_full(text) -> str:
    return clean_light(text).lower()
