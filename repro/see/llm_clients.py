"""Clientes mínimos para LLMs, usando só `requests` (sem SDKs).

Provedores:
  ollama       local, http://localhost:11434 (ou OLLAMA_HOST)
  gemini       Google AI Studio, API compatível com OpenAI        chave em GEMINI_API_KEY
  groq         Groq, API compatível com OpenAI                    chave em GROQ_API_KEY
  openrouter   OpenRouter, API compatível com OpenAI              chave em OPENROUTER_API_KEY
  openai_compat  qualquer endpoint compatível (--base-url e --api-key-env)

Os limites das camadas gratuitas mudam com frequência; o runner tem controle
de requisições por minuto (--rpm) e re-tentativas com espera crescente.
"""
import os
import time

import requests

PRESETS = {
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


class RetryableError(Exception):
    def __init__(self, msg, wait=None):
        super().__init__(msg)
        self.wait = wait


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 base_url: str = None, api_key_env: str = None, seed: int = 42, timeout: int = 180):
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens, self.seed, self.timeout = temperature, max_tokens, seed, timeout
        if provider == "ollama":
            self.base_url = base_url or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            if not self.base_url.startswith("http"):
                self.base_url = "http://" + self.base_url
            self.api_key = None
        else:
            preset_url, preset_env = PRESETS.get(provider, (None, None))
            self.base_url = base_url or preset_url
            env = api_key_env or preset_env
            if not self.base_url:
                raise SystemExit("Para openai_compat informe --base-url.")
            self.api_key = os.environ.get(env) if env else None
            if env and not self.api_key:
                raise SystemExit(f"Defina a variável de ambiente {env} com a sua chave de API.")
        self.base_url = self.base_url.rstrip("/")

    def complete(self, prompt: str) -> str:
        if self.provider == "ollama":
            return self._ollama(prompt)
        return self._openai_compat(prompt)

    def _ollama(self, prompt: str) -> str:
        body = {"model": self.model, "stream": False,
                "messages": [{"role": "user", "content": prompt}],
                "options": {"temperature": self.temperature, "seed": self.seed,
                            "num_predict": self.max_tokens}}
        r = self._post(f"{self.base_url}/api/chat", body, headers={})
        return r.json()["message"]["content"]

    def _openai_compat(self, prompt: str) -> str:
        body = {"model": self.model, "messages": [{"role": "user", "content": prompt}],
                "temperature": self.temperature, "max_tokens": self.max_tokens}
        if self.provider != "gemini":  # a API do Gemini rejeita `seed`
            body["seed"] = self.seed
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        r = self._post(f"{self.base_url}/chat/completions", body, headers=headers)
        data = r.json()
        try:
            return data["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError):
            raise RuntimeError(f"resposta inesperada: {str(data)[:300]}")

    def _post(self, url, body, headers):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as e:
            raise RetryableError(f"conexão: {e}")
        if r.status_code == 429 or r.status_code >= 500:
            try:
                wait = float(r.headers.get("retry-after"))
            except (TypeError, ValueError):
                wait = None
            raise RetryableError(f"HTTP {r.status_code}: {r.text[:200]}", wait=wait)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        return r


def call_with_retry(client: LLMClient, prompt: str, max_retries: int = 6, base_wait: float = 5.0):
    """Devolve (texto, latência_em_s, tentativas). Re-tenta em 429/5xx/timeout."""
    for attempt in range(1, max_retries + 1):
        t0 = time.perf_counter()
        try:
            text = client.complete(prompt)
            return text, time.perf_counter() - t0, attempt
        except RetryableError as e:
            if attempt == max_retries:
                raise
            wait = max(e.wait or 0, base_wait * 2 ** (attempt - 1))
            print(f"    re-tentando em {wait:.0f}s ({e})", flush=True)
            time.sleep(wait)
