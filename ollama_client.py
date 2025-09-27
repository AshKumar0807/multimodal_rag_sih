import subprocess
import os

OLLAMA_CMD = "ollama"
MODEL = "mistral"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)

def run_prompt(prompt: str, mode="offline", timeout=60):
    if mode == "offline":
        try:
            p = subprocess.run([OLLAMA_CMD, "run", MODEL],
                               input=prompt.encode(), capture_output=True, timeout=timeout)
            if p.returncode != 0:
                return f"[Ollama error] {p.stderr.decode()}"
            return p.stdout.decode()
        except Exception as e:
            return f"[Ollama exception] {e}"
    elif mode == "online":
        if not OPENAI_API_KEY:
            return "[OpenAI] API key not set"
        try:
            import openai
            openai.api_key = OPENAI_API_KEY
            res = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[{"role":"user","content":prompt}],
                temperature=0
            )
            return res['choices'][0]['message']['content']
        except Exception as e:
            return f"[OpenAI exception] {e}"
