"""patch_portuguese.py — Ajusta a transcrição (whisperx) do tribev2 para PORTUGUÊS.

O tribev2 fixa a transcrição em inglês (`language = "english"`) e nem lista o
português entre os idiomas suportados — então, sem isto, ele "ouve" áudio em PT como
se fosse inglês (camada de linguagem vira ruído). Este patch:
  1. adiciona portuguese="pt" ao dicionário de idiomas suportados;
  2. troca o idioma padrão de 'english' para 'portuguese'.

Edita /workspace/tribev2/tribev2/eventstransforms.py (instalação editável, fica no
volume e sobrevive a restart). IDEMPOTENTE: rodar de novo não causa problema.

Uso:  python patch_portuguese.py
"""
import os
import pathlib
import sys

repo = os.environ.get("TRIBE_REPO", "/workspace/tribev2")
f = pathlib.Path(repo) / "tribev2" / "eventstransforms.py"
if not f.exists():
    sys.exit(f"Não achei {f}. O tribev2 está instalado em {repo}?")

s = f.read_text()
orig = s

# 1. adiciona portuguese ao dict de idiomas (se ainda não estiver lá)
if 'portuguese="pt"' not in s:
    if 'chinese="zh"' in s:
        s = s.replace('chinese="zh"', 'chinese="zh", portuguese="pt"')
    else:
        print("AVISO: não achei 'chinese=\"zh\"' — o formato do dict pode ter mudado. "
              "Verifique eventstransforms.py manualmente.")

# 2. idioma padrão -> português
s = s.replace('language: str = "english"', 'language: str = "portuguese"')

if s == orig:
    print("Nada a mudar (já estava em português, ou o arquivo tem outro formato).")
else:
    f.write_text(s)
    print(f"✅ Patch aplicado: transcrição agora em PORTUGUÊS ({f})")
