#!/usr/bin/env bash
# setup.sh — Provisiona o pod RunPod para rodar o TRIBE v2.
#
# O QUE FAZ:
#   - cria as pastas persistentes no Network Volume (/workspace)
#   - exporta HF_HOME e TRIBE_CACHE pra /workspace (não re-baixa pesos a cada restart)
#   - instala deps de sistema (ffmpeg p/ decodificar vídeo) e Python (numpy<2.1,
#     tribev2, nilearn, pandas, scipy)
#   - clona o repo tribev2 (pra termos utils_fmri.py e o demo notebook à mão)
#
# O QUE *NÃO* FAZ (de propósito — ver CLAUDE.md item 8):
#   - NÃO reinstala torch (já vem com CUDA no template; reinstalar quebra a GPU)
#   - NÃO faz huggingface-cli login (você roda isso à mão, é interativo)
#
# Uso:  bash setup.sh

set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
HF_HOME_DIR="$WORKSPACE/hf_cache"
TRIBE_CACHE_DIR="$WORKSPACE/tribe_cache"
PROJECT_DIR="$WORKSPACE/project"
TRIBE_REPO="$WORKSPACE/tribev2"

echo "==> 1/6  Criando pastas persistentes no Network Volume ($WORKSPACE)"
mkdir -p "$HF_HOME_DIR" "$TRIBE_CACHE_DIR" "$PROJECT_DIR/videos"

echo "==> 2/6  Persistindo variáveis de ambiente (HF_HOME, TRIBE_CACHE) no ~/.bashrc"
# grava só se ainda não estiver lá, pra não duplicar a cada execução
grep -q "TRIBE_CACHE=" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<EOF

# --- TRIBE v2 project (adicionado pelo setup.sh) ---
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR"
export TRIBE_CACHE="$TRIBE_CACHE_DIR"
export PROJECT_DIR="$PROJECT_DIR"
EOF
# exporta também na sessão atual
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR"
export TRIBE_CACHE="$TRIBE_CACHE_DIR"
export PROJECT_DIR="$PROJECT_DIR"

echo "==> 3/6  Instalando deps de sistema (ffmpeg, git)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git >/dev/null
else
  echo "    (apt-get não encontrado — pulei deps de sistema; garanta ffmpeg manualmente)"
fi

echo "==> 4/6  Fixando numpy < 2.1 (exigência do TRIBE — NÃO mexer no torch)"
pip install -q "numpy<2.1"

echo "==> 5/6  Clonando + instalando o tribev2 (sem tocar no torch)"
if [ ! -d "$TRIBE_REPO/.git" ]; then
  git clone --depth 1 https://github.com/facebookresearch/tribev2.git "$TRIBE_REPO"
else
  echo "    repo já existe em $TRIBE_REPO — git pull"
  git -C "$TRIBE_REPO" pull --ff-only || true
fi
# instala em modo editável, SEM deixar o pip puxar/atualizar torch
pip install -q --no-build-isolation -e "$TRIBE_REPO" || pip install -q -e "$TRIBE_REPO"

echo "==> 6/6  Instalando deps de análise (nilearn, pandas, scipy)"
pip install -q nilearn pandas scipy
# reafirma numpy<2.1 caso alguma dep tenha tentado subir
pip install -q "numpy<2.1"

echo ""
echo "================================================================"
echo " SETUP OK."
echo "   HF_HOME      = $HF_HOME"
echo "   TRIBE_CACHE  = $TRIBE_CACHE"
echo "   PROJECT_DIR  = $PROJECT_DIR  (ponha os .mp4 em $PROJECT_DIR/videos)"
echo "   tribev2 repo = $TRIBE_REPO  (tem utils_fmri.py / demo notebook)"
echo ""
echo " PRÓXIMO:"
echo "   1) huggingface-cli login        # token 'read'; aceite a licença do"
echo "                                    #   meta-llama/Llama-3.2-3B no site da HF"
echo "   2) python smoke_test.py         # tem que terminar com 'MÁQUINA PRONTA'"
echo "================================================================"
