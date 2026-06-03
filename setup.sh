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
PIP_CACHE_DIR_VOL="$WORKSPACE/pip_cache"
UV_CACHE_DIR_VOL="$WORKSPACE/uv_cache"        # whisperx instalado via uvx
TORCH_HOME_VOL="$WORKSPACE/torch_cache"       # modelo de alinhamento wav2vec2
mkdir -p "$PIP_CACHE_DIR_VOL" "$UV_CACHE_DIR_VOL" "$TORCH_HOME_VOL"
grep -q "TRIBE_CACHE=" ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<EOF

# --- TRIBE v2 project (adicionado pelo setup.sh) ---
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR"
export TRIBE_CACHE="$TRIBE_CACHE_DIR"
export PROJECT_DIR="$PROJECT_DIR"
export PIP_CACHE_DIR="$PIP_CACHE_DIR_VOL"
export UV_CACHE_DIR="$UV_CACHE_DIR_VOL"
export TORCH_HOME="$TORCH_HOME_VOL"
export HF_HUB_ENABLE_HF_TRANSFER=0
EOF
# exporta também na sessão atual
export HF_HOME="$HF_HOME_DIR"
export HUGGINGFACE_HUB_CACHE="$HF_HOME_DIR"
export TRIBE_CACHE="$TRIBE_CACHE_DIR"
export PROJECT_DIR="$PROJECT_DIR"
# Caches no volume -> sobrevivem ao restart do container (que zera o resto).
#   PIP_CACHE_DIR : não re-baixa torch/cuda (GBs) ao re-rodar o setup
#   UV_CACHE_DIR  : a transcrição usa 'uvx whisperx' (baixa o whisperx + deps)
#   TORCH_HOME    : modelo de alinhamento wav2vec2 do whisperx
export PIP_CACHE_DIR="$PIP_CACHE_DIR_VOL"
export UV_CACHE_DIR="$UV_CACHE_DIR_VOL"
export TORCH_HOME="$TORCH_HOME_VOL"
# O template liga HF_HUB_ENABLE_HF_TRANSFER=1, mas o 'hf_transfer' não existe no
# ambiente do whisperx (uvx) -> download falha. Desliga o "turbo" pra usar o
# downloader padrão (mais lento, mas funciona).
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "==> 3/6  Instalando deps de sistema (ffmpeg, git)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq ffmpeg git >/dev/null
else
  echo "    (apt-get não encontrado — pulei deps de sistema; garanta ffmpeg manualmente)"
fi

echo "==> 4/6  Guardando a versão do torch do template (o tribev2 costuma rebaixá-la)"
# IMPORTANTE: o template PyTorch 2.8.0 traz o torch que suporta GPUs Blackwell
# (RTX 5090 / RTX PRO 6000 = sm_120). O tribev2, ao instalar, REBAIXA o torch para
# 2.6.0+cu124, que NÃO conhece a 5090 -> "no kernel image is available". Por isso
# guardamos a versão original e restauramos depois.
TORCH_BEFORE=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
echo "    torch atual: ${TORCH_BEFORE:-desconhecido}"

echo "==> 5/6  Clonando + instalando o tribev2"
if [ ! -d "$TRIBE_REPO/.git" ]; then
  git clone --depth 1 https://github.com/facebookresearch/tribev2.git "$TRIBE_REPO"
else
  echo "    repo já existe em $TRIBE_REPO — git pull"
  git -C "$TRIBE_REPO" pull --ff-only || true
fi
pip install -q -e "$TRIBE_REPO" || true

# Restaura o torch original do template se o tribev2 o tiver mudado.
TORCH_AFTER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "")
if [ -n "$TORCH_BEFORE" ] && [ "$TORCH_AFTER" != "$TORCH_BEFORE" ]; then
  echo "    ⚠️  tribev2 mudou o torch: $TORCH_BEFORE -> $TORCH_AFTER. Restaurando o stack do template..."
  # Assume o template RunPod PyTorch 2.8.0 (cu128). Se você usar outro template,
  # ajuste estas versões para as que vinham instaladas (veja TORCH_BEFORE acima).
  # IMPORTANTE: NÃO usar --no-deps aqui. Sem as deps, as libs CUDA (nvidia-*-cu12)
  # ficam na versão antiga (12.4) e dão "undefined symbol: cudaGetDriverEntryPoint...".
  # Com deps, o torch 2.8 puxa as libs CUDA 12.8 corretas. O numpy é re-fixado no 6/6.
  pip install -q --force-reinstall \
    torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128 \
    || echo "    (falha ao restaurar automaticamente — restaure o torch à mão)"
fi

echo "==> 6/6  Instalando deps de análise + acertando versões que o tribev2 exige"
pip install -q nilearn pandas scipy
# O tribev2 0.1.0 exige numpy==2.2.6 (NÃO é <2.1 como dizia o brief antigo).
pip install -q "numpy==2.2.6"
# O neuralset exige exca>=0.5.20, mas as versões mais novas do exca removeram
# 'exca.steps.base.NoValue', que o tribev2 usa -> ImportError. Fixa em 0.5.20.
pip install -q "exca==0.5.20"
# O torchvision puxa pillow 12, mas o moviepy (extrai áudio do vídeo) exige
# pillow<12. Fixa numa 11.x que satisfaz os dois.
pip install -q "pillow<12"

echo ""
echo "================================================================"
echo " SETUP OK."
echo "   HF_HOME      = $HF_HOME"
echo "   TRIBE_CACHE  = $TRIBE_CACHE"
echo "   PROJECT_DIR  = $PROJECT_DIR  (ponha os .mp4 em $PROJECT_DIR/videos)"
echo "   tribev2 repo = $TRIBE_REPO  (tem utils_fmri.py / demo notebook)"
echo ""
echo " PRÓXIMO:"
echo "   1) hf auth login                # ('huggingface-cli' foi descontinuado)"
echo "                                    #   token 'read'; aceite a licença do"
echo "                                    #   meta-llama/Llama-3.2-3B no site da HF"
echo "   2) python smoke_test.py         # tem que terminar com 'MÁQUINA PRONTA'"
echo "================================================================"
