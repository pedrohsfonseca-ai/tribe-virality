# env.sh — exporta todas as variáveis do projeto de uma vez.
#
# O container do RunPod ZERA o ambiente a cada restart (só /workspace sobrevive).
# Em QUALQUER terminal novo, antes de rodar os scripts, faça:
#     cd /workspace/tribe-virality && source env.sh
#
# Assim você garante que os caches apontam pro volume e que o download do whisperx
# (HF_HUB_ENABLE_HF_TRANSFER=0) funciona.

export HF_HOME=/workspace/hf_cache
export HUGGINGFACE_HUB_CACHE=/workspace/hf_cache
export TRIBE_CACHE=/workspace/tribe_cache
export PROJECT_DIR=/workspace/project
export PIP_CACHE_DIR=/workspace/pip_cache
export UV_CACHE_DIR=/workspace/uv_cache
export TORCH_HOME=/workspace/torch_cache
export HF_HUB_ENABLE_HF_TRANSFER=0

echo "Variáveis do projeto carregadas (HF_HOME, TRIBE_CACHE, UV_CACHE_DIR, TORCH_HOME, ...)."
