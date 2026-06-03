"""smoke_test.py — Valida a MÁQUINA INTEIRA antes de gastar GPU com vídeos reais.

Roda um clipe SINTÉTICO de poucos segundos pelo pipeline completo e checa, em ordem,
falhando ALTO e com mensagem clara no primeiro problema:

  1. torch + CUDA + VRAM suficiente
  2. numpy < 2.1
  3. import do tribev2 e do nosso tribe_virality
  4. gera um clipe sintético (ffmpeg)
  5. carrega o TribeModel (baixa pesos pro TRIBE_CACHE)
  6. predict() no clipe -> DESCOBRE o nº REAL de vértices (PONTO FRÁGIL #1)
  7. monta o atlas vértice->rede (nilearn/Destrieux) e alinha contra preds
  8. roda roi_timeseries + neuro_features

Se tudo passar, imprime 'MÁQUINA PRONTA'.

Uso:  python smoke_test.py
"""
import os
import subprocess
import sys
import tempfile


def fail(msg: str, hint: str = "") -> None:
    print(f"\n❌ FALHOU: {msg}")
    if hint:
        print(f"   → {hint}")
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"✅ {msg}")


# --- 1. torch + CUDA -------------------------------------------------------
print("== 1/8  torch + CUDA ==")
try:
    import torch
except ImportError:
    fail("torch não importa.", "O template do RunPod já traz torch — NÃO reinstale. "
         "Confira se o pod é o template PyTorch 2.x.")

if not torch.cuda.is_available():
    fail("CUDA indisponível (torch.cuda.is_available() == False).",
         "Confirme que o pod tem GPU anexada e que você não reinstalou o torch.")

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
ok(f"GPU: {gpu_name} | VRAM total: {vram_gb:.0f} GB | torch {torch.__version__}")
if vram_gb < 30:
    print(f"   ⚠️  predict() do TRIBE tem pico ~24–28 GB. {vram_gb:.0f} GB pode "
          f"estourar em vídeos longos. A40 48GB / A100 40GB+ é o recomendado.")


# --- 2. numpy compatível com o tribev2 (==2.2.6) ---------------------------
print("\n== 2/8  numpy (tribev2 exige 2.2.6) ==")
import numpy as np
from packaging.version import Version
if Version(np.__version__) < Version("2.1"):
    fail(f"numpy {np.__version__} < 2.1.",
         "O tribev2 0.1.0 exige numpy==2.2.6. Rode: pip install 'numpy==2.2.6'")
ok(f"numpy {np.__version__}")


# --- 3. imports do projeto -------------------------------------------------
print("\n== 3/8  imports (tribev2 + tribe_virality) ==")
try:
    from tribev2.demo_utils import TribeModel
except Exception as e:
    fail(f"não consegui importar tribev2.demo_utils.TribeModel ({e}).",
         "Rode o setup.sh. Confira o caminho real do import no repo clonado "
         "(/workspace/tribev2) — pode ser tribev2.demo_utils ou similar.")
try:
    import tribe_virality as tv
except Exception as e:
    fail(f"não consegui importar tribe_virality ({e}).",
         "Rode o smoke_test.py de dentro da pasta do projeto, junto do tribe_virality.py.")
ok("tribev2.demo_utils.TribeModel e tribe_virality importados")


# --- 4. clipe sintético ----------------------------------------------------
print("\n== 4/8  gerando clipe sintético (ffmpeg) ==")
tmpdir = tempfile.mkdtemp(prefix="tribe_smoke_")
clip = os.path.join(tmpdir, "synthetic.mp4")
# 6s de vídeo de teste (barras) + tom de áudio senoidal, ~caso de uso real
cmd = [
    "ffmpeg", "-y", "-loglevel", "error",
    "-f", "lavfi", "-i", "testsrc=duration=6:size=320x240:rate=12",
    "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
    "-pix_fmt", "yuv420p", "-shortest", clip,
]
try:
    subprocess.run(cmd, check=True)
except (subprocess.CalledProcessError, FileNotFoundError) as e:
    fail(f"ffmpeg falhou ao gerar o clipe ({e}).",
         "Instale ffmpeg (o setup.sh faz isso): apt-get install -y ffmpeg")
ok(f"clipe sintético: {clip}")


# --- 5. carregar o modelo --------------------------------------------------
print("\n== 5/8  carregando TribeModel (baixa pesos pro TRIBE_CACHE) ==")
cache = os.environ.get("TRIBE_CACHE", os.path.join(tmpdir, "tribe_cache"))
try:
    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=cache)
except Exception as e:
    fail(f"não consegui carregar o modelo ({e}).",
         "Se for erro de acesso (gated/401), rode `huggingface-cli login` e aceite a "
         "licença do meta-llama/Llama-3.2-3B na HuggingFace (o módulo de TEXTO usa "
         "LLaMA 3.2-3B). Confira também o nome da API no demo do repo.")
ok("modelo carregado")


# --- 6. predict + DESCOBERTA do nº de vértices (PONTO FRÁGIL #1) -----------
print("\n== 6/8  predict() + checagem de vértices ==")
try:
    df = model.get_events_dataframe(video_path=clip)
    preds, segments = model.predict(events=df)
    preds = np.asarray(preds)
except Exception as e:
    fail(f"predict() falhou ({e}).",
         "Se reclamar de modelo gated, veja o passo 5. Se a API for diferente, "
         "confira o tribe_demo.ipynb do repo e me avise o nome certo dos métodos.")

print(f"   preds.shape = {preds.shape}   (T={preds.shape[0]}, V={preds.shape[1]})")
print(f"   >>> NÚMERO REAL DE VÉRTICES: {preds.shape[1]} <<<")
V = preds.shape[1]
if V == tv.N_VERTS_CORTEX:
    ok(f"{V} vértices == {tv.N_VERTS_CORTEX} corticais fsaverage5. Alinhamento direto.")
elif V > tv.N_VERTS_CORTEX:
    print(f"   ⚠️  {V} > {tv.N_VERTS_CORTEX}: provável subcortical anexado. "
          f"align_preds_to_cortex vai fatiar os {tv.N_VERTS_CORTEX} primeiros — "
          f"CONFIRME a ordem no utils_fmri.py do repo antes de confiar.")
else:
    print(f"   ⚠️  {V} < {tv.N_VERTS_CORTEX}: PROVÁVEL medial wall mascarada. O atlas "
          f"cru NÃO alinha. Precisamos do índice dos vértices válidos (ver CLAUDE.md "
          f"#6). O passo 7 vai falhar de propósito — me traga o nº {V} que eu ajusto.")


# --- 7. atlas vértice->rede + alinhamento ---------------------------------
print("\n== 7/8  atlas Destrieux (nilearn) + alinhamento ==")
try:
    vertex_networks = tv.build_vertex_networks(cache_dir=os.environ.get("HF_HOME"))
except Exception as e:
    fail(f"build_vertex_networks falhou ({e}).",
         "Confira a instalação do nilearn e o acesso à internet pra baixar o atlas.")

try:
    roi = tv.roi_timeseries(preds, vertex_networks)
except Exception as e:
    fail(f"roi_timeseries falhou no alinhamento ({e}).",
         f"Quase certo: nº de vértices ({V}) != atlas (20484). É o PONTO FRÁGIL #1. "
         f"Me passe o valor {V} e a saída do utils_fmri.py pra eu corrigir o atlas.")
ok(f"ROI por rede OK — curvas de tamanho T={len(roi)} para {len(tv.NETWORKS)} redes")


# --- 8. features -----------------------------------------------------------
print("\n== 8/8  neuro_features ==")
try:
    feats = tv.neuro_features(roi)
except Exception as e:
    fail(f"neuro_features falhou ({e}).")
ok("features extraídas: " + ", ".join(f"{k}={v:.2f}" for k, v in feats.items()
                                       if k in tv.PRIOR_WEIGHTS))

print("\n" + "=" * 50)
print("  MÁQUINA PRONTA")
print("=" * 50)
