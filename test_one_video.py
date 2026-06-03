"""test_one_video.py — Testa o pipeline COMPLETO num vídeo REAL (com fala).

É o "smoke test de verdade": em vez do clipe sintético (que trava por não ter fala),
roda num reel real e revela o NÚMERO DE VÉRTICES (o dado mais importante p/ o
alinhamento) + as features neurais.

Uso:
  python test_one_video.py /workspace/project/videos/SEU_VIDEO.mp4
"""
import os
import sys
import numpy as np

import tribe_virality as tv

if len(sys.argv) < 2:
    sys.exit("Uso: python test_one_video.py <caminho_do_video.mp4>")
video = sys.argv[1]
if not os.path.exists(video):
    sys.exit(f"Não achei o vídeo: {video}")
print(f"Vídeo: {video}\n")

from tribev2.demo_utils import TribeModel
print("== Carregando modelo (usa o cache do volume) ==")
model = TribeModel.from_pretrained(
    "facebook/tribev2", cache_folder=os.environ.get("TRIBE_CACHE")
)

print("== get_events_dataframe (transcreve o áudio — pode baixar um modelo na 1ª vez) ==")
df = model.get_events_dataframe(video_path=video)

print("== predict (a predição cortical — pode levar 1-3 min) ==")
preds, segments = model.predict(events=df)
preds = np.asarray(preds)
print(f"\n>>> preds.shape = {preds.shape}  ->  NÚMERO DE VÉRTICES = {preds.shape[1]} <<<\n")

print("== Montando atlas Destrieux + ROI por rede ==")
vn = tv.build_vertex_networks(cache_dir=os.environ.get("HF_HOME"))
roi = tv.roi_timeseries(preds, vn)
feats = tv.neuro_features(roi)

print("Features-âncora:",
      {k: round(v, 3) for k, v in feats.items() if k in tv.PRIOR_WEIGHTS})
print("Médias por rede:",
      {net: round(feats[f"{net}_mean"], 3) for net in tv.NETWORKS})
print("\n>>> VÍDEO REAL PROCESSADO COM SUCESSO — MÁQUINA VALIDADA <<<")
