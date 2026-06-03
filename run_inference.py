"""run_inference.py — Loop de inferência: vídeos reais -> tribe_features.csv

Para cada .mp4 em PROJECT_DIR/videos/:
  get_events_dataframe -> predict -> roi_timeseries -> neuro_features
Acumula tudo, calcula o NEURO CONTENT SCORE (relativo ao lote) e salva:
  - tribe_features.csv  (uma linha por vídeo: features + score)
  - preds/<video_id>.npz  (a matriz crua [T,V], pra re-análise sem re-rodar a GPU)

video_id = nome do arquivo sem extensão.

Uso:
  python run_inference.py                      # usa $PROJECT_DIR/videos
  python run_inference.py /caminho/dos/videos  # pasta custom
"""
import os
import sys
import glob
import numpy as np

import tribe_virality as tv


def main() -> None:
    project_dir = os.environ.get("PROJECT_DIR", "/workspace/project")
    videos_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(project_dir, "videos")
    out_csv = os.path.join(project_dir, "tribe_features.csv")
    preds_dir = os.path.join(project_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    videos = sorted(
        glob.glob(os.path.join(videos_dir, "*.mp4"))
        + glob.glob(os.path.join(videos_dir, "*.mov"))
        + glob.glob(os.path.join(videos_dir, "*.MP4"))
    )
    if not videos:
        print(f"❌ Nenhum vídeo encontrado em {videos_dir}. "
              f"Coloque os .mp4 lá (id = nome do arquivo).")
        sys.exit(1)
    print(f"Encontrei {len(videos)} vídeo(s) em {videos_dir}\n")

    # carrega modelo e atlas UMA vez
    from tribev2.demo_utils import TribeModel
    print("Carregando TribeModel...")
    model = TribeModel.from_pretrained(
        "facebook/tribev2", cache_folder=os.environ.get("TRIBE_CACHE")
    )
    print("Montando atlas vértice->rede (Destrieux)...")
    vertex_networks = tv.build_vertex_networks(cache_dir=os.environ.get("HF_HOME"))

    per_video: dict[str, dict] = {}
    failed: list[str] = []
    for i, path in enumerate(videos, 1):
        vid = os.path.splitext(os.path.basename(path))[0]
        print(f"[{i}/{len(videos)}] {vid} ...", flush=True)
        try:
            df = model.get_events_dataframe(video_path=path)
            preds, _segments = model.predict(events=df)
            preds = np.asarray(preds)
            np.savez_compressed(os.path.join(preds_dir, f"{vid}.npz"), preds=preds)

            roi = tv.roi_timeseries(preds, vertex_networks)
            per_video[vid] = tv.neuro_features(roi)
            print(f"     OK  preds={preds.shape}  "
                  f"hook={per_video[vid]['hook_power']:.2f} "
                  f"ret={per_video[vid]['retention_power']:.2f} "
                  f"eng={per_video[vid]['engagement_trigger']:.2f}")
        except Exception as e:
            print(f"     ❌ FALHOU: {e}")
            failed.append(vid)

    if not per_video:
        print("\n❌ Nenhum vídeo processado com sucesso. Rode o smoke_test.py.")
        sys.exit(1)

    feats = tv.features_dataframe(per_video)
    scored = tv.score_videos(feats)
    scored.to_csv(out_csv)

    print(f"\n✅ Salvo: {out_csv}  ({len(scored)} vídeos)")
    if failed:
        print(f"⚠️  Falharam ({len(failed)}): {', '.join(failed)}")
    print("\n=== Ranking por NEURO SCORE (relativo ao lote) ===")
    cols = ["hook_power", "retention_power", "engagement_trigger", "neuro_score"]
    print(scored[cols].round(2).to_string())
    print("\nPRÓXIMO: preencha metrics_template.csv -> metrics.csv com a performance "
          "real e rode o backtest (run_backtest.py).")


if __name__ == "__main__":
    main()
