"""run_backtest.py — Junta o score previsto com a performance REAL e mede se bate.

Lê:
  - tribe_features.csv  (saída do run_inference.py)
  - metrics.csv         (você preenche a partir do metrics_template.csv)

Roda o backtest (Spearman + top-k + correlação por feature) para a métrica-alvo e,
se houver vídeos suficientes (n>=15), tenta a calibração leave-one-out.

Uso:
  python run_backtest.py                          # alvo padrão: retention_pct
  python run_backtest.py reach_vs_typical         # outra coluna-alvo
  python run_backtest.py shares 4                 # alvo=shares, top-k=4
"""
import os
import sys
import pandas as pd

import tribe_virality as tv


def main() -> None:
    project_dir = os.environ.get("PROJECT_DIR", "/workspace/project")
    features_csv = os.path.join(project_dir, "tribe_features.csv")
    metrics_csv = os.path.join(project_dir, "metrics.csv")

    actual_col = sys.argv[1] if len(sys.argv) > 1 else "retention_pct"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else None

    if not os.path.exists(features_csv):
        sys.exit(f"❌ Não achei {features_csv}. Rode run_inference.py primeiro.")
    if not os.path.exists(metrics_csv):
        sys.exit(f"❌ Não achei {metrics_csv}. Preencha o metrics_template.csv e "
                 f"salve como metrics.csv (mesmos video_id do tribe_features.csv).")

    features = pd.read_csv(features_csv, index_col="video_id")
    metrics = pd.read_csv(metrics_csv)
    if actual_col not in metrics.columns:
        sys.exit(f"❌ Coluna '{actual_col}' não existe em metrics.csv. "
                 f"Colunas disponíveis: {list(metrics.columns)}")

    bt = tv.backtest(features, metrics, actual_col=actual_col, k=k)

    print("\n" + "=" * 60)
    print(f" BACKTEST — alvo: {actual_col}")
    print("=" * 60)
    print(f" n (vídeos casados) : {bt['n']}")
    print(f" Spearman rho       : {bt['spearman_rho']:+.3f}  (p={bt['p_value']:.3f})")
    print(f" Interpretação      : {bt['interpretacao']}")
    print(f" Top-{bt['k']} hit rate    : {bt['top_k_hit_rate']:.0%}  "
          f"({bt['top_k_hits']}/{bt['k']} acertos)")
    print(f"   previstos top-{bt['k']}: {bt['top_k_pred']}")
    print(f"   reais     top-{bt['k']}: {bt['top_k_real']}")

    print("\n Correlação por feature (qual sinal mais prevê o alvo):")
    for feat, st in bt["per_feature"].items():
        print(f"   {feat:<22} rho={st['rho']:+.3f}  (p={st['p']:.3f})")

    cal = tv.calibrate_weights(features, metrics, actual_col=actual_col)
    print("\n Calibração leave-one-out:")
    if cal["ok"]:
        print(f"   LOO Spearman rho = {cal['loo_spearman_rho']:+.3f} "
              f"-> {cal['interpretacao']}")
        print(f"   pesos refit: { { k: round(v,3) for k,v in cal['weights_mean'].items() } }")
    else:
        print(f"   {cal['msg']}")

    print("\n LEITURA HONESTA:")
    if bt["n"] < 15:
        print("   n<15 — resultado FRÁGIL. Trate como hipótese, não acurácia.")
    rho = abs(bt["spearman_rho"])
    if rho >= 0.6:
        print("   Sinal promissor. Vale seguir e juntar mais vídeos pra confirmar.")
    elif rho >= 0.3:
        print("   Sinal fraco. Pode ser ruído com poucos vídeos — não decida com base nisso.")
    else:
        print("   Sem poder preditivo nesses dados. O score NÃO serve como preditor aqui.")
    print("=" * 60)


if __name__ == "__main__":
    main()
