"""test_local.py — Valida tribe_virality.py SEM GPU e SEM nilearn.

Gera dados sintéticos: um vértice->rede falso (não precisa do Destrieux pra testar a
matemática) e curvas de ativação fabricadas onde a "verdade" é conhecida — vídeos com
hook/retenção/social mais fortes DEVEM pontuar mais alto. Confirma que o pipeline
inteiro (features -> score -> backtest -> calibração) roda e é coerente.

Rode:  python3 test_local.py
"""
import numpy as np
import pandas as pd

import tribe_virality as tv

rng = np.random.default_rng(42)

N_VERTS = tv.N_VERTS_CORTEX  # 20484
T = 60                       # ~60 timesteps fMRI


def fake_vertex_networks() -> np.ndarray:
    """Distribui os 20484 vértices entre as 6 redes (resto = 'other'), só p/ teste."""
    vn = np.array(["other"] * N_VERTS, dtype=object)
    idx = rng.permutation(N_VERTS)
    chunk = 2000
    for i, net in enumerate(tv.NETWORKS):
        vn[idx[i * chunk:(i + 1) * chunk]] = net
    return vn.astype(str)


def make_video(quality: float, vn: np.ndarray) -> np.ndarray:
    """Fabrica preds [T, V]. `quality` em [0,1] controla hook, retenção e social.

    Vídeo de alta qualidade: abertura forte (hook), atenção que sustenta (retenção),
    e um pico social no meio (engajamento). Assim sabemos a ordem-verdade.
    """
    preds = rng.normal(0.2, 0.02, size=(T, N_VERTS))
    t = np.arange(T)

    # base de atenção: alta qualidade sustenta; baixa qualidade decai
    att_curve = 0.3 + 0.2 * quality - (1 - quality) * 0.004 * t
    # hook: pico de atenção/saliência/visual nos primeiros ~12 timesteps
    hook_bump = np.zeros(T)
    hook_bump[:12] = 0.25 * quality
    # social: pico ("plot twist") por volta de t=30
    soc_bump = np.exp(-((t - 30) ** 2) / 8.0) * (0.4 * quality)

    for net in tv.NETWORKS:
        mask = vn == net
        if net == "attention":
            preds[:, mask] += (att_curve + hook_bump)[:, None]
        elif net in ("salience", "visual"):
            preds[:, mask] += hook_bump[:, None]
        elif net == "social_tpj":
            preds[:, mask] += soc_bump[:, None]
        else:
            preds[:, mask] += 0.1 * quality
    return preds


def main() -> None:
    vn = fake_vertex_networks()
    # 8 vídeos com qualidade-verdade crescente
    qualities = {f"vid_{i}": q for i, q in enumerate(np.linspace(0.1, 0.95, 8))}

    per_video = {}
    for vid, q in qualities.items():
        preds = make_video(q, vn)
        roi = tv.roi_timeseries(preds, vn)
        per_video[vid] = tv.neuro_features(roi)

    feats = tv.features_dataframe(per_video)
    scored = tv.score_videos(feats)
    print("=== Features + NEURO SCORE (ordenado) ===")
    print(scored[["hook_power", "retention_power", "engagement_trigger",
                  "neuro_score"]].round(3).to_string())

    # checagem de sanidade: score deve correlacionar com a qualidade-verdade
    truth = pd.Series(qualities)
    from scipy.stats import spearmanr
    rho_truth, _ = spearmanr(scored["neuro_score"], truth.loc[scored.index])
    print(f"\nSpearman(score, qualidade-verdade) = {rho_truth:.3f} "
          f"(esperado alto, perto de 1.0)")
    assert rho_truth > 0.8, "Score NÃO segue a qualidade-verdade — bug na lógica!"

    # backtest sintético: performance real = qualidade + ruído
    actual = truth + rng.normal(0, 0.08, size=len(truth))
    metrics = pd.DataFrame({"video_id": actual.index, "retention_pct": actual.values})
    bt = tv.backtest(scored, metrics, actual_col="retention_pct")
    print(f"\n=== Backtest (sintético) ===")
    print(f"n={bt['n']}  rho={bt['spearman_rho']:.3f}  "
          f"top-{bt['k']} hit rate={bt['top_k_hit_rate']:.2f}  -> {bt['interpretacao']}")
    print("feature mais correlacionada:",
          next(iter(bt["per_feature"].items())))

    # calibração: com n=8 deve RECUSAR (n<15) — comportamento honesto esperado
    cal = tv.calibrate_weights(scored, metrics, actual_col="retention_pct")
    print(f"\n=== Calibração (n=8) ===")
    print("ok:", cal["ok"], "->", cal.get("msg", "(calibrou)"))
    assert cal["ok"] is False, "Calibração deveria recusar com n<15!"

    # e com n=20 (sintético) deve calibrar
    q20 = {f"v{i}": q for i, q in enumerate(np.linspace(0.1, 0.95, 20))}
    pv20 = {vid: tv.neuro_features(tv.roi_timeseries(make_video(q, vn), vn))
            for vid, q in q20.items()}
    f20 = tv.score_videos(tv.features_dataframe(pv20))
    t20 = pd.Series(q20)
    a20 = t20 + rng.normal(0, 0.08, size=len(t20))
    m20 = pd.DataFrame({"video_id": a20.index, "retention_pct": a20.values})
    cal20 = tv.calibrate_weights(f20, m20, actual_col="retention_pct")
    print(f"calibração n=20: ok={cal20['ok']}  "
          f"LOO rho={cal20['loo_spearman_rho']:.3f}  -> {cal20['interpretacao']}")
    assert cal20["ok"] is True, "Calibração deveria rodar com n=20!"

    print("\n>>> NÚCLEO OK: features, score, backtest e calibração coerentes. <<<")


if __name__ == "__main__":
    main()
