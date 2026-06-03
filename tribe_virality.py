"""tribe_virality.py — Camada de ANÁLISE PURA (sem GPU) do pipeline TRIBE v2.

Este módulo NÃO roda o TRIBE. Ele recebe a saída do TRIBE (a matriz de ativação
cortical prevista, `preds [T, V]`) e a transforma em:

    1. curvas de ativação por rede funcional (visual, auditiva, linguagem,
       social/TPJ, atenção, saliência) -> `roi_timeseries`
    2. features neurais interpretáveis (hook / retenção / engajamento) -> `neuro_features`
    3. um NEURO CONTENT SCORE 0-10 heurístico -> `score_videos`
    4. um BACKTEST contra a performance real (Spearman, top-k) -> `backtest`
    5. calibração leave-one-out dos pesos (só com n suficiente) -> `calibrate_weights`

LEITURA HONESTA (não viole — ver CLAUDE.md):
- O score é uma CAMADA DE INTERPRETAÇÃO. Os pesos em PRIOR_WEIGHTS são um chute
  informado, NÃO calibração. O score só vira "preditor" depois que `backtest` provar
  correlação contra os dados reais do Pedro.
- Tudo aqui é RELATIVO dentro de um lote de vídeos (triagem), não absoluto.

PONTO FRÁGIL #1 (ver CLAUDE.md item 6): o alinhamento vértice->rede depende de
`preds.shape[1]` ser exatamente o que esperamos (20484 vértices corticais fsaverage5,
LH 10242 + RH 10242). Este módulo FALHA ALTO se o número não bater, em vez de produzir
lixo silencioso. Confirme o número real rodando o smoke_test.py no pod.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# 0. Constantes de geometria fsaverage5
# ---------------------------------------------------------------------------
N_VERTS_PER_HEMI = 10242            # fsaverage5: 10242 vértices por hemisfério
N_VERTS_CORTEX = 2 * N_VERTS_PER_HEMI  # 20484 corticais (LH + RH)

# As redes funcionais que vamos extrair. A ordem aqui é a ordem canônica usada
# em todo o pipeline (features, score, relatório).
NETWORKS = ["visual", "auditory", "language", "social_tpj", "attention", "salience"]

# ---------------------------------------------------------------------------
# 1. Vértice -> rede funcional (via atlas Destrieux do fsaverage5)
# ---------------------------------------------------------------------------
# Mapeamento de substrings de rótulos Destrieux (aparc.a2009s) -> rede funcional.
# Cada vértice recebe um rótulo anatômico do Destrieux; classificamos esse rótulo
# numa das nossas redes. Vértices não mapeados (medial wall, "Unknown", regiões
# motoras puras etc.) ficam como "other" e são ignorados nas features.
#
# Justificativa neurocientífica resumida (o que cada rede deveria capturar):
#   visual      -> processamento da imagem/cena   (córtex occipital)
#   auditory    -> som/voz/música                  (giro temporal superior, Heschl)
#   language    -> fala/semântica                  (Broca: frontal inferior; STS)
#   social_tpj  -> "teoria da mente"/compartilhar  (junção temporo-parietal, angular)
#   attention   -> atenção espacial/foco           (sulco intraparietal, parietal sup)
#   salience    -> "isso importa"/surpresa         (ínsula anterior, cingulado anterior)
DESTRIEUX_RULES = {
    "visual": [
        "occipital", "Pole_occipital", "S_calcarine", "G_cuneus", "S_oc_",
        "G_oc-temp_med-Lingual", "G_and_S_occipital_inf", "S_oc-temp",
        "G_oc-temp_lat-fusifor",
    ],
    "auditory": [
        "G_temp_sup-G_T_transv",       # giro de Heschl (córtex auditivo primário)
        "S_temporal_transverse",
        "G_temp_sup-Lateral",
    ],
    "language": [
        "G_front_inf-Triangul",        # Broca (pars triangularis)
        "G_front_inf-Opercular",       # Broca (pars opercularis)
        "S_front_inf",
        "G_temp_sup-Plan_tempo",       # planum temporale
    ],
    "social_tpj": [
        "G_pariet_inf-Angular",        # giro angular (núcleo do TPJ)
        "G_pariet_inf-Supramar",       # supramarginal
        "S_temporal_sup",              # sulco temporal superior posterior
    ],
    "attention": [
        "S_intrapariet_and_P_trans",   # sulco intraparietal (atenção dorsal)
        "G_parietal_sup",
        "S_precentral-sup-part",       # frontal eye fields aproximado
        "G_precuneus",
    ],
    "salience": [
        "G_Ins_lg_and_S_cent_ins",     # ínsula
        "S_circular_insula_ant",       # ínsula anterior
        "G_insular_short",
        "G_and_S_cingul-Ant",          # cingulado anterior
        "G_and_S_cingul-Mid-Ant",
    ],
}


def _label_to_network(label_name: str) -> str:
    """Classifica um rótulo anatômico Destrieux numa rede funcional (ou 'other')."""
    for network, patterns in DESTRIEUX_RULES.items():
        for pat in patterns:
            if pat in label_name:
                return network
    return "other"


def build_vertex_networks(cache_dir: str | None = None) -> np.ndarray:
    """Constrói o array vértice->rede para os 20484 vértices corticais do fsaverage5.

    Usa o atlas Destrieux do nilearn. Retorna um array de strings de tamanho 20484,
    onde cada entrada é o nome da rede (ou 'other').

    Requer nilearn instalado (vem no setup.sh). É a ÚNICA função deste módulo que
    depende de download/atlas; o resto é numpy puro e testável offline.
    """
    try:
        from nilearn import datasets
    except ImportError as e:  # falha cedo e clara (padrão do projeto)
        raise ImportError(
            "nilearn não está instalado. Rode `pip install nilearn` (ou o setup.sh)."
        ) from e

    destrieux = datasets.fetch_atlas_surf_destrieux(data_dir=cache_dir)
    # rótulos vêm como bytes em algumas versões do nilearn; normaliza para str
    labels = [l.decode() if isinstance(l, bytes) else str(l) for l in destrieux["labels"]]

    map_left = np.asarray(destrieux["map_left"]).ravel()    # [10242] índices em labels
    map_right = np.asarray(destrieux["map_right"]).ravel()  # [10242]

    if map_left.shape[0] != N_VERTS_PER_HEMI or map_right.shape[0] != N_VERTS_PER_HEMI:
        raise ValueError(
            f"Atlas Destrieux com tamanho inesperado: LH={map_left.shape[0]}, "
            f"RH={map_right.shape[0]} (esperado {N_VERTS_PER_HEMI} cada). "
            "Versão do nilearn/atlas diferente — verifique antes de prosseguir."
        )

    parcels = np.concatenate([map_left, map_right])  # [20484], ordem LH depois RH
    vertex_networks = np.array([_label_to_network(labels[i]) for i in parcels])
    return vertex_networks


def align_preds_to_cortex(preds: np.ndarray) -> np.ndarray:
    """Garante que `preds` tenha exatamente os 20484 vértices corticais, na ordem certa.

    ESTE É O GUARDA DO PONTO FRÁGIL #1. Trata os casos conhecidos e FALHA ALTO no
    caso perigoso (medial wall mascarada), porque aí não dá pra alinhar sem saber
    quais vértices foram removidos.

    - preds [T, 20484]  -> ok, retorna como está.
    - preds [T, >20484] -> assume que os 20484 primeiros são o córtex fsaverage5
                           (corticais antes de subcorticais) e fatia, com aviso.
    - preds [T, <20484] -> PROVÁVEL medial wall mascarada. ERRO: precisamos do índice
                           dos vértices válidos para alinhar o atlas (ver CLAUDE.md #6).
    """
    if preds.ndim != 2:
        raise ValueError(f"preds deveria ser 2D [T, V], veio shape={preds.shape}")

    V = preds.shape[1]
    if V == N_VERTS_CORTEX:
        return preds
    if V > N_VERTS_CORTEX:
        print(
            f"[ALERTA align] preds tem {V} colunas (>{N_VERTS_CORTEX}). Assumindo que "
            f"os {N_VERTS_CORTEX} primeiros são o córtex fsaverage5 e descartando o "
            f"resto (provável subcortical). CONFIRME isso contra utils_fmri.py do repo."
        )
        return preds[:, :N_VERTS_CORTEX]
    raise ValueError(
        f"preds tem {V} colunas (<{N_VERTS_CORTEX}). PERIGO: provável medial wall "
        f"mascarada. NÃO dá pra alinhar o atlas assumindo o concat cru — o mapeamento "
        f"rede<->vértice ficaria errado e silencioso. Descubra o índice dos vértices "
        f"válidos (utils_fmri.py / tribe_demo.ipynb) e passe um vertex_networks já "
        f"subsetado para roi_timeseries. Ver CLAUDE.md item 6."
    )


# ---------------------------------------------------------------------------
# 2. preds -> curvas por rede
# ---------------------------------------------------------------------------
def roi_timeseries(preds: np.ndarray, vertex_networks: np.ndarray) -> pd.DataFrame:
    """Reduz preds [T, V] a uma curva temporal média por rede funcional.

    Retorna DataFrame [T, n_networks] com uma coluna por rede em NETWORKS.
    """
    preds = align_preds_to_cortex(preds)
    if vertex_networks.shape[0] != preds.shape[1]:
        raise ValueError(
            f"Tamanho de vertex_networks ({vertex_networks.shape[0]}) != número de "
            f"vértices em preds ({preds.shape[1]}). Eles têm que casar 1:1."
        )

    out = {}
    for net in NETWORKS:
        mask = vertex_networks == net
        if mask.sum() == 0:
            raise ValueError(
                f"Nenhum vértice mapeado para a rede '{net}'. O mapeamento Destrieux "
                f"provavelmente quebrou — não confie nos resultados."
            )
        out[net] = preds[:, mask].mean(axis=1)  # média dos vértices da rede a cada t
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# 3. curvas -> features neurais
# ---------------------------------------------------------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0, 0.0) else 0.0


def neuro_features(roi: pd.DataFrame, hook_frac: float = 0.2) -> dict:
    """Extrai features interpretáveis de uma curva-por-rede (um vídeo).

    As três features-âncora seguem a narrativa do reel de referência:
      - hook_power        : quão forte a abertura "acende" atenção/saliência/visual,
                            relativo à média do vídeo. (prende nos primeiros segundos)
      - retention_power   : quão bem a atenção SUSTENTA (penaliza decaimento da 1ª
                            metade para a 2ª). Alto = não cansa.
      - engagement_trigger: picos da rede social/TPJ acima da própria média
                            (momentos de "plot twist"/compartilhamento).
    Mais médias e picos por rede, para o backtest por-feature.

    Todas relativas dentro do próprio vídeo (robusto a escala absoluta do TRIBE).
    """
    T = len(roi)
    if T < 3:
        raise ValueError(f"Série temporal curta demais (T={T}). Vídeo muito curto?")

    n_hook = max(1, int(round(T * hook_frac)))
    feats: dict[str, float] = {}

    # médias e picos por rede (úteis no backtest por-feature)
    for net in NETWORKS:
        c = roi[net].to_numpy()
        feats[f"{net}_mean"] = float(c.mean())
        feats[f"{net}_peak"] = float(c.max())

    whole_mean = roi[NETWORKS].to_numpy().mean()

    # hook_power: ativação de atenção+saliência+visual na abertura vs média global
    hook_nets = ["attention", "salience", "visual"]
    hook_open = roi[hook_nets].iloc[:n_hook].to_numpy().mean()
    feats["hook_power"] = _safe_div(hook_open, whole_mean)

    # retention_power: 1 - decaimento da atenção (1ª metade -> 2ª metade)
    att = roi["attention"].to_numpy()
    half = T // 2
    first, second = att[:half].mean(), att[half:].mean()
    decay = _safe_div(first - second, first)        # >0 = caiu; <0 = subiu
    feats["retention_power"] = 1.0 - decay          # 1.0 = sustentou; >1 = cresceu

    # engagement_trigger: pico social/TPJ acima da sua própria média (spikiness)
    soc = roi["social_tpj"].to_numpy()
    feats["engagement_trigger"] = _safe_div(soc.max() - soc.mean(), soc.mean())

    return feats


def features_dataframe(per_video: dict[str, dict]) -> pd.DataFrame:
    """Empilha features de vários vídeos: {video_id: features_dict} -> DataFrame.

    Index = video_id; uma linha por vídeo, uma coluna por feature.
    """
    df = pd.DataFrame.from_dict(per_video, orient="index")
    df.index.name = "video_id"
    return df


# ---------------------------------------------------------------------------
# 4. features -> NEURO CONTENT SCORE (0-10), heurístico
# ---------------------------------------------------------------------------
# Pesos PRIOR (chute informado, NÃO calibrado). O backtest é quem valida.
PRIOR_WEIGHTS = {
    "hook_power": 0.35,
    "retention_power": 0.40,
    "engagement_trigger": 0.25,
}


def _minmax_to_10(x: np.ndarray) -> np.ndarray:
    """Escala um vetor para 0-10 dentro do lote (triagem RELATIVA)."""
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.full_like(x, 5.0)  # todos iguais -> meio da escala
    return 10.0 * (x - lo) / (hi - lo)


def score_videos(features_df: pd.DataFrame, weights: dict | None = None) -> pd.DataFrame:
    """Calcula o NEURO CONTENT SCORE 0-10 (relativo ao lote) a partir das features.

    Z-score cada feature-âncora dentro do lote, combina pelos pesos, escala 0-10.
    Retorna o features_df com colunas extras: z-scores, composite e `neuro_score`.
    """
    weights = weights or PRIOR_WEIGHTS
    out = features_df.copy()

    composite = np.zeros(len(out), dtype=float)
    for feat, w in weights.items():
        if feat not in out.columns:
            raise KeyError(f"Feature '{feat}' não está no features_df.")
        col = out[feat].to_numpy(dtype=float)
        mu, sd = col.mean(), col.std()
        z = (col - mu) / sd if sd > 1e-12 else np.zeros_like(col)
        out[f"z_{feat}"] = z
        composite += w * z

    out["composite"] = composite
    out["neuro_score"] = _minmax_to_10(composite)
    return out.sort_values("neuro_score", ascending=False)


# ---------------------------------------------------------------------------
# 5. BACKTEST — score previsto vs performance real
# ---------------------------------------------------------------------------
def interpret_rho(rho: float) -> str:
    """Régua honesta de interpretação (ver CLAUDE.md item 7)."""
    a = abs(rho)
    if a >= 0.6:
        return "promissor (≥0.6)"
    if a >= 0.3:
        return "sinal fraco (0.3–0.6)"
    return "não serve como preditor (<0.3)"


def backtest(
    features_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    actual_col: str,
    score_col: str = "neuro_score",
    k: int | None = None,
) -> dict:
    """Compara o score previsto com a performance REAL.

    features_df: index=video_id, precisa conter `score_col` e as features-âncora.
    metrics_df : index=video_id (ou coluna 'video_id'), contém `actual_col`.

    Retorna dict com: n, spearman_rho, p_value, interpretacao, top_k, top_k_hit_rate,
    per_feature (Spearman de cada feature vs real), e o DataFrame alinhado.
    """
    m = metrics_df.copy()
    if "video_id" in m.columns:
        m = m.set_index("video_id")

    joined = features_df.join(m[[actual_col]], how="inner")
    joined = joined.dropna(subset=[score_col, actual_col])
    n = len(joined)
    if n < 3:
        raise ValueError(
            f"Só {n} vídeos com score E métrica real casados. Precisa de pelo menos 3 "
            f"(e idealmente ≥15) pro backtest fazer sentido. Confira os video_id."
        )

    rho, p = spearmanr(joined[score_col], joined[actual_col])

    if k is None:
        k = max(1, n // 3)  # default: terço superior
    top_pred = set(joined.nlargest(k, score_col).index)
    top_real = set(joined.nlargest(k, actual_col).index)
    hit = len(top_pred & top_real)

    # correlação de CADA feature-âncora + médias por rede contra o real
    per_feature = {}
    feature_cols = list(PRIOR_WEIGHTS.keys()) + [f"{net}_mean" for net in NETWORKS]
    for col in feature_cols:
        if col in joined.columns:
            fr, fp = spearmanr(joined[col], joined[actual_col])
            per_feature[col] = {"rho": float(fr), "p": float(fp)}
    # ordena features pela força absoluta da correlação
    per_feature = dict(
        sorted(per_feature.items(), key=lambda kv: -abs(kv[1]["rho"]))
    )

    if n < 15:
        print(
            f"[ATENÇÃO backtest] n={n} (<15). Reporte a correlação, mas trate como "
            f"FRÁGIL — ver CLAUDE.md item 1. Não apresente como acurácia."
        )

    return {
        "n": n,
        "actual_col": actual_col,
        "spearman_rho": float(rho),
        "p_value": float(p),
        "interpretacao": interpret_rho(rho),
        "k": k,
        "top_k_pred": sorted(top_pred),
        "top_k_real": sorted(top_real),
        "top_k_hits": hit,
        "top_k_hit_rate": hit / k,
        "per_feature": per_feature,
        "joined": joined[[score_col, actual_col]],
    }


# ---------------------------------------------------------------------------
# 6. Calibração leave-one-out (só com n suficiente)
# ---------------------------------------------------------------------------
def calibrate_weights(
    features_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    actual_col: str,
    feature_cols: list[str] | None = None,
    min_n: int = 15,
) -> dict:
    """Refit dos pesos das features por regressão linear, validado leave-one-out.

    Para cada vídeo i: treina uma regressão (features padronizadas -> real) em TODOS
    os outros vídeos, prevê i, coleta a previsão. Mede Spearman das previsões LOO vs
    real — isso estima poder preditivo SEM o classificador "decorar" os próprios
    vídeos (ver CLAUDE.md item 1).

    Retorna dict com: pesos médios, loo_spearman, e as previsões LOO.
    Recusa (avisa) se n < min_n.
    """
    m = metrics_df.copy()
    if "video_id" in m.columns:
        m = m.set_index("video_id")
    joined = features_df.join(m[[actual_col]], how="inner").dropna(subset=[actual_col])
    n = len(joined)

    if n < min_n:
        return {
            "ok": False,
            "n": n,
            "msg": (
                f"n={n} < {min_n}. Calibração recusada: com poucos vídeos o refit "
                f"superajusta (decora). Use os PRIOR_WEIGHTS e o backtest até ter mais "
                f"dados. Ver CLAUDE.md item 1 e 7."
            ),
        }

    feature_cols = feature_cols or list(PRIOR_WEIGHTS.keys())
    X = joined[feature_cols].to_numpy(dtype=float)
    y = joined[actual_col].to_numpy(dtype=float)

    # padroniza features (z-score) com estatísticas do conjunto inteiro só p/ reportar
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)

    loo_pred = np.zeros(n)
    coefs = []
    for i in range(n):
        train = np.ones(n, dtype=bool)
        train[i] = False
        Xt, yt = Xz[train], y[train]
        # regressão linear com intercepto via lstsq
        A = np.hstack([np.ones((Xt.shape[0], 1)), Xt])
        beta, *_ = np.linalg.lstsq(A, yt, rcond=None)
        coefs.append(beta[1:])
        loo_pred[i] = beta[0] + Xz[i] @ beta[1:]

    rho, p = spearmanr(loo_pred, y)
    mean_coef = np.mean(coefs, axis=0)
    weights = {c: float(w) for c, w in zip(feature_cols, mean_coef)}

    return {
        "ok": True,
        "n": n,
        "feature_cols": feature_cols,
        "weights_mean": weights,
        "loo_spearman_rho": float(rho),
        "loo_p_value": float(p),
        "interpretacao": interpret_rho(rho),
        "loo_predictions": dict(zip(joined.index, loo_pred)),
    }
